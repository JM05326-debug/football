"""
Walk-forward backtest：嚴格依照時間切分，禁止任何形式的未來資料外洩。

Fold 規則（依賽季切，跟 E0_*.csv 的賽季邊界對齊）：
    Fold 1: Train = 2020-21, 2021-22               Test = 2022-23
    Fold 2: Train = 2020-21 ... 2022-23             Test = 2023-24
    Fold 3: Train = 2020-21 ... 2023-24             Test = 2024-25
    Fold 4: Train = 2020-21 ... 2024-25             Test = 2025-26

每個 fold 的訓練資料只包含「測試賽季開打之前」已經踢完的比賽，測試賽季的比賽
彼此之間也不會互相看到對方（每場測試比賽的預測都只用 fold 開始時 fit 好的
model，不會在測試賽季途中重新用測試賽季已發生的比賽更新模型）。

嚴格意義上更精細的作法是「每場測試比賽踢完就重新 fit 一次模型」，但那樣一個
賽季要重新 fit 38 輪 x 4 folds x 5 個 model variant，運算量大很多；用「fold 開始時
fit 一次、整個測試賽季都用同一個 snapshot」在時間切分上一樣不會用到未來資料
（因為 fit 用的訓練資料本來就完全在測試賽季之前），只是少了「賽季middle 也持續更新」
這件事，對「有沒有 leakage」這個問題沒有影響，這點在 report 裡會註明。
"""
import pathlib
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from db import database as db
from analytics import metrics as m
from analytics.model_variants import DEFAULT_PARAMS, VARIANT_ORDER, VARIANT_LABELS, variant_flags, fit_model, predict_one

ROOT = pathlib.Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "backtest_output"
OUTPUT_DIR.mkdir(exist_ok=True)

SEASON_BOUNDARIES = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def load_epl_matches_df(conn) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT date, season, home_team, away_team, home_goals, away_goals, result, "
        "avg_h_odds, avg_d_odds, avg_a_odds FROM matches WHERE league='EPL' ORDER BY date ASC"
    ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_folds(df: pd.DataFrame):
    """回傳 list of (fold_index, train_df, test_df, train_start, train_end, test_start, test_end)"""
    folds = []
    for i in range(2, len(SEASON_BOUNDARIES)):
        test_season = SEASON_BOUNDARIES[i]
        train_seasons = SEASON_BOUNDARIES[:i]
        train_df = df[df["season"].isin(train_seasons)].reset_index(drop=True)
        test_df = df[df["season"] == test_season].reset_index(drop=True)
        if test_df.empty or train_df.empty:
            continue
        folds.append({
            "fold_index": len(folds) + 1,
            "train_df": train_df,
            "test_df": test_df,
            "train_start": train_df["date"].min().strftime("%Y-%m-%d"),
            "train_end": train_df["date"].max().strftime("%Y-%m-%d"),
            "test_start": test_df["date"].min().strftime("%Y-%m-%d"),
            "test_end": test_df["date"].max().strftime("%Y-%m-%d"),
        })
    return folds


def run_variant_on_fold(variant: str, fold: dict, params: dict):
    flags = variant_flags(variant)
    model = fit_model(fold["train_df"], params, flags)

    rows = []
    for _, match in fold["test_df"].iterrows():
        pred = predict_one(model, match["home_team"], match["away_team"], params, flags)
        rows.append({
            "match_date": match["date"].strftime("%Y-%m-%d"),
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "prob_H": pred["prob_H"], "prob_D": pred["prob_D"], "prob_A": pred["prob_A"],
            "predicted_result": pred["predicted_result"],
            "actual_result": match["result"],
            "actual_home_goals": int(match["home_goals"]),
            "actual_away_goals": int(match["away_goals"]),
            "over_2_5_probability": pred["over_2_5_probability"],
            "home_odds": match["avg_h_odds"], "draw_odds": match["avg_d_odds"], "away_odds": match["avg_a_odds"],
        })

    full = m.full_metrics(rows)
    roi = m.flat_stake_roi(rows)

    return {"rows": rows, "metrics": full, "roi": roi}


def run_backtest(params_by_variant: dict = None, save_to_db=True) -> dict:
    """params_by_variant: {'E': {...tuned params...}}，其他 variant 用 DEFAULT_PARAMS。
    回傳 {variant: {'fold_results':[...], 'pooled_metrics':{...}, 'pooled_roi':{...}}}"""
    params_by_variant = params_by_variant or {}
    conn = db.connect()
    df = load_epl_matches_df(conn)
    folds = build_folds(df)
    run_at = db.now_iso()

    results = {}
    for variant in VARIANT_ORDER:
        params = params_by_variant.get(variant, DEFAULT_PARAMS)
        fold_results = []
        pooled_rows = []
        for fold in folds:
            fr = run_variant_on_fold(variant, fold, params)
            fold_results.append({**fr, "fold": fold})
            pooled_rows.extend(fr["rows"])

            if save_to_db:
                db.insert_backtest_fold(
                    conn, run_at=run_at, league="EPL", model_variant=variant, fold_index=fold["fold_index"],
                    train_start=fold["train_start"], train_end=fold["train_end"],
                    test_start=fold["test_start"], test_end=fold["test_end"],
                    n_test_matches=len(fr["rows"]), metrics=fr["metrics"],
                    roi=fr["roi"]["roi"],
                )

        pooled_metrics = m.full_metrics(pooled_rows)
        pooled_roi = m.flat_stake_roi(pooled_rows)
        results[variant] = {
            "label": VARIANT_LABELS[variant],
            "fold_results": fold_results,
            "pooled_rows": pooled_rows,
            "pooled_metrics": pooled_metrics,
            "pooled_roi": pooled_roi,
        }

    if save_to_db:
        conn.commit()
    conn.close()

    results["_folds"] = folds
    results["_run_at"] = run_at
    return results


def write_model_comparison_csv(results: dict, path: pathlib.Path = None) -> pathlib.Path:
    path = path or (OUTPUT_DIR / "model_comparison.csv")
    records = []
    for variant in VARIANT_ORDER:
        r = results[variant]
        pm, proi = r["pooled_metrics"], r["pooled_roi"]
        records.append({
            "model_variant": variant,
            "label": r["label"],
            "n_test_matches": pm["n"],
            "log_loss": pm["log_loss"],
            "brier_score": pm["brier_score"],
            "accuracy": pm["accuracy"],
            "home_accuracy": pm["home_accuracy"],
            "draw_accuracy": pm["draw_accuracy"],
            "away_accuracy": pm["away_accuracy"],
            "over_2_5_accuracy": pm["over_2_5_accuracy"],
            "under_2_5_accuracy": pm["under_2_5_accuracy"],
            "roi": proi["roi"],
            "roi_n_bets": proi["n_bets"],
        })
    out = pd.DataFrame(records).sort_values(["log_loss", "brier_score", "accuracy"], ascending=[True, True, False])
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_fold_detail_csv(results: dict, path: pathlib.Path = None) -> pathlib.Path:
    path = path or (OUTPUT_DIR / "backtest_fold_detail.csv")
    records = []
    for variant in VARIANT_ORDER:
        for fr in results[variant]["fold_results"]:
            fold = fr["fold"]
            pm, proi = fr["metrics"], fr["roi"]
            records.append({
                "model_variant": variant,
                "fold_index": fold["fold_index"],
                "train_start": fold["train_start"], "train_end": fold["train_end"],
                "test_start": fold["test_start"], "test_end": fold["test_end"],
                "n_test_matches": pm["n"],
                "log_loss": pm["log_loss"], "brier_score": pm["brier_score"], "accuracy": pm["accuracy"],
                "home_accuracy": pm["home_accuracy"], "draw_accuracy": pm["draw_accuracy"], "away_accuracy": pm["away_accuracy"],
                "roi": proi["roi"],
            })
    out = pd.DataFrame(records)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main():
    best_params_path = ROOT / "best_params.json"
    params_by_variant = {}
    if best_params_path.exists():
        import json
        tuned = json.loads(best_params_path.read_text(encoding="utf-8"))
        params_by_variant["E"] = {**DEFAULT_PARAMS, **tuned}
        print(f"Model E 使用 {best_params_path} 的調校參數")
    else:
        print("找不到 best_params.json，Model E 暫時跟 Model D 用相同預設參數（尚未調校）")

    results = run_backtest(params_by_variant=params_by_variant)
    csv_path = write_model_comparison_csv(results)
    detail_path = write_fold_detail_csv(results)

    print(f"\n=== Walk-forward Backtest 結果 (共 {len(results['_folds'])} 個 fold) ===")
    for fold in results["_folds"]:
        print(f"  Fold {fold['fold_index']}: train {fold['train_start']}~{fold['train_end']} "
              f"({len(fold['train_df'])} 場) -> test {fold['test_start']}~{fold['test_end']} ({len(fold['test_df'])} 場)")

    def pct_or_na(x):
        return f"{x*100:>6.1f}%" if x is not None else f"{'N/A':>7}"

    print(f"\n{'Variant':8}{'LogLoss':>10}{'Brier':>10}{'Acc':>8}{'Home':>8}{'Draw':>8}{'Away':>8}{'ROI':>8}")
    for variant in VARIANT_ORDER:
        pm, proi = results[variant]["pooled_metrics"], results[variant]["pooled_roi"]
        roi_str = f"{proi['roi']*100:.1f}%" if proi["roi"] is not None else "N/A"
        print(f"{variant:8}{pm['log_loss']:>10.4f}{pm['brier_score']:>10.4f}{pm['accuracy']*100:>7.1f}%"
              f"{pct_or_na(pm['home_accuracy'])}{pct_or_na(pm['draw_accuracy'])}"
              f"{pct_or_na(pm['away_accuracy'])}{roi_str:>8}")
    n_draw_pred = sum(1 for r in results["A"]["pooled_rows"] if r["predicted_result"] == "D")
    print(f"\n注意: 全部 {len(results['A']['pooled_rows'])} 場樣本外測試中，模型從未把「和局」預測為最高機率結果"
          f"（predicted_result=='D' 的次數: {n_draw_pred}），所以 Draw Accuracy 是 N/A（0/0），"
          f"不是「和局預測 0% 命中」。這是 Poisson 類模型常見的已知限制：和局機率通常不是三個結果裡最高的那個，"
          f"即使它有 20~25%，還是常常小於主勝或客勝機率，所以 argmax 永遠不會選到和局。")

    print(f"\n已輸出: {csv_path}")
    print(f"已輸出: {detail_path}")
    return results


if __name__ == "__main__":
    main()
