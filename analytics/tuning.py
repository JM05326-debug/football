"""
超參數搜尋，validation-only，絕對不碰最後一個 fold（2025-26 賽季）。

做法: coordinate / greedy search，不是完整 grid search —— 每個參數各自試幾個候選值，
固定其他參數，挑出讓「驗證集 log loss」最低的那個值，然後換下一個參數，用剛選出的
值繼續。這是完整 grid search（笛卡兒積會有幾百到上千種組合）在計算量上的合理簡化，
在這裡誠實說明，不假裝是窮舉搜尋。

驗證用的 walk-forward 範圍只到 fold 1~3（train 累積到 2024-25 賽季、test 分別是
2022-23 / 2023-24 / 2024-25），完全不使用 fold 4（test=2025-26）。fold 4 保留給
walkforward.py 最後產生 model_comparison.csv 時，當作 Model E 真正沒看過的樣本外測試。
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from db import database as db
from analytics import metrics as m
from analytics.model_variants import DEFAULT_PARAMS, variant_flags
from analytics.walkforward import load_epl_matches_df, build_folds, run_variant_on_fold

ROOT = pathlib.Path(__file__).parent.parent

# 每個參數的候選值（刻意選擇有物理意義的範圍，不是隨機亂猜）
PARAM_GRID = {
    "half_life_days": [180.0, 380.0, 730.0],   # 半年 / 一個賽季 / 兩年
    "prior_games": [2.0, 4.0, 8.0],            # shrinkage 強度
    "form_weight": [0.0, 0.3, 0.6],            # 近期 10 場手感權重
    "elo_k": [10.0, 20.0, 30.0],
    "elo_home_adv": [50.0, 100.0, 150.0],
    "elo_weight": [0.3, 0.5, 0.7],             # Elo 跟 Poisson 機率混合比重
}

TUNING_VARIANT_FLAGS = variant_flags("D")  # 完整功能集合（decay+form+elo），只調超參數


def validation_log_loss(params: dict, validation_folds: list) -> float:
    pooled_rows = []
    for fold in validation_folds:
        model = None
        from analytics.model_variants import fit_model, predict_one
        model = fit_model(fold["train_df"], params, TUNING_VARIANT_FLAGS)
        for _, match in fold["test_df"].iterrows():
            pred = predict_one(model, match["home_team"], match["away_team"], params, TUNING_VARIANT_FLAGS)
            pooled_rows.append({
                "prob_H": pred["prob_H"], "prob_D": pred["prob_D"], "prob_A": pred["prob_A"],
                "actual_result": match["result"],
            })
    return m.log_loss(pooled_rows)


def greedy_search(validation_folds: list, base_params: dict = None):
    current = dict(base_params or DEFAULT_PARAMS)
    baseline_ll = validation_log_loss(current, validation_folds)
    history = [{"param": None, "value": None, "log_loss": baseline_ll, "note": "baseline (預設參數)"}]

    print(f"Baseline (預設參數) 驗證集 log loss: {baseline_ll:.4f}")

    for param, candidates in PARAM_GRID.items():
        best_val, best_ll = current[param], validation_log_loss(current, validation_folds)
        for val in candidates:
            trial = dict(current)
            trial[param] = val
            ll = validation_log_loss(trial, validation_folds)
            history.append({"param": param, "value": val, "log_loss": ll, "note": ""})
            print(f"  {param}={val}: val log_loss={ll:.4f}")
            if ll < best_ll:
                best_ll, best_val = ll, val
        current[param] = best_val
        print(f"-> 選定 {param} = {best_val} (log_loss={best_ll:.4f})")

    final_ll = validation_log_loss(current, validation_folds)
    return current, final_ll, baseline_ll, history


def main():
    conn = db.connect()
    df = load_epl_matches_df(conn)
    conn.close()

    all_folds = build_folds(df)
    validation_folds = all_folds[:-1]   # fold 1~3；最後一個 fold (2025-26) 完全不碰
    held_out_fold = all_folds[-1]

    print(f"驗證用 fold: {[f['fold_index'] for f in validation_folds]} "
          f"(test 賽季: {[f['test_start'][:4] for f in validation_folds]})")
    print(f"完全隔離、不參與調參的最終測試 fold: {held_out_fold['fold_index']} "
          f"(test {held_out_fold['test_start']} ~ {held_out_fold['test_end']})\n")

    best_params, final_ll, baseline_ll, history = greedy_search(validation_folds)

    improvement = baseline_ll - final_ll
    print(f"\n調校後驗證集 log loss: {final_ll:.4f}（預設參數: {baseline_ll:.4f}，"
          f"{'進步' if improvement > 0 else '沒有進步'} {improvement:+.4f}）")

    output = {
        **best_params,
        "_baseline_validation_log_loss": baseline_ll,
        "_tuned_validation_log_loss": final_ll,
        "_improved": improvement > 0,
        "_validation_folds": [f["fold_index"] for f in validation_folds],
        "_held_out_test_fold": held_out_fold["fold_index"],
        "_search_method": "greedy coordinate search (見 analytics/tuning.py PARAM_GRID)",
        "_tuned_at": db.now_iso(),
    }

    if not (improvement > 0):
        print("調校沒有比預設參數好，仍然輸出 best_params.json 供參考，"
              "但 Model E 若採用它，backtest 報告會誠實顯示沒有優於 Model D。")

    out_path = ROOT / "best_params.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已輸出: {out_path}")

    history_path = ROOT / "backtest_output" / "tuning_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已輸出: {history_path}")

    return output


if __name__ == "__main__":
    main()
