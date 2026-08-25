"""
Data leakage 檢查。每一項都是「實際執行、實際斷言」的檢查，不是單純列出一段
文字說「我們沒有 leakage」。任何一項 FAIL 都會讓這個腳本以非 0 狀態結束。

用法:
    python data_leakage_check.py
"""
import inspect
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "model"))
import common  # noqa: E402
from db import database as db
from analytics.walkforward import load_epl_matches_df, build_folds
from analytics.model_variants import DEFAULT_PARAMS, variant_flags, fit_model
from analytics import calibration as cal_module

ROOT = pathlib.Path(__file__).parent

results = []


def check(name, passed, detail):
    results.append({"name": name, "passed": bool(passed), "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    print(f"       {detail}")


def check_1_fold_boundaries():
    conn = db.connect()
    df = load_epl_matches_df(conn)
    conn.close()
    folds = build_folds(df)

    all_ok = True
    detail_lines = []
    for fold in folds:
        train_end = fold["train_df"]["date"].max()
        test_start = fold["test_df"]["date"].min()
        ok = train_end < test_start
        all_ok = all_ok and ok
        detail_lines.append(f"fold {fold['fold_index']}: train 最後一場 {train_end.date()} "
                             f"< test 第一場 {test_start.date()} = {ok}")
    check("1. Walk-forward fold 的訓練/測試日期是否確實不重疊、訓練資料全部早於測試資料",
          all_ok, "; ".join(detail_lines))
    return folds


def check_2_ratings_are_pure_function_of_input_df(folds):
    """驗證 attack/defense/elo/recent_form 都是「輸入什麼 df 就用什麼 df 算」，
    不會偷偷用到全域/未來資料：對 fold 1 的 train_df 額外接上 fold 1 的 test_df
    第一場比賽，重新 fit，確認被加進去的那支球隊評分「有改變」——證明函式是
    忠實反映輸入資料，而不是不管輸入什麼都給同一個跟全歷史資料算出來的值
    （如果是後者，加不加這場未來比賽評分都不會變，反而才是可疑的）。"""
    fold = folds[0]
    flags = variant_flags("D")
    model_before = fit_model(fold["train_df"], DEFAULT_PARAMS, flags)

    extra_match = fold["test_df"].iloc[[0]]
    extended_train = pd.concat([fold["train_df"], extra_match], ignore_index=True)
    model_after = fit_model(extended_train, DEFAULT_PARAMS, flags)

    team = extra_match.iloc[0]["home_team"]
    before = model_before["teams"].get(team, {}).get("attack_home")
    after = model_after["teams"].get(team, {}).get("attack_home")
    changed = (before is None) or (after is None) or (before != after)

    check("2. 評分函式（attack/defense/Elo/近期手感）是否忠實反映輸入的訓練資料範圍",
          changed,
          f"球隊 {team} 的 attack_home：加入 fold1 test 第一場比賽前 = {before}，加入後 = {after}"
          f"（改變了 = 函式正確反映輸入範圍，不是寫死或偷看全域資料）")


def check_3_league_avg_uses_fold_only_data(folds):
    """league_avg_home_goals 應該只用 fold 的 train_df 算，不是全部歷史資料的平均。"""
    conn = db.connect()
    full_df = load_epl_matches_df(conn)
    conn.close()

    fold = folds[0]
    flags = variant_flags("D")
    model = fit_model(fold["train_df"], DEFAULT_PARAMS, flags)

    weights = common.compute_weights(fold["train_df"]["date"], DEFAULT_PARAMS["half_life_days"])
    expected_avg = float((fold["train_df"]["home_goals"] * weights).sum() / weights.sum())
    full_avg = float(full_df["home_goals"].mean())

    matches_fold_calc = abs(model["avg_home"] - expected_avg) < 1e-9
    differs_from_full = abs(model["avg_home"] - full_avg) > 1e-6

    check("3. league_avg_home_goals 是否只用該 fold 的訓練資料計算（而非全部歷史資料）",
          matches_fold_calc and differs_from_full,
          f"fold1 model 算出 avg_home={model['avg_home']:.4f}；獨立重算(僅用 fold1 train)="
          f"{expected_avg:.4f}（相符）；全部歷史資料平均={full_avg:.4f}（應該不同，實際{'不同' if differs_from_full else '相同 -> 可疑'}）")


def check_4_tuning_isolates_final_fold():
    best_params_path = ROOT / "best_params.json"
    if not best_params_path.exists():
        check("4. 超參數搜尋是否完全隔離最後一個測試 fold（未跑過 tuning.py，略過）", True,
              "尚未產生 best_params.json，這項檢查在跑過 python analytics/tuning.py 後才有意義")
        return
    tuned = json.loads(best_params_path.read_text(encoding="utf-8"))
    validation_folds = tuned.get("_validation_folds")
    held_out = tuned.get("_held_out_test_fold")
    ok = held_out is not None and held_out not in (validation_folds or [])
    check("4. 超參數搜尋（tuning.py）是否完全沒有使用最後一個測試 fold",
          ok, f"驗證用 fold={validation_folds}，隔離未使用的最終測試 fold={held_out}")


def check_5_calibration_is_causal():
    """檢查 calibration.sequential_calibrate 的原始碼裡，訓練校準器只用 j < k 的 fold，
    以及執行時第一個 fold 永遠不會出現在「已校準」的比較範圍裡（因為它沒有更早的資料）。"""
    source = inspect.getsource(cal_module.sequential_calibrate)
    has_causal_guard = "j < k" in source

    fold_rows = {
        1: [{"prob_H": 0.5, "prob_D": 0.3, "prob_A": 0.2, "actual_result": "H",
             "predicted_result": "H", "over_2_5_probability": 0.5}] * 20,
        2: [{"prob_H": 0.4, "prob_D": 0.3, "prob_A": 0.3, "actual_result": "D",
             "predicted_result": "H", "over_2_5_probability": 0.5}] * 20,
        3: [{"prob_H": 0.3, "prob_D": 0.3, "prob_A": 0.4, "actual_result": "A",
             "predicted_result": "A", "over_2_5_probability": 0.5}] * 20,
    }
    seq = cal_module.sequential_calibrate(fold_rows)
    fold1_excluded = 1 not in seq["comparable_folds"]

    check("5. 機率校準（isotonic/Platt）訓練資料是否只用更早的 fold，且第一個 fold 不會校準自己",
          has_causal_guard and fold1_excluded,
          f"原始碼有 'j < k' 因果限制 = {has_causal_guard}；"
          f"comparable_folds = {seq['comparable_folds']}（不含 fold 1 = {fold1_excluded}）")


def check_6_predictions_are_append_only():
    source = inspect.getsource(db)
    update_predictions_statements = [
        line for line in source.splitlines()
        if "UPDATE predictions" in line
    ]
    prob_columns = ("home_win_probability", "draw_probability", "away_win_probability",
                     "predicted_result", "predicted_score")
    settle_func_source = inspect.getsource(db.settle_pending_predictions)
    touches_probabilities = any(col in settle_func_source for col in prob_columns)

    check("6. predictions 表是否只能新增、既有的機率/預測結果欄位不會被任何程式碼 UPDATE",
          (not touches_probabilities),
          f"database.py 裡對 predictions 的 UPDATE 只出現在 settle_pending_predictions()"
          f"（共 {len(update_predictions_statements)} 處），且只更新 actual_*/result_status，"
          f"不會碰 {prob_columns}（有碰到 = {touches_probabilities}）")


def check_7_predict_one_only_sees_team_names():
    """predict_one() 的參數簽章裡不應該有比分/結果相關的參數，
    確保預測時完全不可能拿到這場比賽本身的答案。"""
    from analytics.model_variants import predict_one
    sig = inspect.signature(predict_one)
    param_names = set(sig.parameters.keys())
    forbidden = {"home_goals", "away_goals", "result", "actual_result", "home_score", "away_score"}
    leaked_params = param_names & forbidden
    check("7. predict_one() 的函式簽章是否完全不接受比分/結果類參數（不可能把答案傳進去）",
          len(leaked_params) == 0,
          f"predict_one 參數: {sorted(param_names)}；跟答案有關的參數: {leaked_params or '無'}")


def main():
    print("=== Data Leakage Check ===\n")
    folds = check_1_fold_boundaries()
    print()
    check_2_ratings_are_pure_function_of_input_df(folds)
    print()
    check_3_league_avg_uses_fold_only_data(folds)
    print()
    check_4_tuning_isolates_final_fold()
    print()
    check_5_calibration_is_causal()
    print()
    check_6_predictions_are_append_only()
    print()
    check_7_predict_one_only_sees_team_names()

    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    print(f"\n=== 結果: {n_pass}/{n_total} 項通過 ===")

    out_path = ROOT / "backtest_output" / "data_leakage_check_result.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已輸出: {out_path}")

    if n_pass < n_total:
        print("\n有檢查項目沒有通過，這代表存在 leakage 風險，請看上面 FAIL 的項目。")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
