"""
機率校準（Isotonic Regression / Platt scaling），時間序列安全版本：

    要校準 fold k 的預測，只能用 fold 1..k-1（都已經是「樣本外」的測試結果）
    的「預測機率 vs 實際結果」去 fit 校準器，絕對不會用到 fold k 自己的資料，
    更不會用到全部資料一起 fit 再回頭套用（那樣等於偷看了未來）。

    Fold 1 沒有更早的 fold 可以拿來訓練校準器，所以 fold 1 一律維持原始機率
    （不校準），從 fold 2 開始才有校準結果可比較。

每個結果類別（H/D/A）各自用 one-vs-rest 訓練一個校準器，校準完三個機率一起
正規化讓總和為 1。Isotonic 跟 Platt 都會算，並且都跟「不校準」的原始機率
比較 log loss / brier score，如果校準後反而變差就照實講、不要硬套。
"""
import pathlib
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from analytics import metrics as m

OUTCOMES = ("H", "D", "A")


def _fit_isotonic(train_rows):
    calibrators = {}
    for k in OUTCOMES:
        x = np.array([r[f"prob_{k}"] for r in train_rows])
        y = np.array([1.0 if r["actual_result"] == k else 0.0 for r in train_rows])
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(x, y)
        calibrators[k] = ir
    return calibrators


def _fit_platt(train_rows):
    calibrators = {}
    for k in OUTCOMES:
        x = np.array([r[f"prob_{k}"] for r in train_rows]).reshape(-1, 1)
        y = np.array([1 if r["actual_result"] == k else 0 for r in train_rows])
        if len(np.unique(y)) < 2:
            calibrators[k] = None
            continue
        lr = LogisticRegression()
        lr.fit(x, y)
        calibrators[k] = lr
    return calibrators


def _apply(calibrators, rows, kind: str):
    out = []
    for r in rows:
        raw = {k: r[f"prob_{k}"] for k in OUTCOMES}
        calibrated = {}
        for k in OUTCOMES:
            cal = calibrators.get(k)
            if cal is None:
                calibrated[k] = raw[k]
            elif kind == "isotonic":
                calibrated[k] = float(cal.predict([raw[k]])[0])
            else:  # platt
                calibrated[k] = float(cal.predict_proba([[raw[k]]])[0, 1])
        total = sum(calibrated.values())
        if total <= 0:
            calibrated = raw
        else:
            calibrated = {k: v / total for k, v in calibrated.items()}
        out.append({
            **r,
            "prob_H": calibrated["H"], "prob_D": calibrated["D"], "prob_A": calibrated["A"],
            # 校準後機率排序可能跟原本不一樣，predicted_result 一定要用校準後的機率重新取 argmax，
            # 不然 accuracy 會偷偷沿用校準前的預測結果，讓「校準前後比較」失去意義
            "predicted_result": max(calibrated, key=calibrated.get),
        })
    return out


def sequential_calibrate(fold_rows_by_index: dict) -> dict:
    """fold_rows_by_index: {1: [rows...], 2: [...], 3: [...], 4: [...]}（某一個 model variant 的）。
    回傳 {'raw': {...}, 'isotonic': {...}, 'platt': {...}}，每個底下是
    {'rows': 校準後(或原始)的 fold2+ 全部 rows, 'metrics': full_metrics(...), 'per_fold': {...}}。
    只比較 fold >= 2（fold 1 沒有校準資料），這樣「校準前後」比較的是同一批比賽。
    """
    fold_indices = sorted(fold_rows_by_index.keys())
    comparable_folds = [i for i in fold_indices if i > min(fold_indices)]

    results = {"raw": {"rows": [], "per_fold": {}}, "isotonic": {"rows": [], "per_fold": {}}, "platt": {"rows": [], "per_fold": {}}}

    for k in comparable_folds:
        train_rows = []
        for j in fold_indices:
            if j < k:
                train_rows.extend(fold_rows_by_index[j])
        test_rows = fold_rows_by_index[k]

        iso_cal = _fit_isotonic(train_rows)
        platt_cal = _fit_platt(train_rows)

        raw_rows = test_rows
        iso_rows = _apply(iso_cal, test_rows, "isotonic")
        platt_rows = _apply(platt_cal, test_rows, "platt")

        results["raw"]["rows"].extend(raw_rows)
        results["isotonic"]["rows"].extend(iso_rows)
        results["platt"]["rows"].extend(platt_rows)
        results["raw"]["per_fold"][k] = m.full_metrics(raw_rows)
        results["isotonic"]["per_fold"][k] = m.full_metrics(iso_rows)
        results["platt"]["per_fold"][k] = m.full_metrics(platt_rows)

    for kind in results:
        results[kind]["metrics"] = m.full_metrics(results[kind]["rows"])
        results[kind]["calibration_curve_H"] = m.calibration_curve_one_vs_rest(results[kind]["rows"], "H")

    results["comparable_folds"] = comparable_folds
    return results


def choose_best_method(seq_results: dict) -> str:
    """比較 raw / isotonic / platt 的 log loss，選最低的（沒有進步就維持 raw，
    不要為了「有校準」而硬套一個變差的方法）。"""
    candidates = {k: seq_results[k]["metrics"]["log_loss"] for k in ("raw", "isotonic", "platt")}
    return min(candidates, key=candidates.get)


def main():
    import json
    from analytics.walkforward import run_backtest
    from analytics.model_variants import DEFAULT_PARAMS

    root = pathlib.Path(__file__).resolve().parent.parent
    params_by_variant = {}
    best_params_path = root / "best_params.json"
    if best_params_path.exists():
        tuned = json.loads(best_params_path.read_text(encoding="utf-8"))
        params_by_variant["E"] = {**DEFAULT_PARAMS, **{k: v for k, v in tuned.items() if not k.startswith("_")}}

    results = run_backtest(params_by_variant=params_by_variant, save_to_db=False)
    # 用目前 pooled log loss 最低的 variant 來示範校準效果
    best_variant = min(("A", "B", "C", "D", "E"), key=lambda v: results[v]["pooled_metrics"]["log_loss"])
    print(f"用 log loss 最低的 Model {best_variant} 示範機率校準")

    fold_rows_by_index = {fr["fold"]["fold_index"]: fr["rows"] for fr in results[best_variant]["fold_results"]}
    seq = sequential_calibrate(fold_rows_by_index)

    print(f"\n比較範圍: fold {seq['comparable_folds']}（fold 1 沒有更早的資料可以拿來訓練校準器，不列入比較）")
    print(f"{'方法':10}{'LogLoss':>10}{'Brier':>10}{'Accuracy':>10}")
    for kind in ("raw", "isotonic", "platt"):
        met = seq[kind]["metrics"]
        print(f"{kind:10}{met['log_loss']:>10.4f}{met['brier_score']:>10.4f}{met['accuracy']*100:>9.1f}%")

    best = choose_best_method(seq)
    print(f"\n最佳方法: {best}"
          + ("（校準沒有帶來改善，維持原始機率）" if best == "raw" else "（採用這個校準方法）"))

    import json
    out = {
        "best_variant": best_variant,
        "comparable_folds": seq["comparable_folds"],
        "best_method": best,
        "methods": {
            kind: {"metrics": seq[kind]["metrics"]} for kind in ("raw", "isotonic", "platt")
        },
        "raw_curve_H": seq["raw"]["calibration_curve_H"],
    }
    out_path = root / "backtest_output" / "calibration_result.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已輸出: {out_path}")

    return seq, best, best_variant


if __name__ == "__main__":
    main()
