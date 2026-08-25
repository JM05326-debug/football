"""
評估指標：Accuracy / Log Loss / Brier Score / Calibration / 分結果類型的 Accuracy /
大小分 Accuracy / ROI。全部是「樣本外」評估用的函式，不做任何訓練，輸入什麼就算什麼，
不偷看真實結果以外的資訊。
"""
import math

import numpy as np

OUTCOMES = ("H", "D", "A")


def _clip(p, eps=1e-12):
    return min(max(p, eps), 1 - eps)


def accuracy(rows) -> float:
    """rows: list of dict，需有 predicted_result 與 actual_result"""
    if not rows:
        return None
    correct = sum(1 for r in rows if r["predicted_result"] == r["actual_result"])
    return correct / len(rows)


def log_loss(rows) -> float:
    """多類別 log loss = -mean(log(p_實際發生的類別))"""
    if not rows:
        return None
    total = 0.0
    for r in rows:
        p = r[f"prob_{r['actual_result']}"]
        total += -math.log(_clip(p))
    return total / len(rows)


def brier_score(rows) -> float:
    """多類別 Brier score（原始 Brier 1950 定義）: mean_i sum_k (p_ik - y_ik)^2，
    y 是 one-hot 實際結果。範圍 0（完美）~ 2（最差）。"""
    if not rows:
        return None
    total = 0.0
    for r in rows:
        for k in OUTCOMES:
            y = 1.0 if r["actual_result"] == k else 0.0
            p = r[f"prob_{k}"]
            total += (p - y) ** 2
    return total / len(rows)


def result_type_accuracy(rows, result_type: str) -> dict:
    """針對「模型預測是 result_type」的那些場次，算命中率。
    回傳 {'n':預測次數, 'correct':命中次數, 'accuracy':...}，n=0 時 accuracy=None。"""
    subset = [r for r in rows if r["predicted_result"] == result_type]
    n = len(subset)
    correct = sum(1 for r in subset if r["actual_result"] == result_type)
    return {"n": n, "correct": correct, "accuracy": (correct / n) if n else None}


def over_under_accuracy(rows) -> dict:
    """rows 需有 over_2_5_probability、actual_home_goals、actual_away_goals。
    模型預測「大」的定義: over_2_5_probability > 0.5。"""
    over_n = over_correct = under_n = under_correct = 0
    for r in rows:
        if r.get("actual_home_goals") is None:
            continue
        actual_total = r["actual_home_goals"] + r["actual_away_goals"]
        actual_over = actual_total > 2.5
        predicted_over = r["over_2_5_probability"] > 0.5
        if predicted_over:
            over_n += 1
            over_correct += 1 if actual_over else 0
        else:
            under_n += 1
            under_correct += 1 if not actual_over else 0
    return {
        "over_2_5_accuracy": (over_correct / over_n) if over_n else None,
        "under_2_5_accuracy": (under_correct / under_n) if under_n else None,
        "over_n": over_n, "under_n": under_n,
    }


def full_metrics(rows) -> dict:
    """彙整以上所有指標成一個 dict，給 DB 寫入 / report 使用。"""
    ou = over_under_accuracy(rows)
    return {
        "n": len(rows),
        "accuracy": accuracy(rows),
        "log_loss": log_loss(rows),
        "brier_score": brier_score(rows),
        "home_accuracy": result_type_accuracy(rows, "H")["accuracy"],
        "draw_accuracy": result_type_accuracy(rows, "D")["accuracy"],
        "away_accuracy": result_type_accuracy(rows, "A")["accuracy"],
        "over_2_5_accuracy": ou["over_2_5_accuracy"],
        "under_2_5_accuracy": ou["under_2_5_accuracy"],
    }


def rolling_windows(rows_sorted_by_date: list) -> dict:
    """rows 需已依 match_date 升冪排序。回傳 last10/last50/last100/all 四組 full_metrics()。"""
    windows = {}
    for name, n in (("last10", 10), ("last50", 50), ("last100", 100)):
        subset = rows_sorted_by_date[-n:] if len(rows_sorted_by_date) >= 1 else []
        windows[name] = full_metrics(subset)
    windows["all"] = full_metrics(rows_sorted_by_date)
    return windows


def flat_stake_roi(rows, stake=1.0) -> dict:
    """用市場平均收盤賠率（matches.avg_h_odds/avg_d_odds/avg_a_odds）算「每場都對預測結果
    下注 stake」的報酬率。rows 需有 predicted_result、actual_result、home_odds/draw_odds/away_odds
    （沒有賠率資料的場次會被跳過，不計入分母）。
    ROI = (總回收 - 總投入) / 總投入。回傳 None 代表完全沒有賠率資料可用（不要假裝有 ROI）。"""
    total_staked = 0.0
    total_returned = 0.0
    n_bets = 0
    for r in rows:
        odds_key = {"H": "home_odds", "D": "draw_odds", "A": "away_odds"}[r["predicted_result"]]
        odds = r.get(odds_key)
        if odds is None:
            continue
        n_bets += 1
        total_staked += stake
        if r["predicted_result"] == r["actual_result"]:
            total_returned += stake * odds
    if n_bets == 0:
        return {"roi": None, "n_bets": 0, "total_staked": 0.0, "total_returned": 0.0}
    return {
        "roi": (total_returned - total_staked) / total_staked,
        "n_bets": n_bets,
        "total_staked": total_staked,
        "total_returned": total_returned,
    }


def calibration_curve_one_vs_rest(rows, outcome: str, n_bins=10):
    """outcome 的 one-vs-rest calibration curve：把 prob_{outcome} 分成 n_bins 個等寬區間，
    每個區間算「平均預測機率」vs「實際發生 outcome 的比例」。用來畫 calibration curve、
    也用來判斷模型是否 over/under-confident。回傳 list of {bin_mid, mean_predicted, actual_frequency, n}。"""
    probs = np.array([r[f"prob_{outcome}"] for r in rows])
    actual = np.array([1.0 if r["actual_result"] == outcome else 0.0 for r in rows])
    if len(probs) == 0:
        return []
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({
            "bin_mid": float((lo + hi) / 2),
            "mean_predicted": float(probs[mask].mean()),
            "actual_frequency": float(actual[mask].mean()),
            "n": n,
        })
    return out
