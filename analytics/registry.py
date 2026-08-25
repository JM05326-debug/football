"""
模型版本管理：每次「重新訓練 production 模型」都走這裡，不會直接覆蓋舊模型。

流程:
    1. 用全部目前可用的歷史資料 fit 一個新模型（Model E 設定：完整功能 + best_params.json）
    2. 用 walk-forward backtest 算出這個設定的樣本外表現（validation/test metrics）
    3. 跟目前 is_production=1 的版本比較（用 log loss，其次 brier score）
    4. 只有新模型「真的比較好」才 set_production_model()，否則維持舊版本、
       新版本照樣存進 model_versions 表（is_production=0），保留紀錄但不啟用
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "model"))
import common  # noqa: E402
from db import database as db
from analytics.model_variants import DEFAULT_PARAMS, variant_flags, fit_model
from analytics.walkforward import load_epl_matches_df, run_backtest

ROOT = pathlib.Path(__file__).parent.parent
BEST_PARAMS_PATH = ROOT / "best_params.json"

PRODUCTION_FLAGS = variant_flags("D")  # 完整功能集合：decay + form + elo


def load_production_params() -> dict:
    if BEST_PARAMS_PATH.exists():
        tuned = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        params = {**DEFAULT_PARAMS, **{k: v for k, v in tuned.items() if not k.startswith("_")}}
        return params
    return dict(DEFAULT_PARAMS)


def evaluate_candidate(params: dict) -> dict:
    """跑一次完整 walk-forward（4 個 fold）評估這組參數的樣本外表現，
    回傳 pooled metrics，當作這個候選版本的 test_metrics。"""
    results = run_backtest(params_by_variant={"E": params}, save_to_db=False)
    return results["E"]["pooled_metrics"]


def is_better(candidate_metrics: dict, current_metrics: dict) -> bool:
    """log loss 越低越好，其次 brier score。current_metrics 是 None（沒有 production 版本）
    時，candidate 一定算比較好（第一個版本）。"""
    if current_metrics is None:
        return True
    if candidate_metrics["log_loss"] != current_metrics["log_loss"]:
        return candidate_metrics["log_loss"] < current_metrics["log_loss"]
    return candidate_metrics["brier_score"] < current_metrics["brier_score"]


def train_and_maybe_promote(league="EPL") -> dict:
    conn = db.connect()
    params = load_production_params()

    df = load_epl_matches_df(conn)
    train_cutoff = df["date"].max().strftime("%Y-%m-%d")

    candidate_metrics = evaluate_candidate(params)

    current = db.get_production_model(conn, league)
    current_metrics = json.loads(current["test_metrics_json"]) if current and current["test_metrics_json"] else None

    version = db.next_version_string(conn, league)
    promote = is_better(candidate_metrics, current_metrics)

    db.insert_model_version(
        conn, version=version, league=league, train_data_cutoff=train_cutoff,
        params=params, test_metrics=candidate_metrics, is_production=promote,
        notes=f"walk-forward pooled log_loss={candidate_metrics['log_loss']:.4f}, "
              f"accuracy={candidate_metrics['accuracy']:.4f}"
              + ("" if current is None else f"; 前一版 {current['version']} log_loss="
                                             f"{current_metrics['log_loss']:.4f}"),
    )
    if promote:
        db.set_production_model(conn, version, league)

    conn.commit()
    conn.close()

    return {
        "version": version,
        "promoted": promote,
        "candidate_metrics": candidate_metrics,
        "previous_version": current["version"] if current else None,
        "previous_metrics": current_metrics,
        "params": params,
    }


def fit_production_model(league="EPL"):
    """實際 fit 出可以拿來預測的 model dict（給 predict.py 用），用目前 production
    版本記錄的參數、全部歷史資料。"""
    conn = db.connect()
    prod = db.get_production_model(conn, league)
    df = load_epl_matches_df(conn)
    conn.close()

    params = json.loads(prod["params_json"]) if prod else load_production_params()
    model = fit_model(df, params, PRODUCTION_FLAGS)
    version = prod["version"] if prod else "unversioned"
    return model, params, version


def main():
    result = train_and_maybe_promote()
    cm = result["candidate_metrics"]
    print(f"新版本: {result['version']}")
    print(f"  Walk-forward log_loss={cm['log_loss']:.4f}, brier={cm['brier_score']:.4f}, accuracy={cm['accuracy']*100:.1f}%")
    if result["previous_metrics"]:
        pm = result["previous_metrics"]
        print(f"  前一個 production 版本 {result['previous_version']}: "
              f"log_loss={pm['log_loss']:.4f}, brier={pm['brier_score']:.4f}, accuracy={pm['accuracy']*100:.1f}%")
    print(f"  {'[已升級] 設為 production 模型' if result['promoted'] else '[未升級] 沒有優於現有版本，維持原本的 production 模型'}")
    return result


if __name__ == "__main__":
    main()
