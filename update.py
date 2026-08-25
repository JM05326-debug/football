"""
每天開一次電腦、跑一次 `python update.py`，自動完成全部更新流程：

    更新資料 -> 檢查新比賽 -> 更新歷史資料(matches) -> 取得最新結果(settle predictions)
    -> 更新 predictions -> 重新計算模型表現(model_metrics) -> 判斷是否需要重新訓練
    -> (需要的話) 重新訓練 + walk-forward 驗證 + 跟現有 production 比較 -> 只有更好才升級
    -> 產生 model_report.html

設計成可以隨時重複執行、中途失敗也可以直接重跑：
    - matches 用 upsert，同一場比賽重複匯入不會變成兩筆
    - predictions 只在「這場比賽 + 目前 production 版本」還沒有預測紀錄時才新增，
      不會每天對同一場未開打的比賽重複灌預測
    - 任何一個步驟丟例外，只會記錄下來、跳到下一步，不會刪除任何既有資料，
      也不會讓整個程式當掉（除了資料庫本身壞掉這種真的沒辦法的情況）
"""
import json
import logging
import pathlib
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "data"))

from db import database as db
from db import load_matches

ROOT = pathlib.Path(__file__).parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

RETRAIN_MIN_NEW_MATCHES = 10  # 累積至少這麼多場新賽果，才觸發重新訓練（避免雜訊）

logger = logging.getLogger("update")


def setup_logging():
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(LOG_DIR / f"update_{ts}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)
    return file_handler.baseFilename


def step(name):
    """裝飾器：包住每個步驟，記錄開始/結束/例外，例外不會讓整個 update.py 中斷。"""
    def decorator(fn):
        def wrapped(summary, *args, **kwargs):
            logger.info(f"--- {name} 開始 ---")
            try:
                result = fn(*args, **kwargs)
                summary["steps"][name] = {"ok": True, "result": result}
                logger.info(f"--- {name} 完成: {result} ---")
                return result
            except Exception as e:
                summary["steps"][name] = {"ok": False, "error": str(e)}
                logger.error(f"--- {name} 失敗: {e} ---")
                logger.error(traceback.format_exc())
                return None
        return wrapped
    return decorator


@step("抓取最新資料")
def fetch_latest_data():
    import importlib
    results = {}
    for module_name in ("fetch_epl_data", "fetch_fixtures"):
        mod = importlib.import_module(module_name)
        mod.main()
        results[module_name] = "ok"
    return results


@step("更新歷史比賽資料庫")
def update_matches_db():
    conn = db.connect()
    n_epl = load_matches.load_epl_matches(conn)
    conn.commit()
    total_epl = db.count_matches(conn, "EPL")
    conn.close()
    return {"n_epl_upserted": n_epl, "total_epl_matches": total_epl}


@step("結算已完成比賽的預測結果")
def settle_predictions():
    conn = db.connect()
    n_settled = db.settle_pending_predictions(conn, league="EPL")
    conn.commit()
    counts = db.count_predictions(conn, league="EPL")
    conn.close()
    return {"newly_settled": n_settled, "current_counts": counts}


@step("為新賽程產生預測")
def predict_new_fixtures():
    from analytics.registry import fit_production_model, train_and_maybe_promote, PRODUCTION_FLAGS
    from analytics.model_variants import predict_one
    import common

    conn = db.connect()
    prod = db.get_production_model(conn, "EPL")
    if prod is None:
        conn.close()
        logger.info("目前沒有 production 模型，先訓練一個")
        train_and_maybe_promote("EPL")
        conn = db.connect()
        prod = db.get_production_model(conn, "EPL")

    fixtures_path = ROOT / "data" / "raw" / "fixtures_epl.json"
    if not fixtures_path.exists():
        conn.close()
        return {"n_new_predictions": 0, "note": "找不到 fixtures_epl.json"}

    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    model, params, version = fit_production_model("EPL")

    n_new = 0
    n_skipped_low_data = 0
    for fx in fixtures:
        home = common.canonical_team_name(fx["HomeTeam"])
        away = common.canonical_team_name(fx["AwayTeam"])
        match_date = fx["DateUtc"][:10]
        match_key = db.make_match_key(match_date, home, away)

        if db.prediction_exists(conn, match_key, version):
            continue

        pred = predict_one(model, home, away, params, PRODUCTION_FLAGS)
        db.insert_prediction(
            conn, match_date=match_date, league="EPL", home_team=home, away_team=away,
            home_win_probability=pred["prob_H"], draw_probability=pred["prob_D"], away_win_probability=pred["prob_A"],
            predicted_result=pred["predicted_result"], predicted_score=pred["predicted_score"],
            over_2_5_probability=pred["over_2_5_probability"], under_2_5_probability=pred["under_2_5_probability"],
            model_version=version,
        )
        n_new += 1
        if pred["low_confidence"]:
            n_skipped_low_data += 1

    conn.commit()
    conn.close()
    return {"n_new_predictions": n_new, "n_low_confidence": n_skipped_low_data, "model_version": version}


@step("重新計算模型表現指標")
def recompute_metrics():
    from analytics import metrics as m

    conn = db.connect()
    prod = db.get_production_model(conn, "EPL")
    if prod is None:
        conn.close()
        return {"note": "沒有 production 模型，略過"}

    settled = db.fetch_settled_predictions(conn, league="EPL", model_version=prod["version"])
    rows = [
        {"prob_H": r["home_win_probability"], "prob_D": r["draw_probability"], "prob_A": r["away_win_probability"],
         "predicted_result": r["predicted_result"], "actual_result": r["actual_result"],
         "over_2_5_probability": r["over_2_5_probability"],
         "actual_home_goals": r["actual_home_goals"], "actual_away_goals": r["actual_away_goals"]}
        for r in settled
    ]
    if not rows:
        conn.close()
        return {"note": "目前沒有已結算的正式預測，還沒有東西可以算"}

    windows = m.rolling_windows(rows)
    for window_name, metrics in windows.items():
        if metrics["n"] == 0:
            continue
        db.insert_model_metrics(conn, model_version=prod["version"], league="EPL",
                                 window=window_name, n_predictions=metrics["n"], metrics=metrics)
    conn.commit()
    conn.close()
    return {"n_settled": len(rows), "windows_recorded": [k for k, v in windows.items() if v["n"] > 0]}


@step("檢查是否需要重新訓練")
def check_retrain_condition():
    conn = db.connect()
    prod = db.get_production_model(conn, "EPL")
    if prod is None:
        conn.close()
        return {"should_retrain": True, "reason": "還沒有任何 production 模型"}

    n_new_matches = conn.execute(
        "SELECT COUNT(*) c FROM matches WHERE league='EPL' AND date > ?",
        (prod["train_data_cutoff"],),
    ).fetchone()["c"]
    conn.close()

    should = n_new_matches >= RETRAIN_MIN_NEW_MATCHES
    return {
        "should_retrain": should,
        "n_new_matches_since_last_train": n_new_matches,
        "threshold": RETRAIN_MIN_NEW_MATCHES,
        "reason": f"累積 {n_new_matches} 場新賽果 {'>=' if should else '<'} 門檻 {RETRAIN_MIN_NEW_MATCHES}",
    }


@step("重新訓練並評估是否升級 production 模型")
def retrain_and_maybe_promote():
    from analytics.registry import train_and_maybe_promote
    result = train_and_maybe_promote("EPL")
    return {
        "new_version": result["version"],
        "promoted": result["promoted"],
        "log_loss": result["candidate_metrics"]["log_loss"],
        "accuracy": result["candidate_metrics"]["accuracy"],
    }


@step("產生報告")
def generate_report():
    from reports.generate_report import main as report_main
    path = report_main()
    return {"path": str(path)}


def main():
    log_path = setup_logging()
    logger.info(f"=== update.py 開始 (log: {log_path}) ===")

    conn = db.connect()
    db.init_db()
    run_id = db.start_update_run(conn)
    conn.commit()
    conn.close()

    summary = {"started_at": db.now_iso(), "steps": {}}

    fetch_latest_data(summary)
    update_matches_db(summary)
    settle_predictions(summary)
    predict_new_fixtures(summary)
    recompute_metrics(summary)
    retrain_check = check_retrain_condition(summary)

    if retrain_check and retrain_check.get("should_retrain"):
        retrain_and_maybe_promote(summary)
    else:
        logger.info(f"--- 跳過重新訓練: {retrain_check} ---")
        summary["steps"]["重新訓練並評估是否升級 production 模型"] = {"ok": True, "result": "skipped", "reason": retrain_check}

    generate_report(summary)

    n_failed = sum(1 for s in summary["steps"].values() if not s.get("ok", True))
    status = "success" if n_failed == 0 else "failed"

    conn = db.connect()
    db.finish_update_run(conn, run_id, status, summary)
    conn.commit()
    conn.close()

    logger.info(f"=== update.py 結束，狀態: {status}（{n_failed} 個步驟失敗） ===")
    print(f"\n完整 log: {log_path}")
    print(f"model_report.html: {ROOT / 'model_report.html'}")

    return summary


if __name__ == "__main__":
    main()
