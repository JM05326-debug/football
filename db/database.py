"""
SQLite 資料庫存取層。所有其他模組（predict.py / update.py / backtest / report）
都透過這裡的函式讀寫資料庫，不直接下 SQL，確保「predictions 只能新增、不能覆蓋」
這條規則在單一地方被強制執行。
"""
import json
import pathlib
import sqlite3
from datetime import datetime, timezone

DB_PATH = pathlib.Path(__file__).parent / "predictions.db"
SCHEMA_PATH = pathlib.Path(__file__).parent / "schema.sql"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = connect()
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------- matches ----------------

def upsert_match(conn, league, date, home_team, away_team, home_goals, away_goals, source, season=None,
                  avg_h_odds=None, avg_d_odds=None, avg_a_odds=None, avg_over25_odds=None, avg_under25_odds=None):
    """歷史比賽結果是客觀事實，允許用 INSERT OR REPLACE 更新（例如資料源修正舊比分），
    這跟 predictions 的「不可覆蓋」規則不衝突 —— matches 存的是事實，predictions 存的是
    「當初做出的判斷」。"""
    result = "H" if home_goals > away_goals else ("A" if home_goals < away_goals else "D")
    conn.execute(
        """INSERT INTO matches (league, season, date, home_team, away_team, home_goals, away_goals, result, source,
             avg_h_odds, avg_d_odds, avg_a_odds, avg_over25_odds, avg_under25_odds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(league, date, home_team, away_team) DO UPDATE SET
             home_goals=excluded.home_goals, away_goals=excluded.away_goals,
             result=excluded.result, season=excluded.season, source=excluded.source,
             avg_h_odds=excluded.avg_h_odds, avg_d_odds=excluded.avg_d_odds, avg_a_odds=excluded.avg_a_odds,
             avg_over25_odds=excluded.avg_over25_odds, avg_under25_odds=excluded.avg_under25_odds""",
        (league, season, date, home_team, away_team, home_goals, away_goals, result, source,
         avg_h_odds, avg_d_odds, avg_a_odds, avg_over25_odds, avg_under25_odds),
    )


def count_matches(conn, league=None) -> int:
    if league:
        return conn.execute("SELECT COUNT(*) c FROM matches WHERE league=?", (league,)).fetchone()["c"]
    return conn.execute("SELECT COUNT(*) c FROM matches").fetchone()["c"]


# ---------------- predictions ----------------

def make_match_key(match_date, home_team, away_team) -> str:
    return f"{match_date}_{home_team}_{away_team}"


def prediction_exists(conn, match_key, model_version) -> bool:
    row = conn.execute(
        "SELECT 1 FROM predictions WHERE match_key=? AND model_version=? LIMIT 1",
        (match_key, model_version),
    ).fetchone()
    return row is not None


def insert_prediction(conn, *, match_date, league, home_team, away_team,
                       home_win_probability, draw_probability, away_win_probability,
                       predicted_result, predicted_score,
                       over_2_5_probability, under_2_5_probability, model_version,
                       prediction_date=None) -> int:
    """永遠新增一筆，絕不 UPDATE/覆蓋既有預測。回傳新 prediction_id。"""
    match_key = make_match_key(match_date, home_team, away_team)
    cur = conn.execute(
        """INSERT INTO predictions (
             match_key, prediction_date, match_date, league, home_team, away_team,
             home_win_probability, draw_probability, away_win_probability,
             predicted_result, predicted_score, over_2_5_probability, under_2_5_probability,
             model_version, result_status
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
        (match_key, prediction_date or now_iso(), match_date, league, home_team, away_team,
         home_win_probability, draw_probability, away_win_probability,
         predicted_result, predicted_score, over_2_5_probability, under_2_5_probability,
         model_version),
    )
    return cur.lastrowid


def settle_pending_predictions(conn, league=None) -> int:
    """把 result_status='pending' 的預測，對照 matches 表看是否已經有實際結果，
    有的話補上 actual_* 欄位並標記 settled。找不到實際結果的維持 pending，不會出錯。
    回傳這次新結算的筆數。"""
    query = "SELECT prediction_id, match_date, home_team, away_team, league FROM predictions WHERE result_status='pending'"
    params = ()
    if league:
        query += " AND league=?"
        params = (league,)
    pending = conn.execute(query, params).fetchall()

    settled = 0
    for p in pending:
        m = conn.execute(
            "SELECT home_goals, away_goals, result FROM matches WHERE league=? AND date=? AND home_team=? AND away_team=?",
            (p["league"], p["match_date"], p["home_team"], p["away_team"]),
        ).fetchone()
        if m is None:
            continue
        conn.execute(
            """UPDATE predictions SET actual_home_goals=?, actual_away_goals=?, actual_result=?, result_status='settled'
               WHERE prediction_id=?""",
            (m["home_goals"], m["away_goals"], m["result"], p["prediction_id"]),
        )
        settled += 1
    return settled


def fetch_settled_predictions(conn, league=None, model_version=None, limit=None):
    query = "SELECT * FROM predictions WHERE result_status='settled'"
    params = []
    if league:
        query += " AND league=?"
        params.append(league)
    if model_version:
        query += " AND model_version=?"
        params.append(model_version)
    query += " ORDER BY match_date ASC"
    rows = conn.execute(query, params).fetchall()
    rows = [dict(r) for r in rows]
    if limit:
        rows = rows[-limit:]
    return rows


def count_predictions(conn, league=None):
    query = "SELECT result_status, COUNT(*) c FROM predictions"
    params = ()
    if league:
        query += " WHERE league=?"
        params = (league,)
    query += " GROUP BY result_status"
    rows = conn.execute(query, params).fetchall()
    return {r["result_status"]: r["c"] for r in rows}


# ---------------- model_versions ----------------

def insert_model_version(conn, *, version, league, train_data_cutoff, params: dict,
                          validation_metrics: dict = None, test_metrics: dict = None,
                          is_production=False, notes=""):
    conn.execute(
        """INSERT INTO model_versions (version, league, created_at, train_data_cutoff, params_json,
             validation_metrics_json, test_metrics_json, is_production, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(version) DO UPDATE SET
             validation_metrics_json=excluded.validation_metrics_json,
             test_metrics_json=excluded.test_metrics_json,
             is_production=excluded.is_production, notes=excluded.notes""",
        (version, league, now_iso(), train_data_cutoff, json.dumps(params, ensure_ascii=False),
         json.dumps(validation_metrics, ensure_ascii=False) if validation_metrics else None,
         json.dumps(test_metrics, ensure_ascii=False) if test_metrics else None,
         1 if is_production else 0, notes),
    )


def get_production_model(conn, league):
    row = conn.execute(
        "SELECT * FROM model_versions WHERE league=? AND is_production=1 ORDER BY created_at DESC LIMIT 1",
        (league,),
    ).fetchone()
    return dict(row) if row else None


def set_production_model(conn, version, league):
    conn.execute("UPDATE model_versions SET is_production=0 WHERE league=?", (league,))
    conn.execute("UPDATE model_versions SET is_production=1 WHERE version=? AND league=?", (version, league))


def next_version_string(conn, league) -> str:
    rows = conn.execute("SELECT version FROM model_versions WHERE league=?", (league,)).fetchall()
    max_major, max_minor = 0, -1
    for r in rows:
        v = r["version"].lstrip("v")
        try:
            major, minor = v.split(".")
            major, minor = int(major), int(minor)
        except ValueError:
            continue
        if (major, minor) > (max_major, max_minor):
            max_major, max_minor = major, minor
    if max_minor == -1:
        return "v1.0"
    return f"v{max_major}.{max_minor + 1}"


# ---------------- model_metrics ----------------

def insert_model_metrics(conn, *, model_version, league, window, n_predictions, metrics: dict):
    conn.execute(
        """INSERT INTO model_metrics (computed_at, model_version, league, window, n_predictions,
             accuracy, log_loss, brier_score, home_accuracy, draw_accuracy, away_accuracy,
             over_2_5_accuracy, under_2_5_accuracy)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (now_iso(), model_version, league, window, n_predictions,
         metrics.get("accuracy"), metrics.get("log_loss"), metrics.get("brier_score"),
         metrics.get("home_accuracy"), metrics.get("draw_accuracy"), metrics.get("away_accuracy"),
         metrics.get("over_2_5_accuracy"), metrics.get("under_2_5_accuracy")),
    )


def fetch_metrics_history(conn, league, window="all"):
    rows = conn.execute(
        "SELECT * FROM model_metrics WHERE league=? AND window=? ORDER BY computed_at ASC",
        (league, window),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------- backtest_folds ----------------

def insert_backtest_fold(conn, *, run_at, league, model_variant, fold_index,
                          train_start, train_end, test_start, test_end,
                          n_test_matches, metrics: dict, roi=None):
    conn.execute(
        """INSERT INTO backtest_folds (run_at, league, model_variant, fold_index, train_start, train_end,
             test_start, test_end, n_test_matches, accuracy, log_loss, brier_score,
             home_accuracy, draw_accuracy, away_accuracy, roi)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_at, league, model_variant, fold_index, train_start, train_end, test_start, test_end,
         n_test_matches, metrics.get("accuracy"), metrics.get("log_loss"), metrics.get("brier_score"),
         metrics.get("home_accuracy"), metrics.get("draw_accuracy"), metrics.get("away_accuracy"), roi),
    )


def fetch_backtest_folds(conn, league=None, run_at=None):
    query = "SELECT * FROM backtest_folds WHERE 1=1"
    params = []
    if league:
        query += " AND league=?"
        params.append(league)
    if run_at:
        query += " AND run_at=?"
        params.append(run_at)
    query += " ORDER BY model_variant, fold_index"
    return [dict(r) for r in conn.execute(query, params).fetchall()]


# ---------------- update_runs ----------------

def start_update_run(conn) -> int:
    cur = conn.execute("INSERT INTO update_runs (started_at, status) VALUES (?, 'running')", (now_iso(),))
    return cur.lastrowid


def finish_update_run(conn, run_id, status, summary: dict):
    conn.execute(
        "UPDATE update_runs SET finished_at=?, status=?, summary_json=? WHERE id=?",
        (now_iso(), status, json.dumps(summary, ensure_ascii=False), run_id),
    )
