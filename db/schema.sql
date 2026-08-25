-- 足球預測模型 SQLite schema
-- 設計原則: predictions 只能 INSERT，不能覆蓋既有預測（見 database.py 的存取函式）

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT NOT NULL,              -- 'EPL' / 'INTL'
    season TEXT,                       -- 例如 '2023-24'，international 可為 NULL
    date TEXT NOT NULL,                -- ISO 'YYYY-MM-DD'
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_goals INTEGER NOT NULL,
    away_goals INTEGER NOT NULL,
    result TEXT NOT NULL,              -- 'H' / 'D' / 'A'
    source TEXT NOT NULL,              -- 例如 'football-data.co.uk'
    -- 市場平均收盤賠率（football-data.co.uk 的 AvgH/AvgD/AvgA/Avg>2.5/Avg<2.5），
    -- 只有 EPL 有，international 資料源沒有賠率，維持 NULL。用來算 ROI，不是拿來當模型輸入特徵。
    avg_h_odds REAL,
    avg_d_odds REAL,
    avg_a_odds REAL,
    avg_over25_odds REAL,
    avg_under25_odds REAL,
    UNIQUE(league, date, home_team, away_team)
);

CREATE INDEX IF NOT EXISTS idx_matches_league_date ON matches(league, date);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_key TEXT NOT NULL,           -- '{date}_{home_team}_{away_team}'，用來對應同一場比賽的多筆預測
    prediction_date TEXT NOT NULL,     -- 這筆預測產生的時間 (ISO datetime)
    match_date TEXT NOT NULL,          -- 比賽日期 'YYYY-MM-DD'
    league TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_win_probability REAL NOT NULL,
    draw_probability REAL NOT NULL,
    away_win_probability REAL NOT NULL,
    predicted_result TEXT NOT NULL,        -- 'H' / 'D' / 'A'
    predicted_score TEXT NOT NULL,         -- 例如 '2-1'
    over_2_5_probability REAL NOT NULL,
    under_2_5_probability REAL NOT NULL,
    model_version TEXT NOT NULL,
    -- 賽後補上的實際結果，result_status 在補上之前一律是 'pending'
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    actual_result TEXT,
    result_status TEXT NOT NULL DEFAULT 'pending'  -- 'pending' / 'settled'
);

CREATE INDEX IF NOT EXISTS idx_predictions_match_key ON predictions(match_key);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(result_status);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);

CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,          -- 例如 'v1.0'
    league TEXT NOT NULL,
    created_at TEXT NOT NULL,
    train_data_cutoff TEXT NOT NULL,   -- 訓練資料最後日期 'YYYY-MM-DD'
    params_json TEXT NOT NULL,
    validation_metrics_json TEXT,
    test_metrics_json TEXT,
    is_production INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    league TEXT NOT NULL,
    window TEXT NOT NULL,              -- 'last10' / 'last50' / 'last100' / 'all'
    n_predictions INTEGER NOT NULL,
    accuracy REAL,
    log_loss REAL,
    brier_score REAL,
    home_accuracy REAL,
    draw_accuracy REAL,
    away_accuracy REAL,
    over_2_5_accuracy REAL,
    under_2_5_accuracy REAL
);

CREATE TABLE IF NOT EXISTS backtest_folds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    league TEXT NOT NULL,
    model_variant TEXT NOT NULL,       -- 'A' / 'B' / 'C' / 'D' / 'E'
    fold_index INTEGER NOT NULL,
    train_start TEXT NOT NULL,
    train_end TEXT NOT NULL,
    test_start TEXT NOT NULL,
    test_end TEXT NOT NULL,
    n_test_matches INTEGER NOT NULL,
    accuracy REAL,
    log_loss REAL,
    brier_score REAL,
    home_accuracy REAL,
    draw_accuracy REAL,
    away_accuracy REAL,
    roi REAL
);

CREATE TABLE IF NOT EXISTS update_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',   -- 'running' / 'success' / 'failed'
    summary_json TEXT
);
