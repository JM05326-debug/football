"""
把 data/raw 底下的歷史比賽 CSV 灌進 SQLite 的 matches 表，當作後續所有分析
（backtest / metrics / report）唯一的資料來源，避免每個模組各自重複解析 CSV
用不同邏輯，彼此結果對不齊。

可重複執行：用 (league, date, home_team, away_team) 當 unique key，
重複匯入不會產生重複列。
"""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from db import database as db

ROOT = pathlib.Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"

SEASON_ORDER = ["2021", "2122", "2223", "2324", "2425", "2526"]


def season_label(code: str) -> str:
    """'2021' -> '2020-21'，'2122' -> '2021-22' ..."""
    if code == "2021":
        return "2020-21"
    start = "20" + code[:2]
    end = code[2:]
    return f"{start}-{end}"


def parse_epl_date(s: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"無法解析日期: {s}")


def load_epl_matches(conn) -> int:
    n = 0
    for code in SEASON_ORDER:
        path = RAW_DIR / f"E0_{code}.csv"
        if not path.exists():
            continue
        available_cols = pd.read_csv(path, nrows=0).columns
        odds_cols = [c for c in ("AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5") if c in available_cols]
        df = pd.read_csv(path, usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"] + odds_cols)
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df["Date"] = df["Date"].apply(parse_epl_date)
        season = season_label(code)
        for _, row in df.iterrows():
            def odd(col):
                v = row.get(col)
                return float(v) if pd.notna(v) else None

            db.upsert_match(
                conn, league="EPL", date=row["Date"].strftime("%Y-%m-%d"),
                home_team=row["HomeTeam"], away_team=row["AwayTeam"],
                home_goals=int(row["FTHG"]), away_goals=int(row["FTAG"]),
                source="football-data.co.uk", season=season,
                avg_h_odds=odd("AvgH"), avg_d_odds=odd("AvgD"), avg_a_odds=odd("AvgA"),
                avg_over25_odds=odd("Avg>2.5"), avg_under25_odds=odd("Avg<2.5"),
            )
            n += 1
    return n


def main():
    db.init_db()
    conn = db.connect()
    n_epl = load_epl_matches(conn)
    conn.commit()
    total_epl = db.count_matches(conn, "EPL")
    conn.close()
    print(f"匯入/更新 EPL {n_epl} 筆")
    print(f"matches 表目前共有 EPL {total_epl} 場")


if __name__ == "__main__":
    main()
