"""
訓練英超比賽預測模型
方法: Dixon-Coles Poisson 模型
  - 每隊有「主場/客場 進攻強度」與「主場/客場 防守強度」
  - 用指數衰減對近期賽果加權（越近期比重越高）
  - 用 Dixon-Coles rho 修正低比分（0-0, 1-0, 0-1, 1-1）之間的相關性
  - 另外計算 Elo Rating、最近 10 場戰績手感、主客場細分數據（見 common.py）

輸出:
  - model/model.json          給人查看/除錯用
  - web/model_data.js         給網頁直接載入 (避免 file:// 下 CORS 問題)
"""
import json
import pathlib
from datetime import datetime

import pandas as pd

import common

RAW_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw"
MODEL_JSON = pathlib.Path(__file__).parent / "model.json"
WEB_JS = pathlib.Path(__file__).parent.parent / "web" / "model_data.js"

HALF_LIFE_DAYS = 380  # 約一個賽季，越舊的比賽權重衰減越多


def load_matches() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(RAW_DIR.glob("E0_*.csv")):
        df = pd.read_csv(csv_path, usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    def parse_date(s: str) -> datetime:
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"無法解析日期: {s}")

    all_df["Date"] = all_df["Date"].apply(parse_date)
    all_df = all_df.rename(columns={
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals",
    })
    all_df["home_goals"] = all_df["home_goals"].astype(int)
    all_df["away_goals"] = all_df["away_goals"].astype(int)
    return all_df.sort_values("date").reset_index(drop=True)


def main():
    df = load_matches()
    weights = common.compute_weights(df["date"], HALF_LIFE_DAYS)
    ratings, avg_home, avg_away = common.fit_team_strengths(df, weights)
    rho = common.fit_rho(df, weights, ratings, avg_home, avg_away)

    elo = common.compute_elo_ratings(df)
    recent_form, home_away_splits = common.compute_recent_form_and_splits(df)
    draw_rate = float((df["home_goals"] == df["away_goals"]).mean())

    for t in ratings:
        ratings[t]["elo"] = round(elo.get(t, common.INITIAL_ELO), 1)
        ratings[t]["recent_form"] = recent_form.get(t)
        ratings[t]["home_away_splits"] = home_away_splits.get(t)

    model = {
        "league": "英超 Premier League",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "matches_used": len(df),
        "league_avg_home_goals": avg_home,
        "league_avg_away_goals": avg_away,
        "rho": rho,
        "avg_draw_rate": draw_rate,
        "teams": ratings,
    }

    MODEL_JSON.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    WEB_JS.write_text(
        "// 由 model/train.py 自動產生，請勿手動修改\n"
        f"const EPL_MODEL_DATA = {json.dumps(model, ensure_ascii=False)};\n",
        encoding="utf-8",
    )

    print(f"共使用 {len(df)} 場比賽資料，涵蓋 {len(ratings)} 支球隊")
    print(f"聯盟平均主場進球: {avg_home:.3f}, 平均客場進球: {avg_away:.3f}")
    print(f"Dixon-Coles rho: {rho:.3f}")
    print(f"平均和局率: {draw_rate:.3f}")
    print(f"已輸出: {MODEL_JSON}")
    print(f"已輸出: {WEB_JS}")


if __name__ == "__main__":
    main()
