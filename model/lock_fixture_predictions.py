"""
鎖定英超賽程預測

用目前的模型（model.json）對「還沒鎖定過」的賽程算一次 Dixon-Coles 預測，
結果永久寫入 data/fixture_predictions_lock.json，之後不管模型怎麼重新訓練，
這筆比賽的預測都不會再被改動 —— 確保預測是「賽前鎖定」，不是賽後用更多資料
回頭調整出來的。

已經鎖定過的比賽，這裡只會做兩件事：
  1. 完全不碰預測本身
  2. 如果 fixturedownload 那邊已經有比分了，把實際結果補進同一筆紀錄，
     用來對照「當初鎖定的預測」準不準

輸出:
  - data/fixture_predictions_lock.json   永久保存的鎖定紀錄（含歷史賽季）
  - web/fixtures_data.js                 給網頁顯示用（只含本賽季）
"""
import json
import pathlib
from datetime import datetime, timezone

import common

ROOT = pathlib.Path(__file__).parent.parent
FIXTURES_JSON = ROOT / "data" / "raw" / "fixtures_epl.json"
MODEL_JSON = pathlib.Path(__file__).parent / "model.json"
LOCK_JSON = ROOT / "data" / "fixture_predictions_lock.json"
WEB_JS = ROOT / "web" / "fixtures_data.js"


def load_locks() -> dict:
    if LOCK_JSON.exists():
        return json.loads(LOCK_JSON.read_text(encoding="utf-8"))
    return {}


def fixture_key(fixture: dict) -> str:
    date = fixture["DateUtc"][:10]
    return f"{date}_{fixture['HomeTeam']}_{fixture['AwayTeam']}"


def result_letter(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def predicted_result(entry: dict) -> str:
    probs = {"H": entry["prob_home"], "D": entry["prob_draw"], "A": entry["prob_away"]}
    return max(probs, key=probs.get)


def lock_new_prediction(fixture: dict, model: dict) -> dict:
    home = common.canonical_team_name(fixture["HomeTeam"])
    away = common.canonical_team_name(fixture["AwayTeam"])
    home_rating = model["teams"].get(home, common.DEFAULT_RATING)
    away_rating = model["teams"].get(away, common.DEFAULT_RATING)
    low_confidence = home not in model["teams"] or away not in model["teams"]

    pred = common.predict_match(
        home_rating, away_rating,
        model["league_avg_home_goals"], model["league_avg_away_goals"], model["rho"],
    )

    return {
        "match_number": fixture["MatchNumber"],
        "round": fixture["RoundNumber"],
        "date_utc": fixture["DateUtc"],
        "home_team": home,
        "away_team": away,
        "low_confidence": low_confidence,
        "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prob_home": pred["prob_home"],
        "prob_draw": pred["prob_draw"],
        "prob_away": pred["prob_away"],
        "expected_home_goals": pred["expected_home_goals"],
        "expected_away_goals": pred["expected_away_goals"],
        "top_scores": pred["top_scores"],
        "actual": None,
    }


def main():
    fixtures = json.loads(FIXTURES_JSON.read_text(encoding="utf-8"))
    model = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    locks = load_locks()

    newly_locked = 0
    newly_settled = 0
    low_confidence_count = 0

    season_keys = []
    for fixture in fixtures:
        key = fixture_key(fixture)
        season_keys.append(key)

        if key not in locks:
            locks[key] = lock_new_prediction(fixture, model)
            newly_locked += 1
            if locks[key]["low_confidence"]:
                low_confidence_count += 1

        entry = locks[key]
        hg, ag = fixture.get("HomeTeamScore"), fixture.get("AwayTeamScore")
        if entry["actual"] is None and hg is not None and ag is not None:
            actual_letter = result_letter(hg, ag)
            entry["actual"] = {
                "home_goals": hg,
                "away_goals": ag,
                "result": actual_letter,
                "predicted_result": predicted_result(entry),
                "correct": actual_letter == predicted_result(entry),
            }
            newly_settled += 1

    LOCK_JSON.write_text(json.dumps(locks, ensure_ascii=False, indent=2), encoding="utf-8")

    season_fixtures = [locks[k] for k in season_keys]
    season_fixtures.sort(key=lambda e: e["date_utc"])
    WEB_JS.write_text(
        "// 由 model/lock_fixture_predictions.py 自動產生，請勿手動修改\n"
        f"const EPL_FIXTURES = {json.dumps(season_fixtures, ensure_ascii=False)};\n",
        encoding="utf-8",
    )

    print(f"本次新鎖定 {newly_locked} 場預測（其中 {low_confidence_count} 場因缺乏歷史資料而信心度較低）")
    print(f"本次新增 {newly_settled} 場已結束比賽的實際結果對照")
    print(f"目前共鎖定 {len(locks)} 場比賽（含歷史賽季），本賽季 {len(season_fixtures)} 場")
    print(f"已輸出: {LOCK_JSON}")
    print(f"已輸出: {WEB_JS}")


if __name__ == "__main__":
    main()
