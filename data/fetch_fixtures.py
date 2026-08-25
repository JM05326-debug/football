"""
下載英超賽程（含未來場次與已完成場次的比分）
資料來源: fixturedownload.com（免費公開 JSON feed，涵蓋整個賽季賽程）

football-data.co.uk 只提供「已賽完」的比賽結果，沒有未來賽程，
所以未來賽程另外用這個來源，球隊命名兩邊略有差異（見 model/common.py 的
FIXTURE_TEAM_ALIASES），下載下來的原始資料不做名稱轉換，轉換交給
model/lock_fixture_predictions.py 處理。
"""
import datetime
import pathlib

import requests

RAW_DIR = pathlib.Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)

DEST = RAW_DIR / "fixtures_epl.json"


def current_season() -> int:
    """英超賽季固定約 8 月開打、隔年 5 月結束，用「賽季開打那年」當作 season 代號
    （例如 2026-27 賽季代號是 2026）。7 月之前都算上一個賽季還沒結束。"""
    today = datetime.date.today()
    return today.year if today.month >= 7 else today.year - 1


def main():
    season = current_season()
    url = f"https://fixturedownload.com/feed/json/epl-{season}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    fixtures = resp.json()

    if not isinstance(fixtures, list) or not fixtures:
        raise RuntimeError(f"賽程資料格式異常或為空: {url}")

    DEST.write_bytes(resp.content)
    print(f"已下載 {season}-{season + 1} 賽季賽程: {DEST}（共 {len(fixtures)} 場）")


if __name__ == "__main__":
    main()
