"""
下載英超 (Premier League) 歷史比賽數據
資料來源: football-data.co.uk (免費公開，允許個人/研究用途下載)
"""
import pathlib
import requests

RAW_DIR = pathlib.Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)

# 賽季代碼: 例如 "2324" 代表 2023-24 賽季
SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"


def fetch_season(season: str) -> bool:
    url = BASE_URL.format(season=season)
    dest = RAW_DIR / f"E0_{season}.csv"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        if len(resp.content) < 500:
            print(f"[skip] {season}: 回應內容過小，可能該賽季尚無資料")
            return False
        dest.write_bytes(resp.content)
        print(f"[ok]   {season}: 已儲存至 {dest}")
        return True
    except requests.RequestException as e:
        print(f"[fail] {season}: {e}")
        return False


def main():
    ok_count = 0
    for season in SEASONS:
        if fetch_season(season):
            ok_count += 1
    print(f"\n完成，共成功下載 {ok_count}/{len(SEASONS)} 個賽季")


if __name__ == "__main__":
    main()
