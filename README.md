# 比賽預測模型

Dixon-Coles Poisson 英超比賽預測模型，靜態網頁 + GitHub Actions 排程自動更新。
網頁只做英超，不用手動選隊，直接列出整季賽程（依輪次分組），每場都顯示鎖定預測。

- 網頁: `web/`（純前端，讀取 `model_data.js` / `fixtures_data.js`），每場比賽卡片顯示：
  主/和/客機率、總進球大小分 2.5、各隊預期進球、最終比分預測 Top 3、
  全場總進球數分布（0-1/2-3/4-5/6+），已完賽的還會顯示實際比分跟命中與否
- 模型訓練: `model/train.py`
- 資料抓取: `data/fetch_epl_data.py`（英超歷史賽果）、`data/fetch_fixtures.py`（英超賽程，含未來場次，來源 fixturedownload.com）
- 賽程鎖定預測: `model/lock_fixture_predictions.py`
  - 每天抓到新賽程時，用當下的模型算一次 Dixon-Coles 機率就永久寫入 `data/fixture_predictions_lock.json`，之後模型重新訓練也不會回頭修改這場比賽的預測（賽前鎖定，非賽後校正）
  - 比賽結束後會自動把實際比分補進同一筆紀錄，網頁會顯示鎖定的預測命中率
  - 沒有英超歷史資料的球隊（剛升班）會用聯盟平均值當預設強度算出預測，並標記「⚠️升班馬」信心度較低
  - 大小分 2.5、總進球數分布是網頁端用鎖定當下存的期望進球即時算出來顯示用的，數學上與鎖定的主/和/客機率一致

國際賽（國家隊）功能已經完全移除：網站、排程、抓資料/訓練腳本（`fetch_intl_data.py`、
`train_intl.py`）、歷史資料（`international_results.csv`）、中文球隊對照表裡的國家隊
翻譯都已經刪除，本機分析系統（`db/`、`update.py`）也不再匯入國際賽資料。
- 自動排程: `.github/workflows/update-deploy.yml`
  - 每天台北時間 03:00 / 12:00 / 15:00 / 19:00 自動抓取最新賽果與賽程、重新訓練模型、鎖定新賽程預測、提交變更並部署到 GitHub Pages
  - 也可以在 GitHub repo 的 Actions 頁籤手動點選 "Run workflow" 立即更新
  - 排程時間可修改 workflow 檔案裡的 `cron` 設定

## 本機開發

```
pip install -r requirements.txt
python server.py
```

瀏覽器打開 http://localhost:8000，此時網頁上會顯示「更新最新數據並重新訓練」按鈕（雲端版本則會自動每日更新，不顯示此按鈕）。

## 本機模型評估系統（SQLite + Walk-forward Backtest）

跟上面的雲端網頁是兩個獨立的系統：雲端網頁只負責「顯示」鎖定的賽程預測；
這裡是給你自己在本機追蹤「模型到底準不準」用的完整分析工具，資料庫在
`db/predictions.db`（不進版控，機器本地狀態）。

- `python update.py` — 每天開一次電腦執行一次：抓資料 → 更新歷史比賽 → 結算已完賽預測
  → 幫新賽程產生預測 → 重新計算表現指標 → 視情況重新訓練並跟現有 production 模型比較
  → 只有更好才升級 → 產生 `model_report.html`
- `python predict.py --date YYYY-MM-DD --home 主隊 --away 客隊` — 單場互動式預測，
  每次呼叫都會新增一筆紀錄到資料庫，不會覆蓋舊預測
- `python analytics/walkforward.py` — Walk-forward backtest（禁止未來資料外洩，
  依賽季切 fold），輸出 `backtest_output/model_comparison.csv`
- `python analytics/tuning.py` — 超參數搜尋（只用 train/validation，最後一個賽季完全隔離），
  輸出 `best_params.json`
- `python analytics/calibration.py` — Isotonic / Platt 機率校準示範，時間序列安全
- `python data_leakage_check.py` — 7 項自動化 leakage 檢查（不是文字宣稱，是真的斷言）

團隊名稱要用 football-data.co.uk 的拼法（例如 `Man United` 不是 `Man Utd`、
`Tottenham` 不是 `Spurs`），跟雲端網頁那邊的賽程來源命名不同，見 `model/common.py`
的 `FIXTURE_TEAM_ALIASES`。
