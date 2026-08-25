"""
本地伺服器：
  1. 提供 web/ 資料夾的網頁（取代直接雙擊 index.html）
  2. 提供 POST /api/update，讓網頁上的「更新最新數據」按鈕可以
     觸發抓資料 + 重新訓練模型的腳本，不需要手動開終端機

用法: 在 match_predictor 資料夾下執行
    python server.py
然後瀏覽器打開 http://localhost:8000
"""
import json
import pathlib
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).parent
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
PORT = 8000


def run_script(cwd: pathlib.Path, script: str) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        return {
            "script": script,
            "ok": result.returncode == 0,
            "output": (result.stdout or "") + (result.stderr or ""),
        }
    except subprocess.TimeoutExpired:
        return {
            "script": script,
            "ok": False,
            "output": f"執行超過 300 秒逾時（可能是網路太慢或資料源沒回應），已中止。",
        }
    except Exception as e:
        return {
            "script": script,
            "ok": False,
            "output": f"執行時發生例外: {e}",
        }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        # 避免瀏覽器快取到舊的 model_data.js / fixtures_data.js
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def do_POST(self):
        if self.path != "/api/update":
            self.send_response(404)
            self.end_headers()
            return

        steps = []
        try:
            steps.append(run_script(DATA_DIR, "fetch_epl_data.py"))
            steps.append(run_script(DATA_DIR, "fetch_fixtures.py"))
            steps.append(run_script(MODEL_DIR, "train.py"))
            steps.append(run_script(MODEL_DIR, "lock_fixture_predictions.py"))
            all_ok = all(s["ok"] for s in steps)
            body = json.dumps({"success": all_ok, "steps": steps}, ensure_ascii=False).encode("utf-8")
            self.send_response(200 if all_ok else 500)
        except Exception as e:
            body = json.dumps({"success": False, "steps": steps, "error": str(e)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)

        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    print(f"伺服器已啟動: http://localhost:{PORT}")
    print("按 Ctrl+C 可以停止伺服器")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n伺服器已停止")


if __name__ == "__main__":
    main()
