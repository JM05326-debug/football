"""
產生 model_report.html。分兩大塊:
  1. Walk-forward backtest 結果（歷史樣本外評估，資料豐富，馬上就有東西可以看）
  2. Production 即時追蹤（db/predictions.db 裡的 predictions，剛開始會很少，
     隨著 update.py 每天執行、真實比賽踢完，才會慢慢累積）

用法:
    python reports/generate_report.py
"""
import base64
import io
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from db import database as db
from analytics import metrics as m
from analytics.model_variants import VARIANT_ORDER

ROOT = pathlib.Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "model_report.html"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def chart_model_comparison(comparison_rows):
    # matplotlib 預設字型沒有中文字形，圖表內一律用英文標籤，避免在沒裝 CJK 字型的機器
    # （包含 GitHub Actions 的 ubuntu runner）上出現缺字方框；中文說明留在圖表外的 HTML 文字。
    fig, ax = plt.subplots(figsize=(6, 3.5))
    variants = [r["model_variant"] for r in comparison_rows]
    log_losses = [r["log_loss"] for r in comparison_rows]
    colors = ["#3b82f6" if v != "E" else "#22c55e" for v in variants]
    ax.bar(variants, log_losses, color=colors)
    ax.set_ylabel("Log Loss (lower = better)")
    ax.set_title("Walk-forward Log Loss by Model Variant")
    ax.set_ylim(min(log_losses) * 0.97, max(log_losses) * 1.02)
    return fig_to_base64(fig)


def chart_fold_trend(fold_rows, variant):
    rows = [r for r in fold_rows if r["model_variant"] == variant]
    rows.sort(key=lambda r: r["fold_index"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([r["fold_index"] for r in rows], [r["accuracy"] * 100 for r in rows], marker="o", label="Accuracy")
    ax2 = ax.twinx()
    ax2.plot([r["fold_index"] for r in rows], [r["log_loss"] for r in rows], marker="s", color="#ef4444", label="Log Loss")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax2.set_ylabel("Log Loss")
    ax.set_title(f"Model {variant} Performance by Fold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    return fig_to_base64(fig)


def chart_calibration_curve(curve_points, title):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    if curve_points:
        xs = [p["mean_predicted"] for p in curve_points]
        ys = [p["actual_frequency"] for p in curve_points]
        sizes = [max(20, p["n"]) for p in curve_points]
        ax.scatter(xs, ys, s=sizes, color="#3b82f6", alpha=0.8, label="actual")
        ax.plot(xs, ys, color="#3b82f6", alpha=0.5)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("actual frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    return fig_to_base64(fig)


def build_live_section(conn):
    counts = db.count_predictions(conn, league="EPL")
    settled = db.fetch_settled_predictions(conn, league="EPL")
    rows_for_metrics = [
        {"prob_H": r["home_win_probability"], "prob_D": r["draw_probability"], "prob_A": r["away_win_probability"],
         "predicted_result": r["predicted_result"], "actual_result": r["actual_result"],
         "over_2_5_probability": r["over_2_5_probability"],
         "actual_home_goals": r["actual_home_goals"], "actual_away_goals": r["actual_away_goals"]}
        for r in settled
    ]
    windows = m.rolling_windows(rows_for_metrics) if rows_for_metrics else None
    return {"counts": counts, "n_settled": len(settled), "windows": windows}


def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_windows_table(windows):
    if not windows:
        return "<p class='muted'>目前資料庫裡還沒有已結算的正式預測（見上方「即時追蹤」說明），先看 walk-forward backtest 的結果。</p>"
    rows_html = ""
    for name, label in (("last10", "最近 10 場"), ("last50", "最近 50 場"), ("last100", "最近 100 場"), ("all", "全部")):
        w = windows[name]
        if w["n"] == 0:
            continue
        rows_html += f"""<tr>
            <td>{label}</td><td>{w['n']}</td>
            <td>{w['accuracy']*100:.1f}%</td>
            <td>{w['log_loss']:.4f}</td>
            <td>{w['brier_score']:.4f}</td>
            <td>{'N/A' if w['home_accuracy'] is None else f"{w['home_accuracy']*100:.1f}%"}</td>
            <td>{'N/A' if w['draw_accuracy'] is None else f"{w['draw_accuracy']*100:.1f}%"}</td>
            <td>{'N/A' if w['away_accuracy'] is None else f"{w['away_accuracy']*100:.1f}%"}</td>
        </tr>"""
    return f"""<table>
        <tr><th>區間</th><th>場數</th><th>Accuracy</th><th>Log Loss</th><th>Brier</th><th>主勝</th><th>和局</th><th>客勝</th></tr>
        {rows_html}
    </table>"""


def main():
    conn = db.connect()

    comparison_path = ROOT / "backtest_output" / "model_comparison.csv"
    import csv
    comparison_rows = []
    if comparison_path.exists():
        with open(comparison_path, encoding="utf-8-sig") as f:
            comparison_rows = list(csv.DictReader(f))
            for r in comparison_rows:
                for k in ("log_loss", "brier_score", "accuracy", "home_accuracy", "away_accuracy", "roi"):
                    r[k] = float(r[k]) if r.get(k) not in (None, "") else None

    fold_rows = db.fetch_backtest_folds(conn, league="EPL")

    leakage_path = ROOT / "backtest_output" / "data_leakage_check_result.json"
    leakage_results = json.loads(leakage_path.read_text(encoding="utf-8")) if leakage_path.exists() else []

    best_params_path = ROOT / "best_params.json"
    best_params = json.loads(best_params_path.read_text(encoding="utf-8")) if best_params_path.exists() else None

    production = db.get_production_model(conn, "EPL")

    live = build_live_section(conn)

    charts = {}
    if comparison_rows:
        sorted_rows = sorted(comparison_rows, key=lambda r: VARIANT_ORDER.index(r["model_variant"]))
        charts["model_comparison"] = chart_model_comparison(sorted_rows)
    if fold_rows:
        charts["fold_trend"] = chart_fold_trend(fold_rows, "E" if any(r["model_variant"] == "E" for r in fold_rows) else "D")

    calibration_curve_html = ""
    calib_result_path = ROOT / "backtest_output" / "calibration_result.json"
    if calib_result_path.exists():
        calib = json.loads(calib_result_path.read_text(encoding="utf-8"))
        charts["calibration"] = chart_calibration_curve(calib["raw_curve_H"], "Calibration curve (home win prob., raw)")

    comparison_table_rows = ""
    for r in comparison_rows:
        comparison_table_rows += f"""<tr {"class='best-row'" if r['model_variant']=='E' else ''}>
            <td>{r['model_variant']}</td><td>{html_escape(r['label'])}</td>
            <td>{r['log_loss']:.4f}</td><td>{r['brier_score']:.4f}</td><td>{r['accuracy']*100:.1f}%</td>
            <td>{'N/A' if r['home_accuracy'] is None else f"{r['home_accuracy']*100:.1f}%"}</td>
            <td>{'N/A' if r.get('draw_accuracy') in (None, '') else f"{float(r['draw_accuracy'])*100:.1f}%"}</td>
            <td>{'N/A' if r['away_accuracy'] is None else f"{r['away_accuracy']*100:.1f}%"}</td>
            <td>{'N/A' if r['roi'] is None else f"{r['roi']*100:.1f}%"}</td>
        </tr>"""

    leakage_rows = "".join(
        f"<tr><td>{'PASS' if c['passed'] else 'FAIL'}</td><td>{html_escape(c['name'])}</td>"
        f"<td class='muted'>{html_escape(c['detail'])}</td></tr>"
        for c in leakage_results
    )
    n_leak_pass = sum(1 for c in leakage_results if c["passed"])

    html = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>模型評估報告</title>
<style>
body {{ font-family: -apple-system, "Microsoft JhengHei", sans-serif; max-width: 980px; margin: 0 auto; padding: 24px; color:#0f172a; background:#f8fafc; }}
h1 {{ font-size: 1.5rem; }}
h2 {{ font-size: 1.15rem; margin-top: 36px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.85rem; background:white; }}
th, td {{ border: 1px solid #e2e8f0; padding: 6px 10px; text-align: center; }}
th {{ background: #f1f5f9; }}
.best-row {{ background: #dcfce7; font-weight: 600; }}
.muted {{ color: #64748b; font-size: 0.8rem; text-align: left; }}
.summary-cards {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px; flex: 1 1 160px; }}
.card .value {{ font-size: 1.6rem; font-weight: 700; }}
.card .label {{ font-size: 0.8rem; color: #64748b; }}
img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; background: white; }}
.pass {{ color: #16a34a; font-weight: 700; }}
.fail {{ color: #dc2626; font-weight: 700; }}
.charts {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.charts > div {{ flex: 1 1 400px; }}
</style></head>
<body>
<h1>⚽ 足球預測模型評估報告</h1>
<p class="muted">產生時間: {db.now_iso()}｜Production 版本: {production['version'] if production else '尚未訓練'}</p>

<h2>Production 摘要</h2>
<div class="summary-cards">
  <div class="card"><div class="value">{production['version'] if production else 'N/A'}</div><div class="label">Production 版本</div></div>
  <div class="card"><div class="value">{live['counts'].get('pending', 0) + live['counts'].get('settled', 0)}</div><div class="label">總預測場數</div></div>
  <div class="card"><div class="value">{live['counts'].get('settled', 0)}</div><div class="label">已完成場數</div></div>
  <div class="card"><div class="value">{live['counts'].get('pending', 0)}</div><div class="label">Pending 場數</div></div>
</div>

<h2>即時預測追蹤（db/predictions.db，隨每天 update.py 累積）</h2>
{render_windows_table(live['windows'])}

<h2>Walk-forward Backtest：Model A~E 比較（歷史樣本外測試，1520 場/model，4 個 fold）</h2>
<table>
<tr><th>Model</th><th>說明</th><th>Log Loss</th><th>Brier</th><th>Accuracy</th><th>主勝</th><th>和局</th><th>客勝</th><th>ROI（用市場賠率）</th></tr>
{comparison_table_rows}
</table>
<p class="muted">綠色列 = Model E（完整模型 + 調校過的超參數），照 Log Loss → Brier → Accuracy 排序。
「和局」欄位常常是 N/A：Poisson 類模型幾乎不會把和局算成單場機率最高的結果，這不是 bug，是已知限制。</p>

<div class="charts">
  <div>{f'<img src="data:image/png;base64,{charts["model_comparison"]}">' if "model_comparison" in charts else ''}</div>
  <div>{f'<img src="data:image/png;base64,{charts["fold_trend"]}">' if "fold_trend" in charts else ''}</div>
</div>

{f'''<h2>機率校準曲線</h2><div class="charts"><div><img src="data:image/png;base64,{charts["calibration"]}"></div></div>''' if "calibration" in charts else ''}

<h2>Data Leakage Check：{n_leak_pass}/{len(leakage_results)} 項通過</h2>
<table>
<tr><th>結果</th><th>檢查項目</th><th>細節</th></tr>
{leakage_rows}
</table>

<h2>目前 Production 參數</h2>
<pre style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px;overflow-x:auto;">{html_escape(json.dumps(best_params, ensure_ascii=False, indent=2)) if best_params else '尚未調校（使用預設參數）'}</pre>

</body></html>"""

    OUTPUT_PATH.write_text(html, encoding="utf-8")
    conn.close()
    print(f"已輸出: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
