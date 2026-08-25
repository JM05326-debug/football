// 英超賽程總覽：不用手動選隊，直接列出整季賽程跟鎖定預測。
// H/D/A 機率、期望進球、Top 5 比分都是 model/lock_fixture_predictions.py 鎖定當下算好、
// 存在 fixtures_data.js 裡的值，這裡完全不重新計算、不會因為模型重新訓練而改變。
// 大小分 2.5 跟總進球分布區間是「用同一組已鎖定的期望進球 + 目前模型的 rho」在瀏覽器端
// 額外算出來的顯示用資訊，數學上與鎖定當下一致，只是 rho 這個小幅修正值沒有跟著鎖定存檔。

const MAX_GOALS = 8;
const TOP_SCORES_SHOWN = 3;

function poissonPmf(k, lambda) {
  let factorial = 1;
  for (let i = 2; i <= k; i++) factorial *= i;
  return Math.exp(-lambda) * Math.pow(lambda, k) / factorial;
}

function dixonColesTau(x, y, lamH, lamA, rho) {
  if (x === 0 && y === 0) return 1 - lamH * lamA * rho;
  if (x === 0 && y === 1) return 1 + lamH * rho;
  if (x === 1 && y === 0) return 1 + lamA * rho;
  if (x === 1 && y === 1) return 1 - rho;
  return 1;
}

function scoreMatrixFromExpectedGoals(lamH, lamA, rho) {
  const matrix = [];
  let total = 0;
  for (let h = 0; h <= MAX_GOALS; h++) {
    const row = [];
    for (let a = 0; a <= MAX_GOALS; a++) {
      const p = Math.max(poissonPmf(h, lamH) * poissonPmf(a, lamA) * dixonColesTau(h, a, lamH, lamA, rho), 0);
      row.push(p);
      total += p;
    }
    matrix.push(row);
  }
  for (let h = 0; h <= MAX_GOALS; h++)
    for (let a = 0; a <= MAX_GOALS; a++)
      matrix[h][a] /= total;
  return matrix;
}

function over2_5(matrix) {
  let over = 0;
  for (let h = 0; h <= MAX_GOALS; h++)
    for (let a = 0; a <= MAX_GOALS; a++)
      if (h + a > 2.5) over += matrix[h][a];
  return over;
}

function totalGoalsBuckets(matrix) {
  const buckets = { "0-1": 0, "2-3": 0, "4-5": 0, "6+": 0 };
  for (let h = 0; h <= MAX_GOALS; h++) {
    for (let a = 0; a <= MAX_GOALS; a++) {
      const total = h + a;
      const p = matrix[h][a];
      if (total <= 1) buckets["0-1"] += p;
      else if (total <= 3) buckets["2-3"] += p;
      else if (total <= 5) buckets["4-5"] += p;
      else buckets["6+"] += p;
    }
  }
  return buckets;
}

function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

function formatFixtureDate(dateUtc) {
  const d = new Date(dateUtc.replace(" ", "T"));
  return d.toLocaleString("zh-Hant", {
    month: "2-digit", day: "2-digit", weekday: "short", hour: "2-digit", minute: "2-digit",
  });
}

const RESULT_LABEL = { H: "主勝", D: "和局", A: "客勝" };

function fixtureCardHtml(f) {
  const home = zhName(f.home_team);
  const away = zhName(f.away_team);
  const rho = EPL_MODEL_DATA.rho;
  const matrix = scoreMatrixFromExpectedGoals(f.expected_home_goals, f.expected_away_goals, rho);
  const over = over2_5(matrix);
  const buckets = totalGoalsBuckets(matrix);
  const topScores = f.top_scores.slice(0, TOP_SCORES_SHOWN);

  const lowConfBadge = f.low_confidence
    ? '<span class="low-confidence-badge" title="缺乏該隊英超歷史資料（例如剛升班），預測信心度較低">⚠️升班馬</span>'
    : "";

  const actualHtml = f.actual
    ? `<div class="fixture-actual">
        <span class="result-badge ${f.actual.correct ? "correct" : "wrong"}">${f.actual.correct ? "✓ 命中" : "✗ 未命中"}</span>
        <span>實際比分 ${f.actual.home_goals} - ${f.actual.away_goals}（${RESULT_LABEL[f.actual.result]}）</span>
      </div>`
    : "";

  return `
  <article class="fixture-card ${f.actual ? "settled" : ""}">
    <div class="fixture-head">
      <span class="fixture-date">${formatFixtureDate(f.date_utc)}</span>
      <span class="fixture-teams">${home} <span class="vs-sep">vs</span> ${away}${lowConfBadge}</span>
    </div>

    <div class="outcome-bar small">
      <div class="outcome-seg home" style="flex-basis:${pct(f.prob_home)}"><span>${pct(f.prob_home)}</span></div>
      <div class="outcome-seg draw" style="flex-basis:${pct(f.prob_draw)}"><span>${pct(f.prob_draw)}</span></div>
      <div class="outcome-seg away" style="flex-basis:${pct(f.prob_away)}"><span>${pct(f.prob_away)}</span></div>
    </div>
    <div class="outcome-labels small">
      <span>${home} 勝</span><span>和局</span><span>${away} 勝</span>
    </div>

    <div class="fixture-stats">
      <div class="stat-block">
        <div class="stat-label">總進球大小分 2.5</div>
        <div class="stat-value">大 ${pct(over)}／小 ${pct(1 - over)}</div>
      </div>
      <div class="stat-block">
        <div class="stat-label">各隊預期進球</div>
        <div class="stat-value">${home} ${f.expected_home_goals.toFixed(1)}／${away} ${f.expected_away_goals.toFixed(1)}</div>
      </div>
      <div class="stat-block">
        <div class="stat-label">最終比分預測 Top ${TOP_SCORES_SHOWN}</div>
        <div class="stat-value">${topScores.map((s) => `${s.h}-${s.a} (${pct(s.p)})`).join("、")}</div>
      </div>
      <div class="stat-block">
        <div class="stat-label">全場總進球數分布</div>
        <div class="stat-value">0-1: ${pct(buckets["0-1"])}／2-3: ${pct(buckets["2-3"])}／4-5: ${pct(buckets["4-5"])}／6+: ${pct(buckets["6+"])}</div>
      </div>
    </div>

    ${actualHtml}
  </article>`;
}

function renderFixtures() {
  const listEl = document.getElementById("fixturesList");
  const summaryEl = document.getElementById("fixturesSummary");

  if (typeof EPL_FIXTURES === "undefined" || !EPL_FIXTURES.length) {
    summaryEl.textContent = "目前沒有賽程資料";
    return;
  }

  const settled = EPL_FIXTURES.filter((f) => f.actual);
  const correctCount = settled.filter((f) => f.actual.correct).length;
  summaryEl.textContent = settled.length
    ? `本賽季共鎖定 ${EPL_FIXTURES.length} 場｜已完賽 ${settled.length} 場，命中 ${correctCount} 場（${pct(correctCount / settled.length)}）`
    : `本賽季共鎖定 ${EPL_FIXTURES.length} 場｜賽季尚未開始結算戰績`;

  const rounds = new Map();
  for (const f of EPL_FIXTURES) {
    if (!rounds.has(f.round)) rounds.set(f.round, []);
    rounds.get(f.round).push(f);
  }
  const roundNumbers = Array.from(rounds.keys()).sort((a, b) => a - b);

  let html = "";
  for (const round of roundNumbers) {
    const fixtures = rounds.get(round);
    const hasUnsettled = fixtures.some((f) => !f.actual);
    const roundSettled = fixtures.filter((f) => f.actual).length;
    html += `<details class="round-group" ${hasUnsettled ? "open" : ""}>
      <summary>第 ${round} 輪（${roundSettled}/${fixtures.length} 場已完賽）</summary>
      <div class="round-fixtures">
        ${fixtures.map(fixtureCardHtml).join("")}
      </div>
    </details>`;
  }
  listEl.innerHTML = html;
}

function updateModelInfo() {
  document.getElementById("modelInfo").textContent =
    `模型: ${EPL_MODEL_DATA.league} | 訓練時間: ${EPL_MODEL_DATA.trained_at} | 使用 ${EPL_MODEL_DATA.matches_used} 場比賽資料`;
}

async function updateData() {
  const btn = document.getElementById("updateBtn");
  const status = document.getElementById("updateStatus");

  btn.disabled = true;
  status.classList.remove("error", "success");
  status.textContent = "更新中，抓取最新賽果並重新訓練模型，可能需要 1-3 分鐘，請耐心等待...";

  try {
    const resp = await fetch("/api/update", { method: "POST" });
    const data = await resp.json();

    if (data.success) {
      status.classList.add("success");
      status.textContent = "更新完成！3 秒後自動重新整理頁面套用最新數據...";
      setTimeout(() => location.reload(), 3000);
    } else {
      status.classList.add("error");
      const failedSteps = (data.steps || []).filter((s) => !s.ok);
      const detail = failedSteps
        .map((s) => `${s.script}: ${(s.output || "").trim().slice(-300)}`)
        .join("\n");
      status.textContent = detail
        ? `更新失敗：\n${detail}`
        : `更新失敗：${data.error || "未知錯誤"}`;
      btn.disabled = false;
    }
  } catch (err) {
    status.classList.add("error");
    status.textContent =
      "無法連線到更新伺服器。請確認你是用 server.py 啟動這個網頁（雙擊 index.html 打開的話無法使用此功能）。";
    btn.disabled = false;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const isLocal = location.hostname === "localhost" || location.hostname === "127.0.0.1";
  document.getElementById("localUpdatePanel").classList.toggle("hidden", !isLocal);
  document.getElementById("cloudUpdatePanel").classList.toggle("hidden", isLocal);
  if (isLocal) {
    document.getElementById("updateBtn").addEventListener("click", updateData);
  }

  renderFixtures();
  updateModelInfo();
});
