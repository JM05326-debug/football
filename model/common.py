"""
model/train.py 的訓練邏輯，統一欄位:
    date, home_team, away_team, home_goals, away_goals

新增的三個特徵:
  - Elo Rating: 逐場比賽依序更新，比分差距越大調整幅度越大（沿用世界足球 Elo 排名的做法）
  - 最近 N 場戰績 (recent form): 每隊最近 10 場的場均得失球，
    換算成相對於該隊長期平均的「近期手感」係數，供預測時做小幅加權調整
  - 主客場細分數據: 主場/客場各自的場均進球、失球、戰績，純粹給網頁顯示用
"""
import math

import numpy as np

RECENT_N = 10
FORM_CLAMP = (0.7, 1.3)  # 避免近期樣本數少（10場）時，手感係數過度膨脹

INITIAL_ELO = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 100.0

PRIOR_GAMES = 4.0  # shrinkage 用的「虛擬場次」，場次越少的隊伍強度越往聯盟平均拉


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def compute_weights(dates, half_life_days):
    latest = dates.max()
    age_days = (latest - dates).dt.days
    return 0.5 ** (age_days / half_life_days)


def fit_team_strengths(df, weights, prior_games=PRIOR_GAMES):
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))

    league_avg_home_goals = (df["home_goals"] * weights).sum() / weights.sum()
    league_avg_away_goals = (df["away_goals"] * weights).sum() / weights.sum()

    stats = {t: {"home_gf_w": 0.0, "home_ga_w": 0.0, "home_w": 0.0,
                 "away_gf_w": 0.0, "away_ga_w": 0.0, "away_w": 0.0} for t in teams}

    for idx, row in df.iterrows():
        w = weights.iloc[idx]
        h, a = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        stats[h]["home_gf_w"] += hg * w
        stats[h]["home_ga_w"] += ag * w
        stats[h]["home_w"] += w
        stats[a]["away_gf_w"] += ag * w
        stats[a]["away_ga_w"] += hg * w
        stats[a]["away_w"] += w

    ratings = {}
    for t in teams:
        s = stats[t]
        home_w = s["home_w"] if s["home_w"] > 0 else 1e-9
        away_w = s["away_w"] if s["away_w"] > 0 else 1e-9

        raw_attack_home = (s["home_gf_w"] / home_w) / league_avg_home_goals
        raw_defense_home = (s["home_ga_w"] / home_w) / league_avg_away_goals
        raw_attack_away = (s["away_gf_w"] / away_w) / league_avg_away_goals
        raw_defense_away = (s["away_ga_w"] / away_w) / league_avg_home_goals

        def shrink(raw, n):
            return (raw * n + 1.0 * prior_games) / (n + prior_games)

        ratings[t] = {
            "attack_home": shrink(raw_attack_home, home_w),
            "defense_home": shrink(raw_defense_home, home_w),
            "attack_away": shrink(raw_attack_away, away_w),
            "defense_away": shrink(raw_defense_away, away_w),
            "games_played": float(s["home_w"] + s["away_w"]),
        }

    return ratings, league_avg_home_goals, league_avg_away_goals


def poisson_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dixon_coles_tau(x, y, lam_h, lam_a, rho):
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def match_log_likelihood(df, weights, ratings, league_avg_home_goals, league_avg_away_goals, rho):
    total = 0.0
    for idx, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        rh, ra = ratings[h], ratings[a]
        lam_h = league_avg_home_goals * rh["attack_home"] * ra["defense_away"]
        lam_a = league_avg_away_goals * ra["attack_away"] * rh["defense_home"]
        p = poisson_pmf(hg, lam_h) * poisson_pmf(ag, lam_a)
        tau = dixon_coles_tau(hg, ag, lam_h, lam_a, rho)
        p = max(p * tau, 1e-10)
        total += weights.iloc[idx] * math.log(p)
    return total


def fit_rho(df, weights, ratings, league_avg_home_goals, league_avg_away_goals,
            rho_search_min=-30, rho_search_max=10):
    """跟 match_log_likelihood 數學上完全等價，但用 numpy 向量化取代 df.iterrows()，
    因為 rho 搜尋要對同一份訓練資料重複算幾十次概似值，iterrows 版本在 backtest /
    參數搜尋（要重複 fit 上百次）會慢到不可行。等價性由 test_common.py 驗證。"""
    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    hg = df["home_goals"].to_numpy(dtype=float)
    ag = df["away_goals"].to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float) if hasattr(weights, "to_numpy") else np.asarray(weights, dtype=float)

    attack_home = np.array([ratings[t]["attack_home"] for t in home])
    defense_away = np.array([ratings[t]["defense_away"] for t in away])
    attack_away = np.array([ratings[t]["attack_away"] for t in away])
    defense_home = np.array([ratings[t]["defense_home"] for t in home])

    lam_h = league_avg_home_goals * attack_home * defense_away
    lam_a = league_avg_away_goals * attack_away * defense_home

    log_fact_h = np.array([math.lgamma(x + 1) for x in hg])
    log_fact_a = np.array([math.lgamma(x + 1) for x in ag])
    log_poisson = (hg * np.log(lam_h) - lam_h - log_fact_h) + (ag * np.log(lam_a) - lam_a - log_fact_a)

    best_rho, best_ll = 0.0, -math.inf
    for rho_int in range(rho_search_min, rho_search_max + 1):
        rho = rho_int / 100.0
        tau = np.ones_like(lam_h)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
        tau[m01] = 1 + lam_h[m01] * rho
        tau[m10] = 1 + lam_a[m10] * rho
        tau[m11] = 1 - rho

        p = np.exp(log_poisson) * tau
        p = np.maximum(p, 1e-10)
        ll = float(np.sum(w * np.log(p)))
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return best_rho


def compute_elo_ratings(df, elo_k=ELO_K, elo_home_adv=ELO_HOME_ADV):
    """依照比赛日期先後逐場更新 Elo，比分差距越大調整幅度越大（G 值），
    主場有固定的 Elo 主場優勢加成。回傳 dict: team -> 最終 Elo。
    elo_k / elo_home_adv 可覆寫，供 backtest / 參數搜尋使用不同設定。"""
    elo = {}

    def get(t):
        return elo.setdefault(t, INITIAL_ELO)

    for _, row in df.sort_values("date").iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        rh, ra = get(home), get(away)

        expected_home = 1 / (1 + 10 ** (-((rh + elo_home_adv) - ra) / 400))
        if hg > ag:
            score_home = 1.0
        elif hg == ag:
            score_home = 0.5
        else:
            score_home = 0.0

        gd = abs(hg - ag)
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)

        delta = elo_k * g * (score_home - expected_home)
        elo[home] = rh + delta
        elo[away] = ra - delta

    return elo


MAX_GOALS = 8  # 必須與 web/app.js 的 MAX_GOALS 一致
FORM_WEIGHT = 0.3  # 必須與 web/app.js 的 FORM_WEIGHT 一致

# 沒有英超歷史資料的球隊（例如剛升班的球隊）使用聯盟平均值當作預設強度，
# 讓賽程鎖定預測仍可產生結果，但會被標記為低信心度（見 lock_fixture_predictions.py）
DEFAULT_RATING = {
    "attack_home": 1.0, "defense_home": 1.0,
    "attack_away": 1.0, "defense_away": 1.0,
    "elo": INITIAL_ELO, "recent_form": None, "games_played": 0.0,
}

# 賽程資料來源（fixturedownload.com）與歷史比賽資料來源（football-data.co.uk）
# 兩邊球隊命名不完全一致，這裡列出已知的對照，其餘名稱直接視為相同
FIXTURE_TEAM_ALIASES = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}


def canonical_team_name(fixture_team_name):
    return FIXTURE_TEAM_ALIASES.get(fixture_team_name, fixture_team_name)


def _form_multiplier(team_rating, key, form_weight):
    recent_form = team_rating.get("recent_form")
    factor = recent_form[key] if recent_form else 1.0
    return 1 + form_weight * (factor - 1)


def predict_match(home_rating, away_rating, league_avg_home_goals, league_avg_away_goals, rho,
                   use_form=True, form_weight=FORM_WEIGHT):
    """複製 web/app.js 的 Dixon-Coles 預測邏輯（不含中立場地模式，鎖定賽程一律有主場）。
    use_form=False 時完全不套用近期手感調整（給 backtest 的 Model A/B 用）。
    回傳: prob_home/prob_draw/prob_away、期望進球、最可能比分 Top 5。"""
    if use_form:
        home_attack_mult = _form_multiplier(home_rating, "attack_factor", form_weight)
        away_defense_mult = _form_multiplier(away_rating, "defense_factor", form_weight)
        away_attack_mult = _form_multiplier(away_rating, "attack_factor", form_weight)
        home_defense_mult = _form_multiplier(home_rating, "defense_factor", form_weight)
    else:
        home_attack_mult = away_defense_mult = away_attack_mult = home_defense_mult = 1.0

    lam_h = (
        league_avg_home_goals
        * home_rating["attack_home"]
        * away_rating["defense_away"]
        * home_attack_mult
        * away_defense_mult
    )
    lam_a = (
        league_avg_away_goals
        * away_rating["attack_away"]
        * home_rating["defense_home"]
        * away_attack_mult
        * home_defense_mult
    )

    matrix = {}
    total = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = poisson_pmf(h, lam_h) * poisson_pmf(a, lam_a) * dixon_coles_tau(h, a, lam_h, lam_a, rho)
            p = max(p, 0.0)
            matrix[(h, a)] = p
            total += p

    p_home = p_draw = p_away = p_over25 = 0.0
    scores = []
    for (h, a), p in matrix.items():
        p /= total
        if h > a:
            p_home += p
        elif h == a:
            p_draw += p
        else:
            p_away += p
        if h + a > 2.5:
            p_over25 += p
        scores.append({"h": h, "a": a, "p": p})

    scores.sort(key=lambda s: s["p"], reverse=True)

    return {
        "prob_home": p_home,
        "prob_draw": p_draw,
        "prob_away": p_away,
        "prob_over_2_5": p_over25,
        "prob_under_2_5": 1 - p_over25,
        "expected_home_goals": lam_h,
        "expected_away_goals": lam_a,
        "top_scores": scores[:5],
    }


def elo_outcome_probs(home_elo, away_elo, draw_rate, elo_home_adv=ELO_HOME_ADV):
    """Elo 版本的勝/和/負機率估計（見 web/app.js 的 computeEloEstimate）：
    用 Elo 期望得分當作「不算平局時的勝率」，平局率用歷史平均和局率。"""
    elo_diff = home_elo + elo_home_adv - away_elo
    expected_home_no_draw = 1 / (1 + 10 ** (-elo_diff / 400))
    return {
        "H": expected_home_no_draw * (1 - draw_rate),
        "D": draw_rate,
        "A": (1 - expected_home_no_draw) * (1 - draw_rate),
    }


def blend_probs(poisson_probs: dict, elo_probs: dict, elo_weight: float) -> dict:
    """poisson_probs / elo_probs 都是 {'H':..,'D':..,'A':..}，線性加權混合後正規化。"""
    blended = {
        k: (1 - elo_weight) * poisson_probs[k] + elo_weight * elo_probs[k]
        for k in ("H", "D", "A")
    }
    total = sum(blended.values())
    return {k: v / total for k, v in blended.items()}


def compute_recent_form_and_splits(df):
    """回傳兩個 dict:
      recent_form: team -> 最近 RECENT_N 場的場均戰績，以及相對長期平均的「手感」係數
      home_away_splits: team -> 主場/客場個別的場均進球失球與戰績（純顯示用）
    """
    history = {}  # team -> list of (date, gf, ga, is_home)
    for _, row in df.iterrows():
        date = row["date"]
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        history.setdefault(home, []).append((date, hg, ag, True))
        history.setdefault(away, []).append((date, ag, hg, False))

    recent_form = {}
    home_away_splits = {}

    for team, games in history.items():
        games_sorted = sorted(games, key=lambda g: g[0])

        total_gf = sum(g[1] for g in games_sorted)
        total_ga = sum(g[2] for g in games_sorted)
        n_total = len(games_sorted)
        overall_gf_pg = total_gf / n_total
        overall_ga_pg = total_ga / n_total

        recent = games_sorted[-RECENT_N:]
        n_recent = len(recent)
        recent_gf = sum(g[1] for g in recent)
        recent_ga = sum(g[2] for g in recent)
        pts = sum(3 if g[1] > g[2] else (1 if g[1] == g[2] else 0) for g in recent)
        wins = sum(1 for g in recent if g[1] > g[2])
        draws = sum(1 for g in recent if g[1] == g[2])
        losses = n_recent - wins - draws

        recent_gf_pg = recent_gf / n_recent
        recent_ga_pg = recent_ga / n_recent

        attack_factor = clamp(recent_gf_pg / overall_gf_pg, *FORM_CLAMP) if overall_gf_pg > 0 else 1.0
        defense_factor = clamp(recent_ga_pg / overall_ga_pg, *FORM_CLAMP) if overall_ga_pg > 0 else 1.0

        recent_form[team] = {
            "games": n_recent,
            "record": f"{wins}勝{draws}平{losses}負",
            "points_per_game": round(pts / n_recent, 2),
            "gf_per_game": round(recent_gf_pg, 2),
            "ga_per_game": round(recent_ga_pg, 2),
            "attack_factor": round(attack_factor, 3),
            "defense_factor": round(defense_factor, 3),
        }

        home_games = [g for g in games_sorted if g[3]]
        away_games = [g for g in games_sorted if not g[3]]

        def summarize(games_list):
            n = len(games_list)
            if n == 0:
                return {"games": 0, "gf_per_game": 0, "ga_per_game": 0, "record": "0勝0平0負"}
            gf = sum(g[1] for g in games_list)
            ga = sum(g[2] for g in games_list)
            w = sum(1 for g in games_list if g[1] > g[2])
            d = sum(1 for g in games_list if g[1] == g[2])
            l = n - w - d
            return {
                "games": n,
                "gf_per_game": round(gf / n, 2),
                "ga_per_game": round(ga / n, 2),
                "record": f"{w}勝{d}平{l}負",
            }

        home_away_splits[team] = {
            "home": summarize(home_games),
            "away": summarize(away_games),
        }

    return recent_form, home_away_splits
