"""
Model A~E 的定義，以及「用訓練資料 fit 一個模型、對測試比賽逐場預測」的共用邏輯。

所有模型都建立在同一組 model/common.py 的 Dixon-Coles + shrinkage 引擎上，
差異只在於：要不要時間衰減加權、要不要近期手感調整、要不要混合 Elo 機率。
這樣比較才公平 —— 差異只來自「有沒有加這個機制」，不是換了完全不同的建模方式。

    A: Dixon-Coles + shrinkage（均勻權重，無 Elo、無近期手感、無時間衰減）
    B: A + Elo（機率層面 50/50 混合 Elo 估計）
    C: B + 近期手感調整
    D: C + 時間衰減加權（= 目前網頁/鎖定預測用的邏輯，只是額外混合了 Elo）
    E: D 但用 tuning.py 找出的 best_params.json，而不是寫死的預設值

正式產品（web/app.js、model/lock_fixture_predictions.py）本身其實沒有混合 Elo 到
最終機率裡（Elo 只是網頁上另外顯示的參考指標），所以嚴格說目前 production 邏輯
最接近 Model D 拿掉 Elo 混合的版本，跟 C/D 不完全相同 —— 這點在 model_comparison.csv
的比較報告裡會註明，不要誤會 D 就是「現在網站在用的東西」。
"""
import sys
import pathlib

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "model"))
import common  # noqa: E402  (model/common.py)

DEFAULT_PARAMS = {
    "half_life_days": 380.0,
    "prior_games": 4.0,
    "form_weight": common.FORM_WEIGHT,
    "elo_k": common.ELO_K,
    "elo_home_adv": common.ELO_HOME_ADV,
    "elo_weight": 0.5,   # Model B/C/D/E 用；A 固定為 0
    "rho_search_min": -30,
    "rho_search_max": 10,
}

VARIANT_ORDER = ["A", "B", "C", "D", "E"]

VARIANT_LABELS = {
    "A": "Dixon-Coles（無 Elo/近期手感/時間衰減）",
    "B": "Dixon-Coles + Elo",
    "C": "Dixon-Coles + Elo + 近期手感",
    "D": "Dixon-Coles + Elo + 近期手感 + 時間衰減",
    "E": "完整模型（D + 調校過的超參數）",
}


def variant_flags(variant: str) -> dict:
    return {
        "use_decay": variant in ("D", "E"),
        "use_form": variant in ("C", "D", "E"),
        "use_elo": variant in ("B", "C", "D", "E"),
    }


def fit_model(train_df: pd.DataFrame, params: dict, flags: dict):
    """train_df 必須已經是「只包含這個 fold 訓練期間之前的比賽」，呼叫端負責切好，
    這裡完全不會再往後看。回傳一個 model dict，供 predict_one() 使用。"""
    train_df = train_df.reset_index(drop=True)

    if flags["use_decay"]:
        weights = common.compute_weights(train_df["date"], params["half_life_days"])
    else:
        weights = pd.Series(1.0, index=train_df.index)

    ratings, avg_home, avg_away = common.fit_team_strengths(train_df, weights, prior_games=params["prior_games"])
    rho = common.fit_rho(
        train_df, weights, ratings, avg_home, avg_away,
        rho_search_min=params.get("rho_search_min", -30), rho_search_max=params.get("rho_search_max", 10),
    )

    elo = common.compute_elo_ratings(train_df, elo_k=params["elo_k"], elo_home_adv=params["elo_home_adv"])
    recent_form, _ = common.compute_recent_form_and_splits(train_df)
    draw_rate = float((train_df["home_goals"] == train_df["away_goals"]).mean())

    for t in ratings:
        ratings[t]["elo"] = elo.get(t, common.INITIAL_ELO)
        ratings[t]["recent_form"] = recent_form.get(t)

    return {
        "teams": ratings,
        "avg_home": avg_home,
        "avg_away": avg_away,
        "rho": rho,
        "draw_rate": draw_rate,
        "train_matches": len(train_df),
        "train_end_date": train_df["date"].max(),
    }


def predict_one(model: dict, home_team: str, away_team: str, params: dict, flags: dict) -> dict:
    home_rating = model["teams"].get(home_team, common.DEFAULT_RATING)
    away_rating = model["teams"].get(away_team, common.DEFAULT_RATING)

    poisson = common.predict_match(
        home_rating, away_rating, model["avg_home"], model["avg_away"], model["rho"],
        use_form=flags["use_form"], form_weight=params["form_weight"],
    )
    poisson_probs = {"H": poisson["prob_home"], "D": poisson["prob_draw"], "A": poisson["prob_away"]}

    if flags["use_elo"] and params["elo_weight"] > 0:
        elo_probs = common.elo_outcome_probs(
            home_rating.get("elo", common.INITIAL_ELO), away_rating.get("elo", common.INITIAL_ELO),
            model["draw_rate"], elo_home_adv=params["elo_home_adv"],
        )
        final_probs = common.blend_probs(poisson_probs, elo_probs, params["elo_weight"])
    else:
        final_probs = poisson_probs

    predicted_result = max(final_probs, key=final_probs.get)
    best_score = poisson["top_scores"][0]

    return {
        "prob_H": final_probs["H"],
        "prob_D": final_probs["D"],
        "prob_A": final_probs["A"],
        "predicted_result": predicted_result,
        "predicted_score": f"{best_score['h']}-{best_score['a']}",
        "top_scores": poisson["top_scores"],
        "over_2_5_probability": poisson["prob_over_2_5"],
        "under_2_5_probability": poisson["prob_under_2_5"],
        "expected_home_goals": poisson["expected_home_goals"],
        "expected_away_goals": poisson["expected_away_goals"],
        "low_confidence": home_team not in model["teams"] or away_team not in model["teams"],
    }
