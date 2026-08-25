"""
互動式預測 CLI。

用法:
    python predict.py
    python predict.py --date 2026-08-22 --home Arsenal --away Chelsea

每一次呼叫都會產生一筆新的 prediction 並寫入 db/predictions.db（不會覆蓋舊的），
用的是目前的 production 模型（model_versions 表裡 is_production=1 的那一版；
如果還沒有任何 production 版本，會先跑一次 analytics/registry.py 訓練出 v1.0）。
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from db import database as db
from analytics.registry import fit_production_model, train_and_maybe_promote


def get_or_train_production_model(league="EPL"):
    conn = db.connect()
    prod = db.get_production_model(conn, league)
    conn.close()
    if prod is None:
        print("目前沒有 production 模型，先訓練一個...")
        train_and_maybe_promote(league)
    return fit_production_model(league)


def predict_and_store(model, params, version, match_date, home_team, away_team, league="EPL"):
    from analytics.model_variants import predict_one, variant_flags
    flags = variant_flags("D")  # production 用完整功能集合（跟 registry.PRODUCTION_FLAGS 一致）

    pred = predict_one(model, home_team, away_team, params, flags)

    conn = db.connect()
    prediction_id = db.insert_prediction(
        conn,
        match_date=match_date, league=league, home_team=home_team, away_team=away_team,
        home_win_probability=pred["prob_H"], draw_probability=pred["prob_D"], away_win_probability=pred["prob_A"],
        predicted_result=pred["predicted_result"], predicted_score=pred["predicted_score"],
        over_2_5_probability=pred["over_2_5_probability"], under_2_5_probability=pred["under_2_5_probability"],
        model_version=version,
    )
    conn.commit()
    conn.close()
    return prediction_id, pred


def print_prediction(home_team, away_team, pred, version):
    print(f"\n=== {home_team} vs {away_team} ===")
    print(f"模型版本: {version}")
    if pred["low_confidence"]:
        print("[注意] 至少一隊缺乏足夠的英超歷史資料（例如剛升班），預測信心度較低")
    print(f"\n主勝: {pred['prob_H']*100:.1f}%")
    print(f"和局: {pred['prob_D']*100:.1f}%")
    print(f"客勝: {pred['prob_A']*100:.1f}%")
    print(f"\n預測比分: {pred['predicted_score']}")
    print("Top 3 最可能比分:")
    for s in pred["top_scores"][:3]:
        print(f"  {s['h']}-{s['a']}  {s['p']*100:.1f}%")
    print(f"\nOver 2.5: {pred['over_2_5_probability']*100:.1f}%")
    print(f"Under 2.5: {pred['under_2_5_probability']*100:.1f}%")
    confidence = max(pred["prob_H"], pred["prob_D"], pred["prob_A"])
    print(f"\n模型信心（最高機率結果的機率值）: {confidence*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="足球比賽預測")
    parser.add_argument("--date", help="比賽日期 YYYY-MM-DD")
    parser.add_argument("--home", help="主隊")
    parser.add_argument("--away", help="客隊")
    parser.add_argument("--league", default="EPL")
    args = parser.parse_args()

    match_date = args.date or input("比賽日期 (YYYY-MM-DD): ").strip()
    home_team = args.home or input("主隊: ").strip()
    away_team = args.away or input("客隊: ").strip()

    if home_team == away_team:
        print("主隊跟客隊不能是同一隊")
        sys.exit(1)

    model, params, version = get_or_train_production_model(args.league)

    if home_team not in model["teams"]:
        print(f"[警告] 資料庫沒有 '{home_team}' 的歷史資料，請確認球隊名稱拼法"
              f"（跟 football-data.co.uk 的命名一致，例如 'Man United' 不是 'Man Utd'）")
    if away_team not in model["teams"]:
        print(f"[警告] 資料庫沒有 '{away_team}' 的歷史資料，請確認球隊名稱拼法")

    prediction_id, pred = predict_and_store(model, params, version, match_date, home_team, away_team, args.league)
    print_prediction(home_team, away_team, pred, version)
    print(f"\n已存入資料庫 (prediction_id={prediction_id})")


if __name__ == "__main__":
    main()
