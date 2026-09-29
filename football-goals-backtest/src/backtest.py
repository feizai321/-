from __future__ import annotations

import math
import os
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yaml
from scipy.optimize import minimize

LEAGUES = {
    "E0": "EPL",
    "SP1": "LaLiga",
    "I1": "SerieA",
    "D1": "Bundesliga",
    "F1": "Ligue1",
    "P1": "PrimeiraLiga",
    "N1": "Eredivisie",
}

SEASON_LABELS = {
    "2122": "2021/22",
    "2223": "2022/23",
    "2324": "2023/24",
    "2425": "2024/25",
    "2526": "2025/26",
}
TEST_SEASONS = ["2324", "2425", "2526"]
WARMUP_SEASONS = ["2122", "2223"]

FEATURES = [
    "context_xgxga",
    "opportunity_creation",
    "opportunity_quality",
    "finishing_residual",
    "opportunity_suppression",
    "goalkeeper",
    "tempo_game_state",
    "set_piece",
    "lineup_strength",
    "home_away",
    "rest_travel",
    "recent_trend",
]


def safe_num(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def xg_proxy(shots, shots_on_target):
    """
    Documented proxy, NOT true xG.
    Calibrated only as a simple shot/SoT proxy for the first baseline.
    """
    shots = 0.0 if pd.isna(shots) else max(0.0, float(shots))
    sot = 0.0 if pd.isna(shots_on_target) else max(0.0, float(shots_on_target))
    return 0.035 * shots + 0.09 * sot


def poisson_over_25(lam):
    lam = max(1e-9, float(lam))
    return 1.0 - math.exp(-lam) * (1.0 + lam + lam * lam / 2.0)


def brier(p, y):
    return float((p - y) ** 2)


def logloss(p, y):
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return float(-(y * math.log(p) + (1 - y) * math.log(1 - p)))


def download_data():
    os.makedirs("data/raw", exist_ok=True)
    frames = []

    for code, league in LEAGUES.items():
        for season_code, season_label in SEASON_LABELS.items():
            url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"
            path = f"data/raw/{code}_{season_code}.csv"

            if not os.path.exists(path):
                resp = requests.get(
                    url,
                    timeout=60,
                    headers={"User-Agent": "football-goals-backtest/1.0"},
                )
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)

            try:
                df = pd.read_csv(path, encoding="latin1")
            except UnicodeDecodeError:
                df = pd.read_csv(path)

            df["League"] = league
            df["SeasonCode"] = season_code
            df["Season"] = season_label
            frames.append(df)

    return pd.concat(frames, ignore_index=True)


class TeamState:
    def __init__(self):
        self.gf = deque(maxlen=10)
        self.ga = deque(maxlen=10)
        self.shots = deque(maxlen=10)
        self.sot = deque(maxlen=10)
        self.corners = deque(maxlen=10)
        self.points = deque(maxlen=10)
        self.xg = deque(maxlen=10)
        self.xga = deque(maxlen=10)
        self.residual = deque(maxlen=10)
        self.last_date = None
        self.elo = 1500.0

    @staticmethod
    def mean(values, default):
        return float(np.mean(values)) if values else float(default)

    def snapshot(self):
        return {
            "gf": self.mean(self.gf, 1.25),
            "ga": self.mean(self.ga, 1.25),
            "shots": self.mean(self.shots, 12.0),
            "sot": self.mean(self.sot, 4.0),
            "corners": self.mean(self.corners, 5.0),
            "points": self.mean(self.points, 1.5),
            "xg": self.mean(self.xg, 1.25),
            "xga": self.mean(self.xga, 1.25),
            "residual": self.mean(self.residual, 0.0),
            "elo": self.elo,
        }

    def update(
        self,
        gf,
        ga,
        shots,
        sot,
        corners,
        points,
        xg,
        xga,
        date,
    ):
        self.gf.append(gf)
        self.ga.append(ga)
        self.shots.append(shots)
        self.sot.append(sot)
        self.corners.append(corners)
        self.points.append(points)
        self.xg.append(xg)
        self.xga.append(xga)
        self.residual.append(gf - xg)
        self.last_date = date


def rest_days(state: TeamState, date):
    if state.last_date is None:
        return 7.0
    return float(max(1, min(30, (date - state.last_date).days)))


def update_elo(home: TeamState, away: TeamState, hg, ag):
    expected_home = 1.0 / (1.0 + 10.0 ** ((away.elo - home.elo) / 400.0))
    actual_home = 1.0 if hg > ag else 0.0 if hg < ag else 0.5
    k = 18.0
    home.elo += k * (actual_home - expected_home)
    away.elo += k * ((1.0 - actual_home) - (1.0 - expected_home))


def clean(raw):
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    num_cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]
    for col in num_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        else:
            raw[col] = np.nan

    raw["DateParsed"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    raw = raw.dropna(
        subset=["DateParsed", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    ).copy()

    return raw.sort_values(["DateParsed", "League"]).reset_index(drop=True)


def build_pregame_rows(df):
    states = defaultdict(TeamState)
    rows = []

    # We keep the row generated BEFORE updating team states with that match.
    for _, r in df.iterrows():
        date = r["DateParsed"]
        home = states[(r["League"], r["HomeTeam"])]
        away = states[(r["League"], r["AwayTeam"])]

        hf = home.snapshot()
        af = away.snapshot()

        comps = {
            "context_xgxga": (hf["xg"] + af["xg"] + hf["xga"] + af["xga"]) / 2,
            "opportunity_creation": (hf["shots"] + af["shots"]) / 2,
            "opportunity_quality": (
                (hf["sot"] / max(hf["shots"], 1.0))
                + (af["sot"] / max(af["shots"], 1.0))
            ) / 2,
            "finishing_residual": (hf["residual"] + af["residual"]) / 2,
            "opportunity_suppression": (hf["xga"] + af["xga"]) / 2,
            "goalkeeper": (
                (hf["ga"] / max(hf["sot"], 1.0))
                + (af["ga"] / max(af["sot"], 1.0))
            ) / 2,
            "tempo_game_state": (
                hf["shots"] + hf["sot"] + af["shots"] + af["sot"]
            ) / 2,
            "set_piece": (hf["corners"] + af["corners"]) / 2,
            "lineup_strength": (hf["elo"] + af["elo"]) / 3000.0,
            "home_away": (hf["gf"] - af["gf"] + af["ga"] - hf["ga"]) / 2,
            "rest_travel": -(
                rest_days(home, date) + rest_days(away, date)
            ) / 2,
            "recent_trend": (hf["points"] + af["points"]) / 2,
        }

        row = {
            **{k: float(v) for k, v in comps.items()},
            "League": r["League"],
            "Season": r["Season"],
            "SeasonCode": r["SeasonCode"],
            "Date": date,
            "HomeTeam": r["HomeTeam"],
            "AwayTeam": r["AwayTeam"],
            "TotalGoals": float(r["FTHG"] + r["FTAG"]),
        }
        for c in ["P>2.5", "P<2.5", "BbAv>2.5", "BbAv<2.5", "P<2.5", "P>2.5"]:
            if c in r.index:
                row[c] = safe_num(r[c])

        rows.append(row)

        hp = 3 if r["FTHG"] > r["FTAG"] else 1 if r["FTHG"] == r["FTAG"] else 0
        ap = 3 if r["FTAG"] > r["FTHG"] else 1 if r["FTHG"] == r["FTAG"] else 0

        home_xg = xg_proxy(r["HS"], r["HST"])
        away_xg = xg_proxy(r["AS"], r["AST"])

        home.update(
            r["FTHG"], r["FTAG"], r["HS"], r["HST"], r["HC"], hp,
            home_xg, away_xg, date
        )
        away.update(
            r["FTAG"], r["FTHG"], r["AS"], r["AST"], r["AC"], ap,
            away_xg, home_xg, date
        )
        update_elo(home, away, r["FTHG"], r["FTAG"])

    return pd.DataFrame(rows)


def fit_model(train, weights):
    x = train[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    means = x.mean()
    scales = x.std(ddof=0).replace(0.0, 1.0)
    z = (x - means) / scales
    z = z.to_numpy()
    w = np.array([float(weights[f]) for f in FEATURES], dtype=float)
    signal = z @ w
    y = train["TotalGoals"].to_numpy(dtype=float)

    def nll(params):
        intercept, beta = params
        lam = np.exp(np.clip(intercept + beta * signal, -3.5, 3.5))
        return float(np.mean(lam - y * np.log(lam + 1e-12)))

    initial = [math.log(max(y.mean(), 1e-3)), 0.10]
    result = minimize(nll, initial, method="Nelder-Mead")

    return {
        "intercept": float(result.x[0]),
        "beta": float(result.x[1]),
        "means": means,
        "scales": scales,
    }


def predict(test, params, model_name):
    x = test[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    z = (x - params["means"]) / params["scales"]
    # Recover the weights from the standardized feature names in the calling scope.
    # The model signal was recalculated by the caller.
    return x, z, model_name


def score_model(train, test, weights, model_name):
    if len(train) < 20 or len(test) == 0:
        return pd.DataFrame()

    params = fit_model(train, weights)

    z_test = (
        test[FEATURES].astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        - params["means"]
    ) / params["scales"]

    w = np.array([float(weights[f]) for f in FEATURES], dtype=float)
    signal_test = z_test.to_numpy() @ w
    lam = np.exp(
        np.clip(
            params["intercept"] + params["beta"] * signal_test,
            -3.5,
            3.5,
        )
    )

    out = test[
        [
            "League",
            "Season",
            "SeasonCode",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "TotalGoals",
        ]
    ].copy()

    out["Model"] = model_name
    out["LambdaTotal"] = lam
    out["P_Over25"] = [poisson_over_25(v) for v in lam]
    out["P_Under25"] = 1.0 - out["P_Over25"]
    out["ActualOver25"] = (out["TotalGoals"] > 2.5).astype(int)
    out["BrierOver25"] = [
        brier(p, y)
        for p, y in zip(out["P_Over25"], out["ActualOver25"])
    ]
    out["LogLossOver25"] = [
        logloss(p, y)
        for p, y in zip(out["P_Over25"], out["ActualOver25"])
    ]

    return out


def summarize(pred):
    rows = []
    for model, g in pred.groupby("Model"):
        err = g["LambdaTotal"] - g["TotalGoals"]
        rows.append(
            {
                "Model": model,
                "Matches": int(len(g)),
                "MAE_lambda": float(np.mean(np.abs(err))),
                "RMSE_lambda": float(np.sqrt(np.mean(err ** 2))),
                "O25_Brier": float(g["BrierOver25"].mean()),
                "O25_LogLoss": float(g["LogLossOver25"].mean()),
                "O25_HitRate": float(g["ActualOver25"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("O25_LogLoss")


def main():
    os.makedirs("results", exist_ok=True)

    with open("config/models.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw = clean(download_data())
    rows = build_pregame_rows(raw)

    preds = []

    for season_code in TEST_SEASONS:
        train = rows[rows["Date"] < rows.loc[rows["SeasonCode"] == season_code, "Date"].min()].copy()
        test = rows[rows["SeasonCode"] == season_code].copy()

        # The expanding-window rule means test season is never used in fitting.
        # The global historical rows are already pre-game snapshots.
        if test.empty:
            continue

        for model_name, weights in cfg["models"].items():
            p = score_model(train, test, weights, model_name)
            if not p.empty:
                preds.append(p)

    if not preds:
        raise RuntimeError("No predictions generated.")

    pred = pd.concat(preds, ignore_index=True)
    summary = summarize(pred)

    by_league = (
        pred.groupby(["Model", "League"])
        .agg(
            Matches=("TotalGoals", "size"),
            O25_Brier=("BrierOver25", "mean"),
            O25_LogLoss=("LogLossOver25", "mean"),
            O25_HitRate=("ActualOver25", "mean"),
        )
        .reset_index()
    )

    by_season = (
        pred.groupby(["Model", "Season"])
        .agg(
            Matches=("TotalGoals", "size"),
            O25_Brier=("BrierOver25", "mean"),
            O25_LogLoss=("LogLossOver25", "mean"),
            O25_HitRate=("ActualOver25", "mean"),
        )
        .reset_index()
    )

    pred.to_csv("results/predictions.csv", index=False)
    summary.to_csv("results/summary.csv", index=False)
    by_league.to_csv("results/by_league.csv", index=False)
    by_season.to_csv("results/by_season.csv", index=False)

    with open("results/run_summary.md", "w", encoding="utf-8") as f:
        f.write("# Football Goals Backtest\n\n")
        f.write(
            f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n"
        )
        f.write("Test: 2023/24, 2024/25, 2025/26\n\n")
        f.write("Warm-up data: 2021/22, 2022/23\n\n")
        f.write(
            "xG field mode: shot/SoT proxy; not true match-level xG.\n\n"
        )
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
