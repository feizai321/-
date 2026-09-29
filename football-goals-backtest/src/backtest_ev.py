from __future__ import annotations

import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass

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
SEASONS = {
    "2122": "2021/22",
    "2223": "2022/23",
    "2324": "2023/24",
    "2425": "2024/25",
    "2526": "2025/26",
}
TEST_SEASONS = ["2324", "2425", "2526"]
FEATURES = [
    "context_xgxga", "opportunity_creation", "opportunity_quality",
    "finishing_residual", "opportunity_suppression", "goalkeeper",
    "tempo_game_state", "set_piece", "lineup_strength", "home_away",
    "rest_travel", "recent_trend",
]


def xg_proxy(shots, sot):
    shots = 0.0 if pd.isna(shots) else max(0.0, float(shots))
    sot = 0.0 if pd.isna(sot) else max(0.0, float(sot))
    return 0.035 * shots + 0.09 * sot


def o25_prob(lam):
    lam = max(1e-9, float(lam))
    return 1.0 - math.exp(-lam) * (1 + lam + lam * lam / 2)


def logloss(p, y):
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def clean(df):
    for c in ["FTHG","FTAG","HS","AS","HST","AST","HC","AC",
              "Avg>2.5","Avg<2.5","PC>2.5","PC<2.5",
              "P>2.5","P<2.5"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    df["DateParsed"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DateParsed","HomeTeam","AwayTeam","FTHG","FTAG"]).copy()
    return df.sort_values(["DateParsed","League"]).reset_index(drop=True)


class State:
    def __init__(self):
        self.gf=deque(maxlen=10); self.ga=deque(maxlen=10)
        self.sh=deque(maxlen=10); self.sot=deque(maxlen=10)
        self.c=deque(maxlen=10); self.p=deque(maxlen=10)
        self.xg=deque(maxlen=10); self.xga=deque(maxlen=10)
        self.res=deque(maxlen=10); self.last=None; self.elo=1500.0

    def mean(self, q, d):
        return float(np.mean(q)) if q else float(d)

    def snap(self):
        return {
            "gf": self.mean(self.gf,1.25), "ga": self.mean(self.ga,1.25),
            "sh": self.mean(self.sh,12), "sot": self.mean(self.sot,4),
            "c": self.mean(self.c,5), "p": self.mean(self.p,1.5),
            "xg": self.mean(self.xg,1.25), "xga": self.mean(self.xga,1.25),
            "res": self.mean(self.res,0), "elo": self.elo
        }

    def update(self,gf,ga,sh,sot,c,p,xg,xga,date):
        self.gf.append(gf); self.ga.append(ga); self.sh.append(sh); self.sot.append(sot)
        self.c.append(c); self.p.append(p); self.xg.append(xg); self.xga.append(xga)
        self.res.append(gf-xg); self.last=date


def rest_days(s,date):
    return 7.0 if s.last is None else float(max(1,min(30,(date-s.last).days)))


def elo_update(h,a,hg,ag):
    eh = 1/(1+10**((a.elo-h.elo)/400))
    sh = 1 if hg>ag else 0 if hg<ag else .5
    k=18
    h.elo += k*(sh-eh)
    a.elo += k*((1-sh)-(1-eh))


def download_all():
    os.makedirs("data/raw",exist_ok=True)
    frames=[]
    for code,league in LEAGUES.items():
        for season,label in SEASONS.items():
            path=f"data/raw/{code}_{season}.csv"
            url=f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            if not os.path.exists(path):
                r=requests.get(url,timeout=60,headers={"User-Agent":"football-goals-ev-backtest/1.0"})
                r.raise_for_status()
                open(path,"wb").write(r.content)
            try:
                d=pd.read_csv(path,encoding="cp1252")
            except Exception:
                d=pd.read_csv(path,encoding="latin1")
            d["League"]=league; d["SeasonCode"]=season; d["Season"]=label
            frames.append(d)
    return pd.concat(frames,ignore_index=True)


def build_rows(df):
    states=defaultdict(State)
    rows=[]
    for _,r in df.iterrows():
        h=states[(r.League,r.HomeTeam)]; a=states[(r.League,r.AwayTeam)]
        hf=h.snap(); af=a.snap(); dt=r.DateParsed
        comps={
            "context_xgxga":(hf["xg"]+af["xg"]+hf["xga"]+af["xga"])/2,
            "opportunity_creation":(hf["sh"]+af["sh"])/2,
            "opportunity_quality":((hf["sot"]/max(hf["sh"],1))+(af["sot"]/max(af["sh"],1)))/2,
            "finishing_residual":(hf["res"]+af["res"])/2,
            "opportunity_suppression":(hf["xga"]+af["xga"])/2,
            "goalkeeper":((hf["ga"]/max(hf["sot"],1))+(af["ga"]/max(af["sot"],1)))/2,
            "tempo_game_state":(hf["sh"]+hf["sot"]+af["sh"]+af["sot"])/2,
            "set_piece":(hf["c"]+af["c"])/2,
            "lineup_strength":(hf["elo"]+af["elo"])/3000,
            "home_away":(hf["gf"]-af["gf"]+af["ga"]-hf["ga"])/2,
            "rest_travel":-(rest_days(h,dt)+rest_days(a,dt))/2,
            "recent_trend":(hf["p"]+af["p"])/2,
        }
        out={**comps,"League":r.League,"Season":r.Season,"SeasonCode":r.SeasonCode,
             "Date":dt,"HomeTeam":r.HomeTeam,"AwayTeam":r.AwayTeam,
             "TotalGoals":float(r.FTHG+r.FTAG)}
        for c in ["Avg>2.5","Avg<2.5","PC>2.5","PC<2.5","P>2.5","P<2.5"]:
            out[c]=r[c]
        rows.append(out)

        hp=3 if r.FTHG>r.FTAG else 1 if r.FTHG==r.FTAG else 0
        ap=3 if r.FTAG>r.FTHG else 1 if r.FTHG==r.FTAG else 0
        hx=xg_proxy(r.HS,r.HST); ax=xg_proxy(r.AS,r.AST)
        h.update(r.FTHG,r.FTAG,r.HS,r.HST,r.HC,hp,hx,ax,dt)
        a.update(r.FTAG,r.FTHG,r.AS,r.AST,r.AC,ap,ax,hx,dt)
        elo_update(h,a,r.FTHG,r.FTAG)
    return pd.DataFrame(rows)


def fit(train,weights):
    x=train[FEATURES].astype(float).replace([np.inf,-np.inf],np.nan).fillna(0)
    mu=x.mean(); sd=x.std(ddof=0).replace(0,1)
    z=((x-mu)/sd).to_numpy()
    w=np.array([float(weights[f]) for f in FEATURES])
    s=z@w; y=train.TotalGoals.to_numpy(dtype=float)

    def nll(par):
        lam=np.exp(np.clip(par[0]+par[1]*s,-3.5,3.5))
        return float(np.mean(lam-y*np.log(lam+1e-12)))
    res=minimize(nll,[math.log(max(y.mean(),1e-3)),.1],method="Nelder-Mead")
    return float(res.x[0]),float(res.x[1]),mu,sd


def predict(test,params,weights):
    intercept,beta,mu,sd=params
    x=test[FEATURES].astype(float).replace([np.inf,-np.inf],np.nan).fillna(0)
    z=((x-mu)/sd).to_numpy()
    w=np.array([float(weights[f]) for f in FEATURES])
    signal=z@w
    lam=np.exp(np.clip(intercept+beta*signal,-3.5,3.5))
    return lam


def add_market_metrics(p):
    rows=[]
    for _,r in p.iterrows():
        ao=r["Avg>2.5"]; au=r["Avg<2.5"]
        co=r["PC>2.5"]; cu=r["PC<2.5"]
        if pd.isna(ao) or pd.isna(au) or ao<=1 or au<=1:
            continue
        io,iu=1/ao,1/au
        sov=io+iu
        mo=io/sov; mu=iu/sov
        model_o=float(r.P_Over25)
        candidates=[
            ("OVER",model_o,ao,mo,co),
            ("UNDER",1-model_o,au,mu,cu),
        ]
        # At most one bet per match: select the side with highest EV.
        best=max(candidates,key=lambda z:z[1]*z[2]-1)
        side,prob,odds,mp,close_odds=best
        ev=prob*odds-1
        selected=(side=="OVER")
        actual_over=(r.TotalGoals>2.5)
        win=int(actual_over==selected)
        profit=odds-1 if win else -1
        entry_imp=1/odds
        close_imp=np.nan if pd.isna(close_odds) or close_odds<=1 else 1/close_odds
        rows.append({**r.to_dict(),"BetSide":side,"BetProb":prob,"EntryOdds":odds,
                     "MarketDeVigProb":mp,"EV":ev,"Win":win,"Profit":profit,
                     "EntryImpliedProb":entry_imp,"CloseOdds":close_odds,
                     "CloseImpliedProb":close_imp})
    return pd.DataFrame(rows)


def max_drawdown(profits):
    if len(profits)==0: return 0.0
    eq=np.cumsum(profits)
    peak=np.maximum.accumulate(np.insert(eq,0,0))[1:]
    dd=eq-peak
    return float(dd.min())


def summarize_bets(bets,threshold):
    b=bets[bets.EV>=threshold].copy()
    if b.empty:
        return {"Threshold":threshold,"Bets":0,"Wins":0,"WinRate":np.nan,
                "Profit":0,"ROI":np.nan,"AvgEV":np.nan,"MaxDrawdown":0,
                "AvgCLVProb":np.nan}
    b=b.sort_values("Date")
    k=(b.BetProb*b.EntryOdds-1)/(b.EntryOdds-1)
    clv=(b.CloseImpliedProb-b.EntryImpliedProb).dropna()
    return {
        "Threshold":threshold,
        "Bets":int(len(b)),
        "Wins":int(b.Win.sum()),
        "WinRate":float(b.Win.mean()),
        "Profit":float(b.Profit.sum()),
        "ROI":float(b.Profit.sum()/len(b)),
        "AvgEV":float(b.EV.mean()),
        "MedianEV":float(b.EV.median()),
        "MaxDrawdown":max_drawdown(b.Profit.to_numpy()),
        "MeanKelly":float(k.mean()),
        "AvgCLVProb":float(clv.mean()) if len(clv) else np.nan,
        "CLVPositiveRate":float((clv>0).mean()) if len(clv) else np.nan,
    }


def main():
    os.makedirs("results_ev",exist_ok=True)
    cfg=yaml.safe_load(open("config/models.yml","r",encoding="utf-8"))
    rows=build_rows(clean(download_all()))
    all_bets=[]

    for season in TEST_SEASONS:
        test=rows[rows.SeasonCode==season].copy()
        if test.empty:
            continue
        first_date=test.Date.min()
        train=rows[rows.Date<first_date].copy()
        for model,weights in cfg["models"].items():
            params=fit(train,weights)
            lam=predict(test,params,weights)
            t=test.copy()
            t["Model"]=model
            t["LambdaTotal"]=lam
            t["P_Over25"]=[o25_prob(x) for x in lam]
            t["P_Under25"]=1-t["P_Over25"]
            t["ActualOver25"]=(t.TotalGoals>2.5).astype(int)
            scored=add_market_metrics(t)
            all_bets.append(scored)

    bets=pd.concat(all_bets,ignore_index=True)
    bet_summaries=[]
    for model,g in bets.groupby("Model"):
        for th in [0.05,0.08]:
            s=summarize_bets(g,th); s["Model"]=model
            bet_summaries.append(s)
    summary=pd.DataFrame(bet_summaries)[
        ["Model","Threshold","Bets","Wins","WinRate","Profit","ROI","AvgEV","MedianEV",
         "MaxDrawdown","MeanKelly","AvgCLVProb","CLVPositiveRate"]
    ]
    summary.to_csv("results_ev/summary_ev_5_8.csv",index=False)

    league_rows=[]
    for (model,league),g in bets.groupby(["Model","League"]):
        for th in [0.05,0.08]:
            s=summarize_bets(g,th); s.update({"Model":model,"League":league})
            league_rows.append(s)
    pd.DataFrame(league_rows).to_csv("results_ev/by_league_ev.csv",index=False)

    season_rows=[]
    for (model,season),g in bets.groupby(["Model","Season"]):
        for th in [0.05,0.08]:
            s=summarize_bets(g,th); s.update({"Model":model,"Season":season})
            season_rows.append(s)
    pd.DataFrame(season_rows).to_csv("results_ev/by_season_ev.csv",index=False)

    bets.to_csv("results_ev/all_candidate_bets.csv",index=False)
    print(summary.to_string(index=False))


if __name__=="__main__":
    main()
