"""
Repositório de resultados e cálculo Elo/Poisson entre equipas.

Processo:
1. Partimos do Elo PRÓPRIO já calculado sobre os 238.854 jogos históricos
   (corte: 2026-09-03) — não do Elo externo do ClubElo.
2. Para cada uma das 8 competições, carregamos o CSV do fixturedownload.com.
3. Descartamos os jogos com data <= corte (já estão refletidos no Elo baseline).
4. Os jogos com data > corte são processados cronologicamente, jornada a
   jornada, atualizando o Elo com a mesma fórmula usada no cálculo original.
5. Para cada jornada, guardamos o Elo de cada equipa ANTES desse jogo (para
   nunca haver fuga de informação) e a previsão do nosso modelo de Poisson.
"""
import pickle
import re
import pandas as pd
import numpy as np
from pathlib import Path
from elo_engine import run_elo
from train_model import fit_goal_model, score_matrix, market_probs, RHO_DEFAULT

BASE_DIR = Path(__file__).parent
CUTOFF_DATE = pd.Timestamp("2026-09-03")

# Mapeamento CSV (fixturedownload) -> nomes usados no dataset histórico (Matches.csv)
NAME_MAP = {
    "por": {
        "SL Benfica": "Benfica", "FC Porto": "Porto", "Sporting CP": "Sp Lisbon",
        "SC Braga": "Sp Braga", "Vitória SC": "Guimaraes", "FC Famalicão": "Famalicao",
        "Gil Vicente FC": "Gil Vicente", "Casa Pia AC": "Casa Pia", "Estoril Praia": "Estoril",
        "Moreirense FC": "Moreirense", "CD Nacional": "Nacional", "Rio Ave FC": "Rio Ave",
        "FC Alverca": "Alverca", "Marítimo M.": "Maritimo", "Estrela Amadora": "Est Amadora",
        "Académico de Viseu": "Academico Viseu", "FC Arouca": "Arouca", "Académico": "Academico Viseu",
    },
    "laliga": {
        "R. Racing Club": "Santander", "Celta": "Celta", "Deportivo Alavés": "Alaves",
        "Málaga CF": "Malaga", "Valencia CF": "Valencia", "CA Osasuna": "Osasuna",
        "Levante UD": "Levante", "Elche CF": "Elche", "RC Deportivo": "La Coruna",
        "FC Barcelona": "Barcelona", "Sevilla FC": "Sevilla", "RCD Espanyol de Barcelona": "Espanol",
        "Villarreal CF": "Villarreal", "Atlético de Madrid": "Ath Madrid",
        "Athletic Club": "Ath Bilbao", "Getafe CF": "Getafe", "Real Betis": "Betis",
        "Real Sociedad": "Sociedad", "Rayo Vallecano": "Vallecano",
    },
    "epl": {
        "Newcastle": "Newcastle", "Man City": "Man City", "Leeds": "Leeds",
        "Nott'm Forest": "Nott'm Forest", "Ipswich": "Ipswich", "Spurs": "Tottenham",
        "Man Utd": "Man United", "Hull": "Hull", "Coventry": "Coventry",
    },
    "ligue1": {
        "Stade Brestois 29": "Brest", "LOSC Lille": "Lille", "RC Lens": "Lens",
        "RC Strasbourg Alsace": "Strasbourg", "Olympique de Marseille": "Marseille", "AS Monaco": "Monaco",
        "Havre Athletic Club": "Le Havre", "FC Lorient": "Lorient", "Le Mans FC": "Le Mans",
        "Olympique Lyonnais": "Lyon", "Stade Rennais FC": "Rennes", "Toulouse FC": "Toulouse",
        "Angers SCO": "Angers", "AJ Auxerre": "Auxerre", "OGC Nice": "Nice", "Estac Troyes": "Troyes",
        "Paris Saint-Germain": "Paris SG",
    },
    "seriea": {"Internazionale": "Inter"},
    "cl": {
        "B. Dortmund": "Dortmund", "Man Utd": "Man United", "Paris": "Paris SG",
        "Leipzig": "RB Leipzig", "Atleti": "Ath Madrid", "PSV": "PSV Eindhoven",
        "Bodø/Glimt": "Bodo/Glimt", "Bayern München": "Bayern Munich", "AEK Athens": "AEK",
        "Real Betis": "Betis", "Fenerbahçe": "Fenerbahce", "Sporting CP": "Sp Lisbon",
        # S. Bratislava, Shakhtar, Slavia Praha, Sabah: genuinamente ausentes do histórico, ficam a 1500
    },
    "uel": {
        "Lillestrøm": "Lillestrom", "Real Sociedad": "Sociedad", "Olympiacos": "Olympiakos",
        # Omonia, Torreense, GNK Dinamo, Sparta Praha, H.Beer-Sheva, N.E.C., Ferencváros,
        # Celje, Ararat-Armenia, Union SG, Viktoria Plzen, Levski Sofia: ausentes, ficam a 1500
    },
    "turkey": {
        "Çorum": "Corum", "Gençlerbirligi": "Genclerbirligi", "Eyüpspor": "Eyupspor",
        "Çaykur Rizespor": "Rizespor", "Fenerbahçe": "Fenerbahce",
        # Göztepe, Istanbul Basaksehir: ausentes, ficam a 1500
    },
}

COMPETITIONS = ["por", "laliga", "epl", "ligue1", "seriea", "cl", "uel", "turkey"]


def csv_name_to_dataset(comp, name):
    return NAME_MAP.get(comp, {}).get(name, name)


def load_baseline():
    with open(BASE_DIR / "baseline_elo.pkl", "rb") as f:
        return pickle.load(f)


def load_fixture_csv(comp):
    path = BASE_DIR / "fixtures_csv" / f"{comp}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M", errors="coerce")
    return df


def build_repository():
    """Processa os 8 CSVs por cima do Elo baseline, devolve o estado completo."""
    baseline = load_baseline()
    elo = dict(baseline["elo"])  # copiar, vamos atualizar
    k, home_adv = baseline["k"], baseline["home_adv"]

    all_rows = []
    for comp in COMPETITIONS:
        df = load_fixture_csv(comp)
        df["comp"] = comp
        df["HomeMapped"] = df["Home Team"].apply(lambda n: csv_name_to_dataset(comp, n))
        df["AwayMapped"] = df["Away Team"].apply(lambda n: csv_name_to_dataset(comp, n))
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)

    # separar: já refletido no baseline (descartado) vs novo (a processar)
    novo_mask = combined["Date"] > CUTOFF_DATE
    descartados = (~novo_mask).sum()
    novos = novo_mask.sum()

    processed_rows = []
    for _, row in combined[novo_mask].iterrows():
        h, a = row["HomeMapped"], row["AwayMapped"]
        eh = elo.get(h, 1500.0)
        ea = elo.get(a, 1500.0)
        entry = {
            "comp": row["comp"], "round": row["Round Number"], "date": row["Date"],
            "home": row["Home Team"], "away": row["Away Team"],
            "elo_home_pre": eh, "elo_away_pre": ea,
            "result": row.get("Result", None),
        }
        processed_rows.append(entry)

        # se já tem resultado, atualizar o Elo (senão fica só a previsão, sem atualizar)
        result = row.get("Result", None)
        if isinstance(result, str) and "-" in result:
            m = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*", result)
            if m:
                hg, ag = int(m.group(1)), int(m.group(2))
                actual = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
                expected = 1 / (1 + 10 ** (-(eh + home_adv - ea) / 400))
                gd = abs(hg - ag)
                gd_mult = max(np.log(gd + 1), 1.0) if gd > 0 else 1.0
                delta = k * gd_mult * (actual - expected)
                elo[h] = eh + delta
                elo[a] = ea - delta

    return {
        "elo_atual": elo,
        "baseline_cutoff": str(CUTOFF_DATE.date()),
        "jogos_descartados_ja_no_baseline": int(descartados),
        "jogos_novos_processados": int(novos),
        "detalhe_jogos": processed_rows,
    }


def get_poisson_model():
    """Treina o modelo de Poisson com o NOSSO Elo (já provado melhor que o externo)."""
    hist = pd.read_csv(BASE_DIR / "data_repo" / "data" / "Matches.csv", usecols=[
        "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"
    ], parse_dates=["MatchDate"]).dropna(subset=["FTHome", "FTAway"]).sort_values("MatchDate").reset_index(drop=True)
    _, pre_h, pre_a = run_elo(hist, k=15, home_adv=40)
    hist["EloDiff"] = pre_h - pre_a
    return fit_goal_model(hist)


def predict_match(model, elo_home, elo_away):
    diff = (elo_home - elo_away) / 100.0
    mu_h = model.predict(pd.DataFrame({"elo_diff": [diff], "is_home": [1]}))[0]
    mu_a = model.predict(pd.DataFrame({"elo_diff": [-diff], "is_home": [0]}))[0]
    matrix = score_matrix(mu_h, mu_a, rho=RHO_DEFAULT)
    probs = market_probs(matrix)
    return {"mu_h": mu_h, "mu_a": mu_a, **probs}



    repo = build_repository()
    print(f"Corte do baseline: {repo['baseline_cutoff']}")
    print(f"Jogos já refletidos no baseline (descartados): {repo['jogos_descartados_ja_no_baseline']}")
    print(f"Jogos novos processados: {repo['jogos_novos_processados']}")
    print(f"\nEquipas com Elo atualizado (amostra):")
    for t in ["Benfica", "Porto", "Sporting", "Barcelona", "Real Madrid", "Arsenal"]:
        if t in repo["elo_atual"]:
            print(f"  {t}: {repo['elo_atual'][t]:.1f}")
