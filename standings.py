"""
Classificações da época e simulação Monte Carlo do resto do calendário.

Importante (limitação assumida, documentada):
- A simulação usa o Elo ATUAL de cada equipa (fixo) para todos os jogos
  por disputar — não atualiza o Elo jogo a jogo dentro de cada simulação.
  É uma aproximação razoável para 50k simulações rápidas, mas significa
  que não capta o efeito de uma equipa "disparar" ou "descair" a meio da
  época dentro da própria simulação.
"""
import re
import numpy as np
import pandas as pd

from repository_engine import load_fixture_csv_raw, csv_name_to_dataset, predict_match_matrix


def _parse_result(result):
    """'2 - 1' -> (2, 1). Devolve (None, None) se ainda não há resultado."""
    if not isinstance(result, str) or "-" not in result:
        return None, None
    m = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*", result)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_season_fixtures(comp):
    """Todos os jogos da época (jornada 1 até à última), jogados e por jogar."""
    df = load_fixture_csv_raw(comp)
    goals = df["Result"].apply(_parse_result)
    df["HomeGoals"] = goals.apply(lambda t: t[0])
    df["AwayGoals"] = goals.apply(lambda t: t[1])
    return df


def compute_standings(df):
    """Classificação atual a partir dos jogos já disputados (goals não nulos)."""
    played = df.dropna(subset=["HomeGoals", "AwayGoals"]).copy()
    teams = pd.unique(pd.concat([df["Home Team"], df["Away Team"]]))
    table = {t: {"Pld": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0} for t in teams}

    for _, row in played.iterrows():
        h, a = row["Home Team"], row["Away Team"]
        hg, ag = int(row["HomeGoals"]), int(row["AwayGoals"])
        table[h]["Pld"] += 1
        table[a]["Pld"] += 1
        table[h]["GF"] += hg
        table[h]["GA"] += ag
        table[a]["GF"] += ag
        table[a]["GA"] += hg
        if hg > ag:
            table[h]["W"] += 1
            table[h]["Pts"] += 3
            table[a]["L"] += 1
        elif hg < ag:
            table[a]["W"] += 1
            table[a]["Pts"] += 3
            table[h]["L"] += 1
        else:
            table[h]["D"] += 1
            table[a]["D"] += 1
            table[h]["Pts"] += 1
            table[a]["Pts"] += 1

    out = pd.DataFrame.from_dict(table, orient="index")
    out["GD"] = out["GF"] - out["GA"]
    out = out.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index()
    out = out.rename(columns={"index": "Equipa"})
    out.index = out.index + 1
    return out[["Equipa", "Pld", "W", "D", "L", "GF", "GA", "GD", "Pts"]]


def simulate_season(comp, current_elo, model, n_sims=50_000, max_goals=8, seed=None):
    """
    Simula o resto da época n_sims vezes (Monte Carlo) a partir da
    classificação atual, usando o Elo atual de cada equipa (fixo) para
    gerar os marcadores por disputar.
    Devolve um DataFrame com, por equipa: posição atual, pontos atuais,
    prob. título (%), prob. Top 4 (%), prob. zona de despromoção (%,
    últimos 3 lugares) e posição média final.
    """
    rng = np.random.default_rng(seed)
    df = load_season_fixtures(comp)

    teams = sorted(pd.unique(pd.concat([df["Home Team"], df["Away Team"]])))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)

    standings_now = compute_standings(df).set_index("Equipa")
    pts0 = np.array([standings_now.loc[t, "Pts"] if t in standings_now.index else 0 for t in teams])
    gf0 = np.array([standings_now.loc[t, "GF"] if t in standings_now.index else 0 for t in teams])
    ga0 = np.array([standings_now.loc[t, "GA"] if t in standings_now.index else 0 for t in teams])

    pts = np.tile(pts0, (n_sims, 1)).astype(np.int32)
    gf = np.tile(gf0, (n_sims, 1)).astype(np.int32)
    ga = np.tile(ga0, (n_sims, 1)).astype(np.int32)

    remaining = df[df["HomeGoals"].isna()]
    grid = np.arange((max_goals + 1) * (max_goals + 1))
    hg_grid = grid // (max_goals + 1)
    ag_grid = grid % (max_goals + 1)

    for _, row in remaining.iterrows():
        h, a = row["Home Team"], row["Away Team"]
        if h not in idx or a not in idx:
            continue
        eh = current_elo.get(csv_name_to_dataset(comp, h), 1500.0)
        ea = current_elo.get(csv_name_to_dataset(comp, a), 1500.0)
        matrix = predict_match_matrix(model, eh, ea, max_goals=max_goals)
        flat = matrix.flatten()
        flat = flat / flat.sum()
        draws = rng.choice(grid, size=n_sims, p=flat)
        hg = hg_grid[draws]
        ag = ag_grid[draws]

        hi, ai = idx[h], idx[a]
        gf[:, hi] += hg
        ga[:, hi] += ag
        gf[:, ai] += ag
        ga[:, ai] += hg

        home_win = hg > ag
        away_win = ag > hg
        draw = ~home_win & ~away_win
        pts[:, hi] += np.where(home_win, 3, np.where(draw, 1, 0))
        pts[:, ai] += np.where(away_win, 3, np.where(draw, 1, 0))

    gd = gf - ga
    # codificação para ordenar tudo de uma vez (vetorizado): Pts > GD > GF
    score = pts.astype(np.int64) * 1_000_000 + (gd + 500) * 1000 + gf
    order = np.argsort(-score, axis=1)  # order[s, pos] = índice da equipa nessa posição, na simulação s
    rank = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    rank[rows, order] = np.arange(T)[None, :]  # rank[s, team_idx] = posição (0 = 1º)

    results = []
    for t in teams:
        i = idx[t]
        r = rank[:, i]
        results.append({
            "Equipa": t,
            "Posição atual": int(standings_now.index.get_loc(t)) + 1 if t in standings_now.index else None,
            "Pontos atuais": int(pts0[i]),
            "Título (%)": round(100 * (r == 0).mean(), 1),
            "Top 4 (%)": round(100 * (r <= 3).mean(), 1),
            "Zona descida (%)": round(100 * (r >= T - 3).mean(), 1),
            "Posição média final": round(1 + r.mean(), 1),
        })

    out = pd.DataFrame(results).sort_values("Posição média final").reset_index(drop=True)
    out.index = out.index + 1
    return out
