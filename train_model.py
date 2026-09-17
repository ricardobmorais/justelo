"""
Modelo de previsão de futebol baseado em Elo + regressão de Poisson.

Ideia central:
  - Elo (ClubElo) já resume a força relativa de qualquer clube europeu,
    incluindo desempenho em competições UEFA — por isso funciona mesmo
    para prever jogos da Champions League usando dados de treino de
    ligas domésticas (onde temos milhares de jogos com Elo + odds + golos).
  - Ajustamos golos esperados (casa e fora) como função da diferença de Elo
    e de um efeito fixo de "vantagem de jogar em casa".
  - O resultado é uma distribuição de Poisson (bivariada, com correlação
    ligeira tipo Dixon-Coles) para o marcador, da qual derivamos 1X2,
    over/under, BTTS, handicaps, etc.

Validação:
  - Backtest out-of-sample por época (treina até época N, testa na N+1).
  - Métricas: log-loss (calibração de probabilidade) e Brier score.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "matches_clean.csv"

RHO_DEFAULT = -0.05  # ajuste Dixon-Coles p/ baixo-scoring (0-0,1-0,0-1,1-1)


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["MatchDate"])
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    return df


def fit_goal_model(df):
    """
    Regressão de Poisson: golos_casa ~ EloDiff
                           golos_fora ~ EloDiff (sinal invertido)
    Formato "long": uma linha por (equipa, jogo, é_casa?), com EloDiff
    sempre da perspetiva da equipa que marcou os golos naquela linha.
    """
    home_rows = pd.DataFrame({
        "goals": df["FTHome"],
        "elo_diff": df["EloDiff"] / 100.0,   # escala: 100 pontos Elo
        "is_home": 1,
    })
    away_rows = pd.DataFrame({
        "goals": df["FTAway"],
        "elo_diff": -df["EloDiff"] / 100.0,
        "is_home": 0,
    })
    long_df = pd.concat([home_rows, away_rows], ignore_index=True)

    model = smf.glm(
        formula="goals ~ elo_diff + is_home",
        data=long_df,
        family=sm.families.Poisson(),
    ).fit()
    return model


def predict_expected_goals(model, home_elo, away_elo):
    elo_diff = (home_elo - away_elo) / 100.0
    mu_home = model.predict(pd.DataFrame({"elo_diff": [elo_diff], "is_home": [1]}))[0]
    mu_away = model.predict(pd.DataFrame({"elo_diff": [-elo_diff], "is_home": [0]}))[0]
    return mu_home, mu_away


def dixon_coles_adj(home_goals, away_goals, mu_home, mu_away, rho):
    """Correção Dixon-Coles para dependência entre golos baixos (0/1)."""
    if home_goals == 0 and away_goals == 0:
        return 1 - (mu_home * mu_away * rho)
    elif home_goals == 0 and away_goals == 1:
        return 1 + (mu_home * rho)
    elif home_goals == 1 and away_goals == 0:
        return 1 + (mu_away * rho)
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def score_matrix(mu_home, mu_away, max_goals=8, rho=RHO_DEFAULT):
    m = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, mu_home) * poisson.pmf(j, mu_away)
            p *= dixon_coles_adj(i, j, mu_home, mu_away, rho)
            m[i, j] = p
    m = m / m.sum()  # renormalizar após ajuste
    return m


def market_probs(matrix):
    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()

    max_g = matrix.shape[0] - 1
    over25 = sum(
        matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1) if i + j > 2.5
    )
    under25 = 1 - over25

    btts_yes = sum(
        matrix[i, j] for i in range(1, max_g + 1) for j in range(1, max_g + 1)
    )
    btts_no = 1 - btts_yes

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_2_5": over25,
        "under_2_5": under25,
        "btts_yes": btts_yes,
        "btts_no": btts_no,
    }


def handicap_prob(matrix, handicap, side="home"):
    """
    Probabilidade de cobrir um handicap asiático (ex: home -1 => home tem de
    ganhar por 2+; valores .25/.75 tratados como média de duas linhas half-win).
    """
    max_g = matrix.shape[0] - 1

    def cover(h, a, line):
        diff = h - a
        return diff + line > 0  # >0 gain, ==0 push, <0 loss

    def result_for_line(line):
        win = 0.0
        push = 0.0
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                diff = (i - j) if side == "home" else (j - i)
                p = matrix[i, j]
                d = diff + line
                if d > 0:
                    win += p
                elif d == 0:
                    push += p
        return win, push

    frac, whole = np.modf(abs(handicap))
    sign = -1 if handicap < 0 else 1
    if abs(frac) < 1e-9:
        win, push = result_for_line(handicap)
        return win / (1 - push) if push < 1 else win
    else:
        # quarter/half line: média de duas linhas adjacentes (ex: -0.75 = -0.5 e -1.0)
        line_a = sign * whole
        line_b = sign * (whole + 1)
        win_a, push_a = result_for_line(line_a)
        win_b, push_b = result_for_line(line_b)
        eff_a = win_a / (1 - push_a) if push_a < 1 else win_a
        return (eff_a + win_b) / 2


if __name__ == "__main__":
    df = load_data()
    print(f"Jogos usados no treino: {len(df)}")
    model = fit_goal_model(df)
    print(model.summary())
    model.save("/home/claude/football_model/goal_model.pkl")
