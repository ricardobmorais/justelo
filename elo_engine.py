"""
Motor de Elo próprio — implementação standard, calibrada nos nossos dados.
Processa os jogos cronologicamente, atualizando o rating de cada equipa jogo a jogo.
"""
import pandas as pd
import numpy as np

def run_elo(df, k=20, home_adv=60, initial=1500):
    """
    df: colunas MatchDate, HomeTeam, AwayTeam, FTHome, FTAway (ordenado por data)
    k: fator de aprendizagem (quanto cada jogo pesa na atualização)
    home_adv: vantagem em pontos Elo por jogar em casa
    Devolve: dict {equipa: elo final}, e um array com o Elo de cada equipa ANTES de cada jogo
              (para usar como feature sem fuga de informação)
    """
    elo = {}
    def get(t):
        if t not in elo:
            elo[t] = initial
        return elo[t]

    pre_home_elo = np.zeros(len(df))
    pre_away_elo = np.zeros(len(df))

    for i, row in enumerate(df.itertuples()):
        eh, ea = get(row.HomeTeam), get(row.AwayTeam)
        pre_home_elo[i] = eh
        pre_away_elo[i] = ea

        # resultado real: 1 = casa venceu, 0.5 = empate, 0 = fora venceu
        if row.FTHome > row.FTAway:
            actual = 1.0
        elif row.FTHome < row.FTAway:
            actual = 0.0
        else:
            actual = 0.5

        # probabilidade esperada de vitória da casa, dado o Elo (com vantagem de casa)
        expected = 1 / (1 + 10 ** (-(eh + home_adv - ea) / 400))

        # ajuste extra pela margem de vitória (goal difference multiplier, comum em Elo de futebol)
        gd = abs(row.FTHome - row.FTAway)
        gd_mult = np.log(gd + 1) if gd > 0 else 1.0
        gd_mult = max(gd_mult, 1.0)

        delta = k * gd_mult * (actual - expected)
        elo[row.HomeTeam] = eh + delta
        elo[row.AwayTeam] = ea - delta

    return elo, pre_home_elo, pre_away_elo


if __name__ == '__main__':
    df = pd.read_csv('../data_repo/data/Matches.csv', usecols=[
        'MatchDate', 'HomeTeam', 'AwayTeam', 'FTHome', 'FTAway'
    ], parse_dates=['MatchDate']).dropna(subset=['FTHome', 'FTAway'])
    df = df.sort_values('MatchDate').reset_index(drop=True)

    elo_final, pre_h, pre_a = run_elo(df, k=20, home_adv=60)
    df['OurHomeElo'] = pre_h
    df['OurAwayElo'] = pre_a
    df.to_parquet('elo_computed.parquet')
    print(f"Processados {len(df)} jogos, {len(elo_final)} equipas")
    print("\nTop 10 Elo final:")
    for t, e in sorted(elo_final.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t}: {e:.0f}")
