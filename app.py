import streamlit as st
import pandas as pd
from pathlib import Path

from repository_engine import build_repository, get_poisson_model, predict_match, COMPETITIONS, csv_name_to_dataset
from standings import load_season_fixtures, compute_standings, simulate_season

st.set_page_config(page_title="JustElo — Repositório", layout="wide")
st.title("⚽ JustElo — Repositório de Elo próprio + Poisson")
st.caption(
    "Elo calculado por nós (não o do ClubElo), sobre 238.854 jogos históricos + os 8 CSVs "
    "do fixturedownload.com. Corte do histórico: 2026-09-03 — jogos depois disso são "
    "processados jornada a jornada por cima do Elo baseline."
)

COMP_LABELS = {
    "por": "Liga Portugal", "laliga": "La Liga", "epl": "Premier League",
    "ligue1": "Ligue 1", "seriea": "Serie A", "cl": "Champions League",
    "uel": "Liga Europa", "turkey": "Liga Turca",
}

FIXTURES_DIR = Path(__file__).parent / "fixtures_csv"


@st.cache_resource(show_spinner="A processar 238.854 jogos históricos + 8 CSVs...")
def load_everything():
    repo = build_repository()
    model = get_poisson_model()
    return repo, model


def reload_everything():
    """Limpa a cache e força o recálculo completo (Elo + modelo + repositório)."""
    load_everything.clear()
    st.cache_data.clear()
    st.rerun()


repo, model = load_everything()

col1, col2, col3 = st.columns(3)
col1.metric("Jogos já no baseline (descartados)", repo["jogos_descartados_ja_no_baseline"])
col2.metric("Jogos novos processados", repo["jogos_novos_processados"])
col3.metric("Corte do histórico", repo["baseline_cutoff"])

st.divider()

comp = st.selectbox("Competição", COMPETITIONS, format_func=lambda c: COMP_LABELS.get(c, c))

tab_jornada, tab_classificacao, tab_simulacao, tab_valor, tab_upload = st.tabs(
    ["📅 Previsões por jornada", "🏆 Classificação atual", "🎲 Simulação Monte Carlo", "💰 Value Betting", "⬆️ Atualizar CSVs"]
)

# ---------------------------------------------------------------- JORNADA
with tab_jornada:
    season_df = load_season_fixtures(comp)
    if season_df.empty:
        st.warning("Sem jogos para esta competição.")
    else:
        rounds = sorted(season_df["Round Number"].unique())
        round_sel = st.selectbox("Jornada", rounds, index=0)
        df_round = season_df[season_df["Round Number"] == round_sel].sort_values("Date")

        st.subheader(f"{COMP_LABELS.get(comp, comp)} — Jornada {round_sel}")

        for _, row in df_round.iterrows():
            eh = repo["elo_atual"].get(csv_name_to_dataset(comp, row["Home Team"]), 1500.0)
            ea = repo["elo_atual"].get(csv_name_to_dataset(comp, row["Away Team"]), 1500.0)
            pred = predict_match(model, eh, ea)
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            with c1:
                st.markdown(f"**{row['Home Team']}** vs **{row['Away Team']}**")
                st.caption(f"Elo: {eh:.0f} vs {ea:.0f} · xG: {pred['mu_h']:.2f} - {pred['mu_a']:.2f}")
            c2.metric("Casa", f"{pred['home_win']*100:.1f}%")
            c3.metric("Empate", f"{pred['draw']*100:.1f}%")
            c4.metric("Fora", f"{pred['away_win']*100:.1f}%")
            if pd.notna(row["HomeGoals"]):
                st.success(f"✅ Resultado final: {int(row['HomeGoals'])} - {int(row['AwayGoals'])}")
            st.divider()

# ---------------------------------------------------------------- CLASSIFICAÇÃO
with tab_classificacao:
    st.subheader(f"Classificação atual — {COMP_LABELS.get(comp, comp)}")
    season_df = load_season_fixtures(comp)
    tabela = compute_standings(season_df)
    st.dataframe(tabela, use_container_width=True, hide_index=False)
    jogados = season_df["HomeGoals"].notna().sum()
    por_jogar = season_df["HomeGoals"].isna().sum()
    st.caption(f"{jogados} jogos disputados · {por_jogar} por disputar nesta competição.")

# ---------------------------------------------------------------- SIMULAÇÃO
with tab_simulacao:
    st.subheader(f"Simulação Monte Carlo — {COMP_LABELS.get(comp, comp)}")
    st.caption(
        "Simula o resto do calendário 50.000 vezes usando o Elo atual de cada equipa "
        "(fixo durante a simulação) e o modelo de Poisson para gerar marcadores. "
        "Não atualiza o Elo jogo a jogo dentro da própria simulação — é uma aproximação."
    )
    n_sims = st.number_input("Número de simulações", min_value=1_000, max_value=200_000, value=50_000, step=1_000)
    if st.button("▶️ Correr simulação", key=f"sim_{comp}"):
        with st.spinner(f"A correr {n_sims:,} simulações..."):
            sim = simulate_season(comp, repo["elo_atual"], model, n_sims=int(n_sims))
        st.dataframe(sim, use_container_width=True, hide_index=False)

# ---------------------------------------------------------------- VALUE BETTING
with tab_valor:
    st.subheader(f"Value Betting — {COMP_LABELS.get(comp, comp)}")
    st.caption(
        "Insere as odds (decimais) que vês na tua casa de apostas para cada resultado. "
        "Comparamos com a probabilidade do nosso modelo, já sem a margem da casa (overround), "
        "e mostramos o valor esperado (EV) de apostar segundo o TEU modelo. "
        "EV positivo não é garantia de ganhar essa aposta — é uma vantagem estatística ao longo de muitas apostas."
    )
    season_df_v = load_season_fixtures(comp)
    rounds_v = sorted(season_df_v["Round Number"].unique())
    round_v = st.selectbox("Jornada", rounds_v, index=0, key="jornada_valor")
    df_round_v = season_df_v[season_df_v["Round Number"] == round_v].sort_values("Date")
    df_round_v = df_round_v[df_round_v["HomeGoals"].isna()]  # só faz sentido apostar em jogos por disputar

    if df_round_v.empty:
        st.info("Todos os jogos desta jornada já têm resultado — nada para apostar aqui.")
    else:
        for i, row in df_round_v.iterrows():
            eh = repo["elo_atual"].get(csv_name_to_dataset(comp, row["Home Team"]), 1500.0)
            ea = repo["elo_atual"].get(csv_name_to_dataset(comp, row["Away Team"]), 1500.0)
            pred = predict_match(model, eh, ea)

            st.markdown(f"**{row['Home Team']}** vs **{row['Away Team']}**")
            c1, c2, c3 = st.columns(3)
            odds_in = {}
            model_p = {"Casa": pred["home_win"], "Empate": pred["draw"], "Fora": pred["away_win"]}
            for col, label in zip((c1, c2, c3), ("Casa", "Empate", "Fora")):
                with col:
                    st.caption(f"{label} · modelo: {model_p[label]*100:.1f}%")
                    odds_in[label] = st.number_input(
                        f"Odd {label}", min_value=1.0, value=1.0, step=0.01,
                        key=f"odd_{comp}_{round_v}_{i}_{label}", label_visibility="collapsed",
                    )

            entered = {k: v for k, v in odds_in.items() if v > 1.0}
            if len(entered) == 3:
                raw_implied = {k: 1.0 / v for k, v in odds_in.items()}
                overround = sum(raw_implied.values())
                fair_implied = {k: raw_implied[k] / overround for k in raw_implied}
                c1, c2, c3 = st.columns(3)
                for col, label in zip((c1, c2, c3), ("Casa", "Empate", "Fora")):
                    with col:
                        ev = model_p[label] * odds_in[label] - 1
                        edge = model_p[label] - fair_implied[label]
                        texto = f"Odd justa: {1/model_p[label]:.2f} · Edge: {edge*100:+.1f}pp · EV: {ev*100:+.1f}%"
                        if ev > 0.02:
                            st.success(texto)
                        elif ev < -0.02:
                            st.error(texto)
                        else:
                            st.caption(texto)
                st.caption(f"Margem da casa (overround) nesta jornada: {(overround-1)*100:.1f}%")
            else:
                st.caption("Preenche as 3 odds para ver o edge e o valor esperado (EV).")
            st.divider()

# ---------------------------------------------------------------- UPLOAD DE CSVs
with tab_upload:
    st.subheader("Atualizar os CSVs de fixtures")
    st.warning(
        "⚠️ Isto substitui o ficheiro **nesta sessão da app** e recalcula tudo de imediato. "
        "No Streamlit Cloud, o ficheiro NÃO fica guardado de forma permanente — se a app "
        "reiniciar (redeploy, inatividade, etc.), volta ao CSV que está no GitHub. "
        "Para tornar a atualização definitiva, sobe o mesmo ficheiro também à pasta "
        "`fixtures_csv/` do repositório no GitHub."
    )
    comp_upload = st.selectbox(
        "Competição a atualizar", COMPETITIONS, format_func=lambda c: COMP_LABELS.get(c, c), key="comp_upload"
    )
    uploaded = st.file_uploader(f"CSV do fixturedownload.com para {COMP_LABELS.get(comp_upload, comp_upload)}", type="csv")
    if uploaded is not None:
        destino = FIXTURES_DIR / f"{comp_upload}.csv"
        if st.button("Guardar e recalcular tudo"):
            destino.write_bytes(uploaded.getvalue())
            st.success(f"{destino.name} atualizado. A recalcular...")
            reload_everything()
