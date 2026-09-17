import streamlit as st
import pandas as pd
from pathlib import Path

from repository_engine import build_repository, get_poisson_model, predict_match, COMPETITIONS
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

tab_jornada, tab_classificacao, tab_simulacao, tab_upload = st.tabs(
    ["📅 Previsões por jornada", "🏆 Classificação atual", "🎲 Simulação Monte Carlo", "⬆️ Atualizar CSVs"]
)

# ---------------------------------------------------------------- JORNADA
with tab_jornada:
    df_jogos = pd.DataFrame(repo["detalhe_jogos"])
    df_comp = df_jogos[df_jogos["comp"] == comp].copy()

    if df_comp.empty:
        st.warning("Sem jogos processados para esta competição.")
    else:
        rounds = sorted(df_comp["round"].unique())
        round_sel = st.selectbox("Jornada", rounds, index=0)
        df_round = df_comp[df_comp["round"] == round_sel].sort_values("date")

        st.subheader(f"{COMP_LABELS.get(comp, comp)} — Jornada {round_sel}")

        for _, row in df_round.iterrows():
            pred = predict_match(model, row["elo_home_pre"], row["elo_away_pre"])
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            with c1:
                st.markdown(f"**{row['home']}** vs **{row['away']}**")
                st.caption(f"Elo: {row['elo_home_pre']:.0f} vs {row['elo_away_pre']:.0f} · xG: {pred['mu_h']:.2f} - {pred['mu_a']:.2f}")
            c2.metric("Casa", f"{pred['home_win']*100:.1f}%")
            c3.metric("Empate", f"{pred['draw']*100:.1f}%")
            c4.metric("Fora", f"{pred['away_win']*100:.1f}%")
            if isinstance(row["result"], str) and "-" in str(row["result"]):
                st.success(f"✅ Resultado final: {row['result']}")
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
