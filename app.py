import streamlit as st
import pandas as pd
from repository_engine import build_repository, get_poisson_model, predict_match, COMPETITIONS

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


@st.cache_resource(show_spinner="A processar 238.854 jogos históricos + 8 CSVs...")
def load_everything():
    repo = build_repository()
    model = get_poisson_model()
    return repo, model


repo, model = load_everything()

col1, col2, col3 = st.columns(3)
col1.metric("Jogos já no baseline (descartados)", repo["jogos_descartados_ja_no_baseline"])
col2.metric("Jogos novos processados", repo["jogos_novos_processados"])
col3.metric("Corte do histórico", repo["baseline_cutoff"])

st.divider()

comp = st.selectbox("Competição", COMPETITIONS, format_func=lambda c: COMP_LABELS.get(c, c))

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
