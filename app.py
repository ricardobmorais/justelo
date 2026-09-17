import streamlit as st
import pandas as pd
import html as html_lib
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


def match_card_html(home, away, date_str, elo_h, elo_a, mu_h, mu_a, home_p, draw_p, away_p, result=None):
    """Cartão de jogo estilo 'app de apostas': crachás com iniciais, chips de
    Elo/xG e as 3 caixas de probabilidade com a mais provável realçada a azul."""
    home_esc, away_esc = html_lib.escape(home), html_lib.escape(away)
    probs = [("home", home_p, home_esc), ("draw", draw_p, "Empate"), ("away", away_p, away_esc)]
    max_key = max(probs, key=lambda t: t[1])[0]

    def box(key, pct, label):
        hi = key == max_key
        border = "#3b82f6" if hi else "#262a37"
        color = "#60a5fa" if hi else "#e5e7eb"
        return (
            f'<div style="flex:1;text-align:center;background:#161923;border:2px solid {border};'
            f'border-radius:10px;padding:14px 6px;">'
            f'<div style="font-size:22px;font-weight:800;color:{color};">{pct*100:.1f}%</div>'
            f'<div style="font-size:12px;color:#8b8f9c;margin-top:2px;">{label}</div></div>'
        )

    boxes_html = "".join(box(k, p, l) for k, p, l in probs)

    result_html = ""
    if result is not None:
        result_html = (
            '<div style="margin-top:12px;background:#0f2e1c;color:#4ade80;border-radius:8px;'
            f'padding:8px 12px;font-size:14px;">✅ Resultado final: {result}</div>'
        )

    return f"""
<div style="background:#12141c;border:1px solid #262a37;border-radius:14px;padding:16px 18px;margin-bottom:14px;">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <div style="width:28px;height:28px;border-radius:50%;background:#2563eb;color:white;display:flex;
                align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;">{home_esc[:1].upper()}</div>
    <span style="font-weight:700;font-size:17px;">{home_esc}</span>
    <span style="color:#8b8f9c;">vs</span>
    <div style="width:28px;height:28px;border-radius:50%;background:#db2777;color:white;display:flex;
                align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;">{away_esc[:1].upper()}</div>
    <span style="font-weight:700;font-size:17px;">{away_esc}</span>
  </div>
  <div style="color:#8b8f9c;font-size:13px;margin:4px 0 10px 0;">{html_lib.escape(str(date_str))}</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">
    <span style="background:#1c2030;border-radius:20px;padding:4px 12px;font-size:13px;color:#c7cad1;">Elo {elo_h:.0f}</span>
    <span style="background:#1c2030;border-radius:20px;padding:4px 12px;font-size:13px;color:#c7cad1;">Elo {elo_a:.0f}</span>
    <span style="background:#1c2030;border-radius:20px;padding:4px 12px;font-size:13px;color:#c7cad1;">🌐 xG médio: {mu_h:.2f} - {mu_a:.2f}</span>
  </div>
  <div style="display:flex;gap:10px;">{boxes_html}</div>
  {result_html}
</div>
"""


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
            result = None
            if pd.notna(row["HomeGoals"]):
                result = f"{int(row['HomeGoals'])} - {int(row['AwayGoals'])}"
            st.markdown(
                match_card_html(
                    row["Home Team"], row["Away Team"], row["Date"], eh, ea,
                    pred["mu_h"], pred["mu_a"], pred["home_win"], pred["draw"], pred["away_win"],
                    result=result,
                ),
                unsafe_allow_html=True,
            )

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
