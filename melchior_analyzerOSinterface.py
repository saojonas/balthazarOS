import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timezone, timedelta
from supabase import create_client
import plotly.express as px

# =========================
# CONFIG (DEPLOY-SAFE)
# =========================

def get_secret(name: str) -> str:
    """
    Produção (Streamlit Cloud): usa st.secrets
    Local (opcional): fallback em variáveis de ambiente
    """
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.getenv(name, "").strip()


SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")

TABLE_OS = "os"
DEFAULT_LOOKBACK_DAYS = 365

# Status oficiais do teu modelo
STATUS_PREOS_ABERTAS = ["BLOQUEADA", "RTE"]
STATUS_OS_EXEC = ["EM_EXECUCAO"]
STATUS_OS_DONE = ["CONCLUIDA"]

# =========================
# SUPABASE
# =========================
def connect():
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("Credenciais do Supabase não encontradas.")
        st.info("No Streamlit Cloud: Settings → Secrets → defina SUPABASE_URL e SUPABASE_KEY.")
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_os_cached(lookback_days: int) -> pd.DataFrame:
    """Carrega dados da tabela public.os dentro de uma janela (lookback em dias)."""
    client = connect()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    resp = (
        client.table(TABLE_OS)
        .select("*")
        .gte("created_at", cutoff)
        .execute()
    )
    return pd.DataFrame(resp.data or [])


# =========================
# NORMALIZAÇÃO + CRITICIDADE
# =========================
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Datas
    for c in ["created_at", "operational_born_at", "execution_started_at", "execution_finished_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    # blocked_by: lista garantida
    if "blocked_by" not in df.columns:
        df["blocked_by"] = [[] for _ in range(len(df))]
    else:
        def _as_list(x):
            if isinstance(x, list):
                return x
            if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
                return []
            if isinstance(x, str):
                try:
                    v = json.loads(x)
                    return v if isinstance(v, list) else []
                except Exception:
                    return []
            return []
        df["blocked_by"] = df["blocked_by"].apply(_as_list)

    # =========================
    # equipamento (jsonb) → extrair desc (fallback)
    # =========================
    if "equipamento" in df.columns:
        def _equip_desc(x):
            if isinstance(x, dict):
                return x.get("desc", "")
            if isinstance(x, str):
                try:
                    v = json.loads(x)
                    if isinstance(v, dict):
                        return v.get("desc", "")
                except Exception:
                    return ""
            return ""
        df["equipamento_desc"] = df["equipamento"].apply(_equip_desc)
    else:
        df["equipamento_desc"] = ""

    now_utc = datetime.now(timezone.utc)

    # Idade em dias (base created_at)
    df["age_days"] = (now_utc - df["created_at"]).dt.total_seconds() / 86400.0
    df["age_days"] = df["age_days"].clip(lower=0)

    # Idade em horas corridas
    df["age_hours"] = (df["age_days"] * 24.0).round(1)

    df["blocked_count"] = df["blocked_by"].apply(lambda lst: len(lst) if isinstance(lst, list) else 0)
    df["blocked_str"] = df["blocked_by"].apply(lambda lst: ", ".join(lst) if lst else "")

    # prioridade (pode estar vazio no piloto)
    df["prioridade_norm"] = df.get("prioridade", "").fillna("").astype(str).str.strip().str.upper()

    # =========================
    # CRITICIDADE (0-100) — índice interno de atenção
    # =========================
    prio_map = {
        "EMERGENCIAL": 40,
        "URGENTE": 35,
        "ALTA": 30,
        "NORMAL": 15,
        "BAIXA": 5,
        "": 10,
        "NAN": 10,
        "NONE": 10,
    }
    prio_score = df["prioridade_norm"].map(lambda x: prio_map.get(x, 12))

    status_map = {
        "BLOQUEADA": 28,
        "EM_EXECUCAO": 22,
        "RTE": 14,
        "CONCLUIDA": 0,
    }
    status_score = df["status"].fillna("").map(lambda s: status_map.get(str(s), 8))

    # Aging: 0..25 pontos até 30 dias (capado)
    age_score = (df["age_days"] / 30.0 * 25.0).clip(0, 25)

    # Bloqueios: 0..15 pontos
    block_score = (df["blocked_count"] * 5.0).clip(0, 15)

    # OS pesa um pouco mais que PRE_OS
    type_bonus = np.where(df["type"].eq("OS"), 6.0, 0.0)

    # “Compras” e “Sem contingente” são travas clássicas: +5
    def has_term(lst, term):
        term = term.upper()
        return any(str(x).strip().upper() == term for x in (lst or []))

    special = df["blocked_by"].apply(
        lambda lst: 5.0 if (has_term(lst, "COMPRAS") or has_term(lst, "SEM_CONTINGENTE")) else 0.0
    )

    df["criticality_score"] = (prio_score + status_score + age_score + block_score + type_bonus + special).clip(0, 100)

    def band(x):
        if x >= 75:
            return "CRÍTICO"
        if x >= 50:
            return "ALTO"
        if x >= 25:
            return "MÉDIO"
        return "BAIXO"

    df["criticality_band"] = df["criticality_score"].apply(band)

    # Emoji para impacto produtivo (se existir)
    if "impacto_produtivo" in df.columns:
        def _impact_emoji(v):
            if v is True:
                return "🔴"
            if v is False:
                return "🔵"
            s = str(v).strip().lower()
            if s in ("true", "1", "sim", "yes"):
                return "🔴"
            if s in ("false", "0", "nao", "não", "no"):
                return "🔵"
            return "—"
        df["impacto_produtivo_emoji"] = df["impacto_produtivo"].apply(_impact_emoji)
    else:
        df["impacto_produtivo_emoji"] = "—"

    return df


# =========================
# UI
# =========================
st.set_page_config(page_title="MAGI@Melchior — Executive", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-note { opacity: 0.75; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("MAGI@Melchior — Painel Executivo")
st.markdown('<div class="small-note">Relatório em tempo real de gestão de frentes de serviço.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuração")
    lookback_days = st.number_input(
        "Janela (dias)", min_value=7, max_value=3650, value=DEFAULT_LOOKBACK_DAYS, step=30
    )
    if st.button("🔄 Recarregar dados"):
        fetch_os_cached.clear()

    st.divider()
    st.header("Filtros (executivo)")
    band_sel = st.multiselect(
        "Criticidade", ["CRÍTICO", "ALTO", "MÉDIO", "BAIXO"], default=["CRÍTICO", "ALTO", "MÉDIO", "BAIXO"]
    )
    type_sel = st.multiselect("Tipo", ["PRE_OS", "OS"], default=["PRE_OS", "OS"])
    status_sel = st.multiselect(
        "Status", ["BLOQUEADA", "RTE", "EM_EXECUCAO", "CONCLUIDA"],
        default=["BLOQUEADA", "RTE", "EM_EXECUCAO", "CONCLUIDA"]
    )
    equip_q = st.text_input("Buscar ativo (TAG/descrição)", value="")
    only_blocked = st.checkbox("Somente bloqueadas (blocked_by ≠ [])", value=False)

with st.spinner("Carregando e preparando dados..."):
    df = normalize(fetch_os_cached(int(lookback_days)))

if df.empty:
    st.warning("Sem dados retornados. Verifique credenciais, RLS e se a tabela `os` existe.")
    st.stop()

# Aplica filtros
d = df.copy()
d = d[d["criticality_band"].isin(band_sel)]
d = d[d["type"].isin(type_sel)]
d = d[d["status"].isin(status_sel)]
if only_blocked:
    d = d[d["blocked_count"] > 0]
if equip_q.strip():
    s = equip_q.strip().upper()
    d = d[
        d["equip_id"].fillna("").astype(str).str.upper().str.contains(s, na=False)
        | d["equip_desc"].fillna("").astype(str).str.upper().str.contains(s, na=False)
        | d["equipamento_desc"].fillna("").astype(str).str.upper().str.contains(s, na=False)
    ]

# KPIs
preos_abertas = d[(d["type"] == "PRE_OS") & (d["status"].isin(STATUS_PREOS_ABERTAS))]
preos_bloq = preos_abertas[preos_abertas["status"] == "BLOQUEADA"]
os_exec = d[(d["type"] == "OS") & (d["status"].isin(STATUS_OS_EXEC))]
os_done = d[(d["type"] == "OS") & (d["status"].isin(STATUS_OS_DONE))]
backlog = d[d["status"] != "CONCLUIDA"]

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Backlog", f"{len(backlog)}")
k2.metric("Pré-OS abertas", f"{len(preos_abertas)}")
k3.metric("Pré-OS bloqueadas", f"{len(preos_bloq)}")
k4.metric("OS em execução", f"{len(os_exec)}")
k5.metric("OS concluídas", f"{len(os_done)}")
k6.metric("Críticos (≥75)", f"{int((d['criticality_band']=='CRÍTICO').sum())}")

st.divider()

# =========================
# MATRIZ DE CRITICIDADE
# =========================
st.subheader("Como calculamos a criticidade (leitura rápida)")

matriz_ludica = pd.DataFrame(
    [
        {"Categoria": "🚨 Urgência (Prioridade)", "Origem": "prioridade", "Peso máximo": "até 40 pts",
         "Descrição breve": "Se foi marcado como emergencial/alta, sobe na fila."},
        {"Categoria": "🧱 Situação (Status)", "Origem": "status", "Peso máximo": "até 28 pts",
         "Descrição breve": "Travado (BLOQUEADA) pesa mais do que pronto (RTE)."},
        {"Categoria": "⏳ Tempo aberto (Idade)", "Origem": "age_days", "Peso máximo": "até 25 pts",
         "Descrição breve": "Quanto mais velho, mais grave do ponto de vista de gestão."},
        {"Categoria": "🔒 Impedimentos (Bloqueios)", "Origem": "blocked_by", "Peso máximo": "até 15 pts",
         "Descrição breve": "Mais bloqueios = mais crítico."},
        {"Categoria": "🏭 Tipo de demanda", "Origem": "type", "Peso máximo": "até 6 pts",
         "Descrição breve": "OS operacional pesa mais do que Pré-OS administrativa."},
        {"Categoria": "⚠️ Gargalos clássicos", "Origem": "blocked_by (COMPRAS / SEM_CONTINGENTE)", "Peso máximo": "até 5 pts",
         "Descrição breve": "Se o bloqueio é compras/contingente, sobe um pouco no alerta."},
    ]
)

st.dataframe(matriz_ludica, use_container_width=True, hide_index=True)

st.caption(
    "📌 **Legenda de criticidade (faixas):** "
    "**CRÍTICO (≥75)** = exige ação imediata; "
    "**ALTO (50–74)** = atenção prioritária; "
    "**MÉDIO (25–49)** = acompanhar e programar; "
    "**BAIXO (<25)** = rotina/sem urgência. "
    "Obs.: Pontuação interna, sujeita a revisões."
)

st.divider()

# =========================
# GRÁFICOS (4)
# =========================
c1, c2 = st.columns([1.05, 0.95])

with c1:
    st.subheader("Distribuição por criticidade")
    band_counts = (
        d["criticality_band"].value_counts()
        .reindex(["CRÍTICO", "ALTO", "MÉDIO", "BAIXO"])
        .fillna(0)
        .reset_index()
    )
    band_counts.columns = ["criticidade", "qtd"]
    fig = px.pie(band_counts, names="criticidade", values="qtd", hole=0.55)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fluxo (funil simples)")
    flow = pd.DataFrame(
        {
            "etapa": ["Pré-OS BLOQUEADA", "Pré-OS RTE", "OS EM_EXECUCAO", "OS CONCLUIDA"],
            "qtd": [
                int(((d["type"] == "PRE_OS") & (d["status"] == "BLOQUEADA")).sum()),
                int(((d["type"] == "PRE_OS") & (d["status"] == "RTE")).sum()),
                int(((d["type"] == "OS") & (d["status"] == "EM_EXECUCAO")).sum()),
                int(((d["type"] == "OS") & (d["status"] == "CONCLUIDA")).sum()),
            ],
        }
    )
    fig2 = px.bar(flow, x="etapa", y="qtd")
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    st.subheader("Bloqueios por motivo (Pré-OS BLOQUEADA)")
    if preos_bloq.empty:
        st.info("Sem Pré-OS bloqueadas no recorte.")
    else:
        exploded = preos_bloq[["id", "blocked_by"]].explode("blocked_by")
        exploded["blocked_by"] = exploded["blocked_by"].fillna("SEM_MOTIVO").astype(str)
        motives = exploded["blocked_by"].value_counts().reset_index()
        motives.columns = ["motivo", "qtd"]
        motives = motives.head(12)
        fig3 = px.bar(motives, x="qtd", y="motivo", orientation="h")
        fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Mapa de calor: Status × Criticidade")
    heat = d.groupby(["status", "criticality_band"]).size().reset_index(name="qtd")
    band_order = ["CRÍTICO", "ALTO", "MÉDIO", "BAIXO"]
    status_order = ["BLOQUEADA", "RTE", "EM_EXECUCAO", "CONCLUIDA"]
    heat["criticality_band"] = pd.Categorical(heat["criticality_band"], categories=band_order, ordered=True)
    heat["status"] = pd.Categorical(heat["status"], categories=status_order, ordered=True)
    pivot = heat.pivot_table(index="status", columns="criticality_band", values="qtd", fill_value=0)

    fig_heat = px.imshow(pivot, text_auto=True, aspect="auto")
    fig_heat.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Criticidade",
        yaxis_title="Status",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# =========================
# DETALHES (GERÊNCIA)
# =========================
st.subheader("Detalhes — visão gerencial")

details = (
    d[d["status"] != "CONCLUIDA"]
    .sort_values(["criticality_score", "age_days"], ascending=[False, False])
    .copy()
)

# score com 2 casas
if "criticality_score" in details.columns:
    details["criticality_score"] = details["criticality_score"].astype(float).round(2)

def keep_existing(df_: pd.DataFrame, cols_: list) -> list:
    return [c for c in cols_ if c in df_.columns]

cols_gerencia = keep_existing(
    details,
    ["criticality_band", "equip_desc", "equipamento_desc", "classe", "tipo_intervencao", "descricao", "impacto_produtivo_emoji"]
)

rename_map = {
    "criticality_band": "Criticidade",
    "equip_desc": "Ativo (campo)",
    "equipamento_desc": "Ativo",
    "classe": "Classe",
    "tipo_intervencao": "Tipo de intervenção",
    "descricao": "Descrição",
    "impacto_produtivo_emoji": "Impacto produtivo",
}

show = details[cols_gerencia].rename(columns=rename_map)
st.dataframe(show, use_container_width=True, hide_index=True)
st.caption("Impacto produtivo: 🔴 = sim / 🔵 = não / — = não informado.")

with st.expander("Abrir visão PCM (detalhada)"):
    pcm_cols = keep_existing(
        details,
        [
            "criticality_band", "criticality_score", "type", "status",
            "age_hours", "age_days", "id",
            "equip_id", "equip_desc", "equipamento_desc", "equip_cat", "equip_gr",
            "cc", "frota", "solicitante",
            "classe", "motivo", "tipo_intervencao", "prioridade_norm",
            "blocked_str", "purchase_ref_type", "purchase_ref_no", "purchase_eta", "purchase_note",
            "execution_started_at", "execution_finished_at"
        ]
    )
    pcm = details[pcm_cols].copy()
    if "criticality_score" in pcm.columns:
        pcm["criticality_score"] = pcm["criticality_score"].astype(float).round(2)
    st.dataframe(pcm, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Baixar visão gerencial (CSV)",
    data=show.to_csv(index=False).encode("utf-8"),
    file_name="melchior_visao_gerencia.csv",
    mime="text/csv",
)
