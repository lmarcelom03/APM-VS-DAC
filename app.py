\
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="APM & DAC 2026 — Peru Compras", page_icon="📊", layout="wide")

# ---------- THEME / STYLES ----------
with open("assets/styles.css","r",encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="header-ribbon">
  <h2 style="margin:0">APM & DAC 2026 — Tablero Ejecutivo</h2>
  <p style="margin:0; opacity:.9">Visual dinámico con estética institucional (Peru Compras)</p>
</div>
""", unsafe_allow_html=True)

# ---------- DATA LOADING ----------
DATA_DIR = Path("data")

@st.cache_data(show_spinner=False)
def load_default():
    apm_gen = pd.read_csv(DATA_DIR/"apm_generica_2026.csv")
    dac_gen = pd.read_csv(DATA_DIR/"dac_generica_2026.csv")
    apm_cc  = pd.read_csv(DATA_DIR/"apm_cc_2026.csv")
    dac_cc  = pd.read_csv(DATA_DIR/"dac_cc_2026.csv")
    return apm_gen, dac_gen, apm_cc, dac_cc

apm_gen, dac_gen, apm_cc, dac_cc = load_default()

# Optional: allow user to upload a fresh Excel and recompute basic summaries
uploaded = st.file_uploader("Sube el Excel original para recalcular (opcional)", type=["xlsx"], accept_multiple_files=False)
if uploaded is not None:
    try:
        xl = pd.ExcelFile(uploaded)
        # Try rebuild minimal summaries (mismo método del pack)
        res_apm_raw = xl.parse("RES_APM_GG")
        apm_work = res_apm_raw.iloc[3:].copy()
        apm_header_vals = res_apm_raw.iloc[2, 2:7].tolist()
        apm_header_cols = res_apm_raw.columns[2:7].tolist()
        rename_map = dict(zip(apm_header_cols, apm_header_vals))
        if 'Unnamed: 7' in apm_work.columns:
            rename_map['Unnamed: 7'] = 'TOTAL'
        apm_work.rename(columns=rename_map, inplace=True)
        gen_cols_apm = [c for c in apm_header_vals if isinstance(c, str)] + (['TOTAL'] if 'TOTAL' in apm_work.columns else [])
        for c in gen_cols_apm:
            apm_work[c] = pd.to_numeric(apm_work[c], errors='coerce')
        apm_gen = apm_work.dropna(subset=gen_cols_apm, how='all')[gen_cols_apm].sum(numeric_only=True).reset_index()
        apm_gen.columns = ['Genérica','APM_2026']

        res_da_raw = xl.parse("RES_DA_GG")
        start_idx = res_da_raw.index[res_da_raw['Unnamed: 10'] == 'GENÉRICAS DE GASTO'][0] + 1
        end_idx = res_da_raw.index[res_da_raw['Unnamed: 10'] == 'TOTAL'][0]
        dac_gen = res_da_raw.loc[start_idx:end_idx-1, ['Unnamed: 10','Unnamed: 11']].copy()
        dac_gen.columns = ['Genérica','DAC_2026']
        dac_gen['DAC_2026'] = pd.to_numeric(dac_gen['DAC_2026'], errors='coerce')

        apmgg = xl.parse("APM-GG")
        header_idx = 3
        apmgg_header_vals = apmgg.iloc[header_idx, 2:8].tolist()
        apmgg2 = apmgg.iloc[4:].copy()
        rename_map2 = {apmgg.columns[i+2]: apmgg_header_vals[i] for i in range(len(apmgg_header_vals))}
        apmgg2.rename(columns=rename_map2, inplace=True)
        apmgg2.rename(columns={apmgg.columns[1]:'CC_T', apmgg.columns[0]:'programa2'}, inplace=True)
        total_cols = [c for c in apmgg2.columns if isinstance(c, str) and c.strip().lower()=='total general']
        if total_cols:
            apmgg2['Total_APM'] = apmgg2[total_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        apm_cc = apmgg2.dropna(subset=['CC_T'])[['CC_T','Total_APM']].dropna()

        da = xl.parse("Res_DA")
        header_idx_da = 3
        headers_da = da.iloc[header_idx_da, 2:].tolist()
        cols_da = da.columns[2:].tolist()
        rename_map_da = {da.columns[0]:'CP', da.columns[1]:'CC_T'}
        for c_name, hval in zip(cols_da, headers_da):
            if isinstance(hval, str) and len(hval.strip())>0:
                rename_map_da[c_name] = hval
        da2 = da.iloc[6:].copy()
        da2.rename(columns=rename_map_da, inplace=True)
        gen_cols_da = [c for c in da2.columns if isinstance(c, str) and (c.startswith('2.') or 'TOTAL' in c.upper())]
        for c in gen_cols_da:
            da2[c] = pd.to_numeric(da2[c], errors='coerce')
        total_col_da = next((c for c in gen_cols_da if 'TOTAL' in c.upper()), None)
        dac_cc = da2.dropna(subset=['CC_T'])[['CC_T', total_col_da]].rename(columns={total_col_da:'Total_DAC'}).dropna()

    except Exception as e:
        st.warning(f"No se pudo procesar el Excel subido: {e}")

# ---------- KPI HEADER ----------
def fmt_money(x):
    try:
        return f"S/ {x:,.0f}".replace(",", ".")
    except:
        return "-"

apm_total = float(apm_gen.loc[apm_gen['Genérica']=='TOTAL','APM_2026'].values[0]) if 'TOTAL' in apm_gen['Genérica'].values else float(apm_gen['APM_2026'].sum())
dac_total = float(dac_gen['DAC_2026'].sum())

k1, k2, k3 = st.columns([1,1,1])
with k1:
    st.markdown('<div class="kpi-card"><div class="kpi-label">APM 2026</div><div class="kpi-value">'+fmt_money(apm_total)+'</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi-card"><div class="kpi-label">DAC 2026</div><div class="kpi-value">'+fmt_money(dac_total)+'</div></div>', unsafe_allow_html=True)
with k3:
    share_apm = apm_total/(apm_total+dac_total) if (apm_total+dac_total)>0 else 0
    st.markdown('<div class="kpi-card"><div class="kpi-label">% APM en Total 2026</div><div class="kpi-value">'+f"{share_apm*100:.1f}%"+"</div></div>", unsafe_allow_html=True)

st.write("")

# ---------- CONTROLS ----------
c1, c2 = st.columns([2,3])
with c1:
    gen_filter = st.multiselect("Filtrar Genéricas", options=sorted(apm_gen['Genérica'].unique()), default=None)
with c2:
    cc_top_n = st.slider("Top N Centros de Costo", 5, 20, 12)

def apply_gen_filter(df, col_name):
    if gen_filter:
        return df[df[col_name].isin(gen_filter)].copy()
    return df.copy()

apm_gen_f = apply_gen_filter(apm_gen, "Genérica")
dac_gen_f = apply_gen_filter(dac_gen, "Genérica")

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 Resumen", "🟥 APM", "🟦 DAC", "⚖️ Comparativo", "🏢 Centros de Costo"])

with tab1:
    # Donut APM vs DAC
    donut = go.Figure(data=[go.Pie(
        labels=["APM 2026","DAC 2026"],
        values=[apm_total, dac_total],
        hole=.6,
        hovertemplate="%{label}<br><b>S/ %{value:,.0f}</b><extra></extra>"
    )])
    donut.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(donut, use_container_width=True)

    st.caption("Vista general del mix APM/DAC en 2026.")

with tab2:
    apm_plot = px.bar(
        apm_gen_f[apm_gen_f['Genérica']!='TOTAL'],
        x="Genérica", y="APM_2026",
        hover_data={"APM_2026":":,.0f"},
        labels={"APM_2026":"Soles"}
    )
    apm_plot.update_layout(height=520, xaxis_tickangle=-30, margin=dict(l=4,r=8,t=28,b=60))
    st.plotly_chart(apm_plot, use_container_width=True)
    st.dataframe(apm_gen, use_container_width=True)

with tab3:
    dac_plot = px.bar(
        dac_gen_f,
        x="Genérica", y="DAC_2026",
        hover_data={"DAC_2026":":,.0f"},
        labels={"DAC_2026":"Soles"}
    )
    dac_plot.update_layout(height=520, xaxis_tickangle=-30, margin=dict(l=4,r=8,t=28,b=60))
    st.plotly_chart(dac_plot, use_container_width=True)
    st.dataframe(dac_gen, use_container_width=True)

with tab4:
    comp = apm_gen_f[apm_gen_f['Genérica']!='TOTAL'].merge(dac_gen_f, on="Genérica", how="outer").fillna(0)
    comp_m = comp.melt(id_vars="Genérica", value_vars=["APM_2026","DAC_2026"], var_name="Tipo", value_name="Soles")
    comp_plot = px.bar(comp_m, x="Genérica", y="Soles", color="Tipo", barmode="group",
                       hover_data={"Soles":":,.0f"})
    comp_plot.update_layout(height=520, xaxis_tickangle=-30, margin=dict(l=4,r=8,t=28,b=60))
    st.plotly_chart(comp_plot, use_container_width=True)

with tab5:
    colA, colB = st.columns(2)
    with colA:
        apm_top = apm_cc.sort_values("Total_APM", ascending=False).head(cc_top_n)
        apm_h = px.bar(apm_top, x="Total_APM", y="CC_T", orientation="h", hover_data={"Total_APM":":,.0f"})
        apm_h.update_layout(height=520, margin=dict(l=4,r=8,t=28,b=20), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(apm_h, use_container_width=True)
    with colB:
        dac_top = dac_cc.sort_values("Total_DAC", ascending=False).head(cc_top_n)
        dac_h = px.bar(dac_top, x="Total_DAC", y="CC_T", orientation="h", hover_data={"Total_DAC":":,.0f"})
        dac_h.update_layout(height=520, margin=dict(l=4,r=8,t=28,b=20), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(dac_h, use_container_width=True)

st.caption("Hecho con ❤️ en Streamlit + Plotly")
