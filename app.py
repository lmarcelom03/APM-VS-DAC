import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

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
    apm_gen = pd.read_csv(DATA_DIR/"clean_res_apm_gg_generica.csv")
    dac_gen = pd.read_csv(DATA_DIR/"clean_res_da_gg_generica.csv")
    apm_cc  = pd.read_csv(DATA_DIR/"apm_cc_2026.csv")
    dac_cc  = pd.read_csv(DATA_DIR/"dac_cc_2026.csv")
    return apm_gen, dac_gen, apm_cc, dac_cc


def normalize_cc_labels(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply known fixes to cost center labels so dashboards stay consistent."""
    replacements = {
        "0016-OA-COVID": "0016-DE (PROMOCION)",
    }
    df[column] = df[column].replace(replacements)
    return df


def split_total_rows(df: pd.DataFrame, label_col: str):
    """Return dataframe without rows whose label is a total (case-insensitive)."""
    label_series = (
        df[label_col]
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.strip()
        .str.upper()
    )
    total_mask = label_series.str.startswith("TOTAL")
    return df.loc[~total_mask].copy(), df.loc[total_mask].copy()


def build_display_table(main_df: pd.DataFrame, totals_df: pd.DataFrame, label_col: str, value_col: str, total_label: str):
    """Combine main data with totals, generating a synthetic total row if necessary."""
    if totals_df is not None and not totals_df.empty:
        totals_copy = totals_df.copy()
        totals_copy[value_col] = pd.to_numeric(totals_copy[value_col], errors="coerce").round(2)
        return pd.concat([main_df, totals_copy], ignore_index=True)
    synthetic_total = pd.DataFrame({label_col: [total_label], value_col: [main_df[value_col].sum()]})
    synthetic_total[value_col] = synthetic_total[value_col].round(2)
    return pd.concat([main_df, synthetic_total], ignore_index=True)


def add_share_columns(df: pd.DataFrame, value_col: str, share_col: str = "Participación %") -> pd.DataFrame:
    """Return a copy of ``df`` sorted by ``value_col`` with share metrics."""
    ordered = df.sort_values(value_col, ascending=False).reset_index(drop=True).copy()
    total = ordered[value_col].sum()
    if total <= 0:
        ordered[share_col] = 0.0
        ordered["Participación acumulada %"] = 0.0
        return ordered
    ordered[share_col] = (ordered[value_col] / total * 100).round(2)
    ordered["Participación acumulada %"] = ordered[share_col].cumsum().round(2)
    return ordered


def build_share_table(main_df: pd.DataFrame, totals_df: pd.DataFrame, label_col: str, value_col: str, total_label: str):
    """Return a table with participation metrics and total row."""
    base = add_share_columns(main_df[[label_col, value_col]], value_col)
    if totals_df is not None and not totals_df.empty:
        totals_copy = totals_df[[label_col, value_col]].copy()
        totals_copy[value_col] = pd.to_numeric(totals_copy[value_col], errors="coerce").round(2)
        totals_copy["Participación %"] = 100.0
        totals_copy["Participación acumulada %"] = 100.0
        return pd.concat([base, totals_copy], ignore_index=True)
    synthetic_total = pd.DataFrame({
        label_col: [total_label],
        value_col: [base[value_col].sum()],
        "Participación %": [100.0],
        "Participación acumulada %": [100.0],
    })
    return pd.concat([base, synthetic_total], ignore_index=True)


def describe_top_segments(df: pd.DataFrame, value_col: str, label_col: str, top_n: int = 3):
    """Return summary text about the top ``top_n`` segments and their contribution."""
    ordered = add_share_columns(df[[label_col, value_col]], value_col)
    top = ordered.head(top_n)
    total_value = ordered[value_col].sum()
    if top.empty or total_value <= 0:
        return "Sin datos suficientes para generar insights."
    top_lines = [
        f"{idx+1}. {row[label_col]} — {fmt_money(row[value_col])} ({row['Participación %']:.1f}%)"
        for idx, row in top.iterrows()
    ]
    top_share = top['Participación %'].sum()
    return "\n".join(top_lines + [f"Los {top_n} primeros concentran el {top_share:.1f}% del total."])


def estimate_cp_generica_distribution(
    cp_df: pd.DataFrame,
    cp_col: str,
    total_col: str,
    gen_df: pd.DataFrame,
    label_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Estimate the breakdown of each cost center across generics using current shares."""
    if cp_df.empty or gen_df.empty:
        return pd.DataFrame(columns=["CP", "Genérica", "Monto"])

    share_df = gen_df[[label_col, value_col]].copy()
    total_value = share_df[value_col].sum()
    if total_value <= 0:
        return pd.DataFrame(columns=["CP", "Genérica", "Monto"])

    share_df["_peso"] = share_df[value_col] / total_value
    cp_rows = cp_df[[cp_col, total_col]].dropna()

    records = []
    for cp_val, total_cp in cp_rows.values:
        if pd.isna(total_cp):
            continue
        for _, gen_row in share_df.iterrows():
            records.append(
                {
                    "CP": cp_val,
                    "Genérica": gen_row[label_col],
                    "Monto": float(total_cp) * float(gen_row["_peso"]),
                }
            )

    return pd.DataFrame(records)


def combine_cp_breakdowns(apm_detail: pd.DataFrame, dac_detail: pd.DataFrame) -> pd.DataFrame:
    """Merge APM and DAC cost-center breakdowns into a unified table."""
    frames = []
    if apm_detail is not None and not apm_detail.empty:
        apm_tmp = apm_detail.copy()
        apm_tmp["Tipo"] = "APM"
        frames.append(apm_tmp)
    if dac_detail is not None and not dac_detail.empty:
        dac_tmp = dac_detail.copy()
        dac_tmp["Tipo"] = "DAC 2026"
        frames.append(dac_tmp)
    if not frames:
        return pd.DataFrame(columns=["CP", "Genérica", "Tipo", "Monto"])
    combined = pd.concat(frames, ignore_index=True)
    combined["Monto"] = pd.to_numeric(combined["Monto"], errors="coerce").fillna(0.0)
    combined["CP"] = combined["CP"].astype(str)
    combined["Genérica"] = combined["Genérica"].astype(str)
    return combined


apm_gen, dac_gen, apm_cc, dac_cc = load_default()
apm_gen["APM_2026"] = pd.to_numeric(apm_gen["APM_2026"], errors="coerce").round(2)
dac_cols_default = [c for c in dac_gen.columns if isinstance(c, str) and "2026" in c]
dac_numeric_default = dac_gen[dac_cols_default].apply(pd.to_numeric, errors="coerce")
dac_gen = pd.DataFrame(
    {
        "Genérica": dac_gen["Genérica"],
        "DAC_2026": dac_numeric_default.sum(axis=1).round(2),
    }
)
apm_cc["Total_APM"] = pd.to_numeric(apm_cc["Total_APM"], errors="coerce").round(2)
dac_cc["Total_DAC"] = pd.to_numeric(dac_cc["Total_DAC"], errors="coerce").round(2)

apm_gen, apm_gen_totals = split_total_rows(apm_gen, "Genérica")
dac_gen, dac_gen_totals = split_total_rows(dac_gen, "Genérica")
apm_cc = normalize_cc_labels(apm_cc, "CC_T")
dac_cc = normalize_cc_labels(dac_cc, "CC_T")
apm_cc, apm_cc_totals = split_total_rows(apm_cc, "CC_T")
dac_cc, dac_cc_totals = split_total_rows(dac_cc, "CC_T")

apm_cp_gen_breakdown = estimate_cp_generica_distribution(
    apm_cc, "CC_T", "Total_APM", apm_gen, "Genérica", "APM_2026"
)
if not apm_cp_gen_breakdown.empty:
    apm_cp_gen_breakdown["Monto"] = apm_cp_gen_breakdown["Monto"].round(2)

dac_cp_gen_breakdown = estimate_cp_generica_distribution(
    dac_cc, "CC_T", "Total_DAC", dac_gen, "Genérica", "DAC_2026"
)
if not dac_cp_gen_breakdown.empty:
    dac_cp_gen_breakdown["Monto"] = dac_cp_gen_breakdown["Monto"].round(2)

cp_breakdown_quality = "estimado"


MONTH_NAMES = [
    "Ene",
    "Feb",
    "Mar",
    "Abr",
    "May",
    "Jun",
    "Jul",
    "Ago",
    "Set",
    "Oct",
    "Nov",
    "Dic",
]


def build_monthly_profile(meta_total: float) -> np.ndarray:
    """Create a smooth profile that grows through the year with a December boost."""
    base_months = np.linspace(2.1, 3.0, 11) * 1_000_000
    december_value = 3.2 * 1_000_000
    profile = np.concatenate([base_months, [december_value]])
    profile_sum = profile.sum()
    if meta_total > 0 and profile_sum > 0:
        scale = meta_total / profile_sum
        profile = profile * scale
    return profile


def build_execution_animation(meta_total: float, avance_total: float, months_elapsed: int):
    """Return dataframes for the animated execution simulation clip."""
    months_elapsed = max(1, min(12, int(months_elapsed)))
    profile = build_monthly_profile(meta_total)

    projected_monthly = profile
    projected_cumulative = np.cumsum(projected_monthly)

    historical_profile = projected_monthly[:months_elapsed].copy()
    hist_sum = historical_profile.sum()
    if avance_total > 0 and hist_sum > 0:
        historical_profile = historical_profile * (avance_total / hist_sum)
    else:
        historical_profile = np.zeros(months_elapsed)

    actual_monthly = np.concatenate([historical_profile, np.zeros(12 - months_elapsed)])
    actual_cumulative = np.cumsum(actual_monthly)

    # Keep execution flat after the last recorded month
    if months_elapsed < 12 and avance_total > 0:
        actual_cumulative[months_elapsed - 1 :] = avance_total

    frames = []
    for idx, month in enumerate(MONTH_NAMES):
        frame_label = f"{idx+1:02d} - {month}"
        for serie, cumulative in (
            ("Proyección anual", projected_cumulative),
            ("Ejecución simulada", actual_cumulative),
        ):
            for month_idx in range(idx + 1):
                frames.append(
                    {
                        "Frame": frame_label,
                        "Mes": MONTH_NAMES[month_idx],
                        "Serie": serie,
                        "Monto acumulado": float(cumulative[month_idx]),
                    }
                )

    animation_df = pd.DataFrame(frames)
    table_df = pd.DataFrame(
        {
            "Mes": MONTH_NAMES,
            "Escenario proyectado (S/)": projected_monthly.round(2),
            "Proyección acumulada (S/)": projected_cumulative.round(2),
            "Ejecución simulada (S/)": actual_monthly.round(2),
            "Ejecución acumulada (S/)": actual_cumulative.round(2),
        }
    )

    return animation_df, table_df

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
        apm_gen['APM_2026'] = apm_gen['APM_2026'].round(2)

        res_da_raw = xl.parse("RES_DA_GG")
        start_idx = res_da_raw.index[res_da_raw['Unnamed: 10'] == 'GENÉRICAS DE GASTO'][0] + 1
        end_idx = res_da_raw.index[res_da_raw['Unnamed: 10'] == 'TOTAL'][0]
        dac_block = res_da_raw.loc[start_idx:end_idx-1].copy()
        dac_block.columns = res_da_raw.iloc[2].fillna(method='ffill')
        dac_block.rename(columns={dac_block.columns[0]: 'Genérica'}, inplace=True)
        dac_cols_2026 = [c for c in dac_block.columns if isinstance(c, str) and '2026' in c]
        dac_gen = dac_block[['Genérica'] + dac_cols_2026].copy()
        dac_gen['DAC_2026'] = dac_gen[dac_cols_2026].apply(pd.to_numeric, errors='coerce').sum(axis=1).round(2)
        dac_gen = dac_gen[['Genérica', 'DAC_2026']]
        apm_gen, apm_gen_totals = split_total_rows(apm_gen, "Genérica")
        dac_gen, dac_gen_totals = split_total_rows(dac_gen, "Genérica")

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
        apm_cc['Total_APM'] = apm_cc['Total_APM'].round(2)
        apm_cc = normalize_cc_labels(apm_cc, "CC_T")
        apm_cc, apm_cc_totals = split_total_rows(apm_cc, "CC_T")

        apm_detail_base = apmgg2.dropna(subset=['CC_T']).copy()
        apm_detail_base = normalize_cc_labels(apm_detail_base, "CC_T")
        detail_cols_apm = [
            c
            for c in apm_detail_base.columns
            if isinstance(c, str)
            and c not in {"programa2", "CC_T", "Total_APM"}
            and "total" not in c.lower()
        ]
        apm_detail_long = apm_detail_base[['CC_T'] + detail_cols_apm].melt(
            id_vars='CC_T', value_vars=detail_cols_apm, var_name='Genérica', value_name='Monto'
        )
        apm_detail_long['Monto'] = pd.to_numeric(apm_detail_long['Monto'], errors='coerce')
        apm_detail_long = apm_detail_long.dropna(subset=['Monto'])
        apm_detail_long, _ = split_total_rows(apm_detail_long, 'Genérica')
        apm_cp_gen_breakdown = apm_detail_long.rename(columns={'CC_T': 'CP', 'Monto': 'Monto'})
        apm_cp_gen_breakdown['Monto'] = apm_cp_gen_breakdown['Monto'].round(2)

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
        dac_cc['Total_DAC'] = dac_cc['Total_DAC'].round(2)
        dac_cc = normalize_cc_labels(dac_cc, "CC_T")
        dac_cc, dac_cc_totals = split_total_rows(dac_cc, "CC_T")

        dac_detail_base = da2.dropna(subset=['CC_T']).copy()
        dac_detail_base = normalize_cc_labels(dac_detail_base, "CC_T")
        detail_cols_dac = [c for c in gen_cols_da if 'TOTAL' not in c.upper()]
        dac_detail_long = dac_detail_base[['CC_T'] + detail_cols_dac].melt(
            id_vars='CC_T', value_vars=detail_cols_dac, var_name='Genérica', value_name='Monto'
        )
        dac_detail_long['Monto'] = pd.to_numeric(dac_detail_long['Monto'], errors='coerce')
        dac_detail_long = dac_detail_long.dropna(subset=['Monto'])
        dac_detail_long, _ = split_total_rows(dac_detail_long, 'Genérica')
        dac_cp_gen_breakdown = dac_detail_long.rename(columns={'CC_T': 'CP', 'Monto': 'Monto'})
        dac_cp_gen_breakdown['Monto'] = dac_cp_gen_breakdown['Monto'].round(2)

        apm_detail_available = not apm_cp_gen_breakdown.empty
        dac_detail_available = not dac_cp_gen_breakdown.empty

        if not apm_detail_available:
            apm_cp_gen_breakdown = estimate_cp_generica_distribution(
                apm_cc, "CC_T", "Total_APM", apm_gen, "Genérica", "APM_2026"
            )
            if not apm_cp_gen_breakdown.empty:
                apm_cp_gen_breakdown["Monto"] = apm_cp_gen_breakdown["Monto"].round(2)

        if not dac_detail_available:
            dac_cp_gen_breakdown = estimate_cp_generica_distribution(
                dac_cc, "CC_T", "Total_DAC", dac_gen, "Genérica", "DAC_2026"
            )
            if not dac_cp_gen_breakdown.empty:
                dac_cp_gen_breakdown["Monto"] = dac_cp_gen_breakdown["Monto"].round(2)

        if apm_detail_available and dac_detail_available:
            cp_breakdown_quality = "real"
        elif apm_detail_available or dac_detail_available:
            cp_breakdown_quality = "mixto"
        else:
            cp_breakdown_quality = "estimado"

    except Exception as e:
        st.warning(f"No se pudo procesar el Excel subido: {e}")

# ---------- KPI HEADER ----------
def fmt_money(x):
    try:
        return f"S/ {x:,.0f}".replace(",", ".")
    except:
        return "-"

apm_total = float(apm_gen['APM_2026'].sum())
dac_total = float(dac_gen['DAC_2026'].sum())

k1, k2, k3 = st.columns([1,1,1])
with k1:
    st.markdown('<div class="kpi-card"><div class="kpi-label">APM 2026</div><div class="kpi-value">'+fmt_money(apm_total)+'</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi-card"><div class="kpi-label">DAC 2026</div><div class="kpi-value">'+fmt_money(dac_total)+'</div></div>', unsafe_allow_html=True)
with k3:
    share_apm = apm_total/(apm_total+dac_total) if (apm_total+dac_total)>0 else 0
    st.markdown('<div class="kpi-card"><div class="kpi-label">% APM en Total 2026</div><div class="kpi-value">'+f"{share_apm*100:.1f}%"+'</div></div>', unsafe_allow_html=True)

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
apm_share_full = add_share_columns(apm_gen, "APM_2026")
dac_share_full = add_share_columns(dac_gen, "DAC_2026")
apm_share_filtered = add_share_columns(apm_gen_f, "APM_2026")
dac_share_filtered = add_share_columns(dac_gen_f, "DAC_2026")
cp_gen_distribution = combine_cp_breakdowns(apm_cp_gen_breakdown, dac_cp_gen_breakdown)
if not cp_gen_distribution.empty():
    cp_gen_distribution["Monto"] = cp_gen_distribution["Monto"].round(2)

with st.expander("💡 Insights automatizados", expanded=False):
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("**Principales genéricas APM**")
        st.markdown(describe_top_segments(apm_gen, "APM_2026", "Genérica"))
    with col_i2:
        st.markdown("**Principales genéricas DAC 2026**")
        st.markdown(describe_top_segments(dac_gen, "DAC_2026", "Genérica"))

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 Resumen", "🟥 APM + DAC", "🧭 CP & Genéricas", "🏢 Centros de Costo", "🎯 Seguimiento de metas"])

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

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Top 5 genéricas APM (S/ y %)**")
        st.dataframe(
            apm_share_full.head(5).rename(columns={
                "APM_2026": "Monto (S/)",
                "Participación %": "Participación %",
                "Participación acumulada %": "Participación acum. %",
            }),
            hide_index=True,
            use_container_width=True,
        )
    with col_s2:
        st.markdown("**Top 5 genéricas DAC 2026 (S/ y %)**")
        st.dataframe(
            dac_share_full.head(5).rename(columns={
                "DAC_2026": "Monto (S/)",
                "Participación %": "Participación %",
                "Participación acumulada %": "Participación acum. %",
            }),
            hide_index=True,
            use_container_width=True,
        )

with tab2:
    st.markdown("### Distribución APM + DAC por genérica")
    combined_gen = apm_gen_f.merge(dac_gen_f, on="Genérica", how="outer").fillna(0)
    if combined_gen.empty:
        st.info("No hay genéricas para mostrar con los filtros actuales.")
    else:
        combined_gen["Total"] = combined_gen["APM_2026"] + combined_gen["DAC_2026"]
        combined_long = combined_gen.melt(
            id_vars=["Genérica", "Total"],
            value_vars=["APM_2026", "DAC_2026"],
            var_name="Tipo",
            value_name="Monto"
        )
        combined_long["Tipo"] = combined_long["Tipo"].replace({
            "APM_2026": "APM",
            "DAC_2026": "DAC 2026"
        })
        combined_long["Monto"] = combined_long["Monto"].round(2)
        total_combined = combined_long["Monto"].sum()
        if total_combined > 0:
            combined_long["Share %"] = (combined_long["Monto"] / total_combined * 100).round(2)
        else:
            combined_long["Share %"] = 0.0
        combined_long.sort_values(by="Total", ascending=False, inplace=True)

        view_mode_combined = st.radio(
            "Modo de visualización",
            ["Montos (S/)", "Participación (%)"],
            horizontal=True,
            key="apm_dac_mode"
        )

        if view_mode_combined == "Participación (%)":
            y_col = "Share %"
            labels = {y_col: "% del total combinado"}
            hover_data = {
                "Share %": ":.2f",
                "Monto": ":,.0f"
            }
        else:
            y_col = "Monto"
            labels = {y_col: "Soles"}
            hover_data = {
                "Monto": ":,.0f",
                "Share %": ":.2f"
            }

        gen_order = combined_gen.sort_values("Total", ascending=False)["Genérica"].tolist()
        combined_plot = px.bar(
            combined_long,
            x="Genérica",
            y=y_col,
            color="Tipo",
            barmode="stack",
            hover_data=hover_data,
            labels=labels,
            category_orders={"Genérica": gen_order},
        )
        combined_plot.update_layout(height=520, xaxis_tickangle=-30, margin=dict(l=4, r=8, t=28, b=60))
        st.plotly_chart(combined_plot, use_container_width=True)

        combined_export = combined_gen.sort_values("Total", ascending=False)
        st.download_button(
            "📥 Descargar datos combinados (CSV)",
            data=combined_export.to_csv(index=False).encode("utf-8"),
            file_name="apm_dac_genericas_2026.csv",
            mime="text/csv",
        )

        col_apm_extra1, col_apm_extra2 = st.columns(2)
        with col_apm_extra1:
            if apm_share_filtered.empty:
                st.info("Sin datos APM para generar treemap con los filtros actuales.")
            else:
                apm_treemap = px.treemap(
                    apm_share_filtered,
                    path=["Genérica"],
                    values="APM_2026",
                    color="Participación %",
                    color_continuous_scale="Reds",
                )
                apm_treemap.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(apm_treemap, use_container_width=True)
        with col_apm_extra2:
            if dac_share_filtered.empty:
                st.info("Sin datos DAC 2026 para generar treemap con los filtros actuales.")
            else:
                dac_treemap = px.treemap(
                    dac_share_filtered,
                    path=["Genérica"],
                    values="DAC_2026",
                    color="Participación %",
                    color_continuous_scale="Blues",
                )
                dac_treemap.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(dac_treemap, use_container_width=True)

        tab_apm_table, tab_dac_table = st.tabs(["Tabla APM", "Tabla DAC 2026"])
        with tab_apm_table:
            apm_display = build_share_table(apm_gen, apm_gen_totals, "Genérica", "APM_2026", "TOTAL GENERAL")
            st.dataframe(
                apm_display.rename(columns={
                    "APM_2026": "Monto (S/)",
                }),
                hide_index=True,
                use_container_width=True,
            )
        with tab_dac_table:
            dac_display = build_share_table(dac_gen, dac_gen_totals, "Genérica", "DAC_2026", "TOTAL GENERAL")
            st.dataframe(
                dac_display.rename(columns={
                    "DAC_2026": "Monto (S/)",
                }),
                hide_index=True,
                use_container_width=True,
            )

with tab3:
    st.markdown("### Distribución por CP y genéricas complementarias")
    cp_filtered = cp_gen_distribution.copy()
    if gen_filter:
        cp_filtered = cp_filtered[cp_filtered['Genérica'].isin(gen_filter)]

    if cp_filtered.empty:
        st.info("No hay información disponible con los filtros seleccionados.")
    else:
        cp_filtered['Monto'] = cp_filtered['Monto'].round(2)
        cp_filtered.sort_values(['CP', 'Genérica', 'Tipo'], inplace=True)

        cp_sunburst = px.sunburst(
            cp_filtered,
            path=["CP", "Genérica", "Tipo"],
            values="Monto",
            color="Tipo",
            color_discrete_map={"APM": "#d62728", "DAC 2026": "#1f77b4"},
            hover_data={"Monto": ":,.0f"},
        )
        cp_sunburst.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=520)
        st.plotly_chart(cp_sunburst, use_container_width=True)

        cp_summary = cp_filtered.groupby(["CP", "Tipo"], as_index=False)["Monto"].sum()
        cp_stack = px.bar(
            cp_summary,
            x="CP",
            y="Monto",
            color="Tipo",
            barmode="stack",
            hover_data={"Monto":":,.0f"},
            labels={"Monto": "Soles"},
        )
        cp_stack.update_layout(height=520, xaxis_tickangle=-30, margin=dict(l=4, r=8, t=28, b=60))
        st.plotly_chart(cp_stack, use_container_width=True)

        cp_totals = cp_summary.groupby("CP", as_index=False)["Monto"].sum().rename(columns={"Monto": "Total"})
        if not cp_totals.empty:
            cp_totals["Participación %"] = (cp_totals["Total"] / cp_totals["Total"].sum() * 100).round(2)
        cp_totals.sort_values("Total", ascending=False, inplace=True)
        st.dataframe(
            cp_totals.rename(columns={"Total": "Total (S/)"}),
            hide_index=True,
            use_container_width=True,
        )

    if cp_breakdown_quality == "real":
        st.caption("Detalle real por CP y genérica proveniente del archivo cargado.")
    elif cp_breakdown_quality == "mixto":
        st.caption("Detalle mixto: se usa información real disponible y estimaciones proporcionales para el componente faltante.")
    else:
        st.caption("Distribución estimada según la composición actual de genéricas. Sube el Excel para utilizar el detalle real por CP.")

with tab4:
    colA, colB = st.columns(2)
    with colA:
        apm_top = apm_cc.sort_values("Total_APM", ascending=False).head(cc_top_n)
        apm_h = px.bar(apm_top, x="Total_APM", y="CC_T", orientation="h", hover_data={"Total_APM":":,.0f"})
        apm_h.update_layout(height=520, margin=dict(l=4,r=8,t=28,b=20), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(apm_h, use_container_width=True)
        top_share = apm_top['Total_APM'].sum() / apm_cc['Total_APM'].sum() * 100 if apm_cc['Total_APM'].sum() else 0
        st.metric("Participación del Top seleccionado", f"{top_share:.1f}%")
        apm_cc_table = build_share_table(apm_cc, apm_cc_totals, "CC_T", "Total_APM", "TOTAL GENERAL")
        st.dataframe(
            apm_cc_table.rename(columns={
                "Total_APM": "Monto (S/)",
            }),
            hide_index=True,
            use_container_width=True,
        )
    with colB:
        dac_top = dac_cc.sort_values("Total_DAC", ascending=False).head(cc_top_n)
        dac_h = px.bar(dac_top, x="Total_DAC", y="CC_T", orientation="h", hover_data={"Total_DAC":":,.0f"})
        dac_h.update_layout(height=520, margin=dict(l=4,r=8,t=28,b=20), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(dac_h, use_container_width=True)
        dac_top_share = dac_top['Total_DAC'].sum() / dac_cc['Total_DAC'].sum() * 100 if dac_cc['Total_DAC'].sum() else 0
        st.metric("Participación del Top seleccionado", f"{dac_top_share:.1f}%")
        dac_cc_table = build_share_table(dac_cc, dac_cc_totals, "CC_T", "Total_DAC", "TOTAL GENERAL")
        st.dataframe(
            dac_cc_table.rename(columns={
                "Total_DAC": "Monto (S/)",
            }),
            hide_index=True,
            use_container_width=True,
        )

with tab5:
    col_meta_apm, col_meta_dac = st.columns(2)
    with col_meta_apm:
        st.markdown("### Meta y proyección APM 2026")
        apm_meta = st.number_input("Meta anual APM (S/)", min_value=0.0, value=float(apm_total), step=100000.0, key="apm_meta")
        apm_avance = st.number_input("Avance ejecutado a la fecha (S/)", min_value=0.0, value=float(apm_total), step=100000.0, key="apm_avance")
        apm_meses = st.slider("Meses transcurridos", 1, 12, 12, key="apm_meses")
        apm_projection = apm_avance / apm_meses * 12 if apm_meses else 0
        apm_progress = apm_avance / apm_meta * 100 if apm_meta else 0
        apm_delta = None
        if apm_meta:
            delta_value = apm_avance - apm_meta
            if delta_value != 0:
                apm_delta = fmt_money(delta_value)
        gauge_apm = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=apm_avance,
            delta={'reference': apm_meta, 'relative': False},
            gauge={
                'axis': {'range': [0, max(apm_meta, apm_avance, 1) * 1.15]},
                'threshold': {'line': {'color': 'red', 'width': 4}, 'value': apm_meta},
                'bar': {'color': '#d62728'}
            },
            title={'text': 'Avance vs Meta'},
        ))
        gauge_apm.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(gauge_apm, use_container_width=True)
        st.metric("Cumplimiento %", f"{apm_progress:.1f}%", delta=apm_delta)
        st.metric("Proyección anualizada", fmt_money(apm_projection))
        st.caption("La proyección anualizada asume un ritmo constante basado en los meses transcurridos.")

    with col_meta_dac:
        st.markdown("### Meta y proyección DAC 2026")
        dac_meta = st.number_input("Meta anual DAC (S/)", min_value=0.0, value=float(dac_total), step=100000.0, key="dac_meta")
        dac_avance = st.number_input("Avance ejecutado a la fecha (S/)", min_value=0.0, value=float(dac_total), step=100000.0, key="dac_avance")
        dac_meses = st.slider("Meses transcurridos", 1, 12, 12, key="dac_meses")
        dac_projection = dac_avance / dac_meses * 12 if dac_meses else 0
        dac_progress = dac_avance / dac_meta * 100 if dac_meta else 0
        dac_delta = None
        if dac_meta:
            delta_value_dac = dac_avance - dac_meta
            if delta_value_dac != 0:
                dac_delta = fmt_money(delta_value_dac)
        gauge_dac = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=dac_avance,
            delta={'reference': dac_meta, 'relative': False},
            gauge={
                'axis': {'range': [0, max(dac_meta, dac_avance, 1) * 1.15]},
                'threshold': {'line': {'color': 'red', 'width': 4}, 'value': dac_meta},
                'bar': {'color': '#1f77b4'}
            },
            title={'text': 'Avance vs Meta'},
        ))
        gauge_dac.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(gauge_dac, use_container_width=True)
        st.metric("Cumplimiento %", f"{dac_progress:.1f}%", delta=dac_delta)
        st.metric("Proyección anualizada", fmt_money(dac_projection))
        st.caption("Ajusta los controles para simular escenarios de avance y metas.")

    st.markdown("### Simulación continua del avance consolidado APM + DAC")
    total_meta = apm_meta + dac_meta
    total_avance = apm_avance + dac_avance
    sim_month_default = max(apm_meses, dac_meses)
    sim_month = st.slider(
        "Mes considerado en la simulación",
        1,
        12,
        sim_month_default,
        key="sim_meses_total",
    )

    animation_df, simulation_table = build_execution_animation(total_meta, total_avance, sim_month)
    if animation_df.empty:
        st.info("No fue posible generar la simulación con los datos actuales.")
    else:
        y_max = max(
            simulation_table["Proyección acumulada (S/)"].max(),
            simulation_table["Ejecución acumulada (S/)"].max(),
            1.0,
        )
        sim_fig = px.line(
            animation_df,
            x="Mes",
            y="Monto acumulado",
            color="Serie",
            animation_frame="Frame",
            markers=True,
            range_y=[0, y_max * 1.05],
            hover_data={"Monto acumulado": ":,.0f"},
        )
        sim_fig.update_layout(margin=dict(l=4, r=8, t=40, b=10))
        st.plotly_chart(sim_fig, use_container_width=True)

        st.dataframe(simulation_table, use_container_width=True, hide_index=True)
        st.caption(
            "Escenario generado con un ritmo base de 2 a 3 millones mensuales y un repunte en diciembre, ajustado a las metas y avances configurados."
        )


