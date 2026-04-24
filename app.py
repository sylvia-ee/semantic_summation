"""
    pixi run streamlit run app.py
"""

import glob
import os
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from graph_builder import build_mcdi_info, build_word_graph, hierarchical_layout
from preprocessing.childes_preprocessing import count_words_in_childes
from preprocessing.mcdi_ibi_preprocessing import (
    create_alt_form_dict,
    exclude_cats,
    exclude_proper_nouns,
    exclude_words,
    grammatically_generated_inclusions,
    manual_inclusions,
    mcdi_ibi_setup,
    strip_syntax,
)
from preprocessing.parse_raw_data_folder import load_sample_dfs, process_data_folder
from preprocessing.childes_preprocessing import childes_cleaner

#_______________
# PATH
#_______________
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
MANUAL_DIR = os.path.join(DATA_DIR, "manual_preprocessing")

#_______________
# COLOR PALETTE
#_______________
GRAMM_COLORS = {
    "singular_noun":            "#5B9BD5",   # blue
    "plural_noun":              "#70AD47",   # green
    "singular_possessive_noun": "#ED7D31",   # orange
    "plural_possessive_noun":   "#BA8CC0",   # lavender
}
BG_COLOR = "#1a1a2e"


def discover_manual_files(manual_dir):
    return {
        "category_exclusions": sorted(glob.glob(os.path.join(manual_dir, "category-exclusions_*.csv"))),
        "word_exclusions":      sorted(glob.glob(os.path.join(manual_dir, "word-exclusions_*.csv"))),
        "word_inclusions":      sorted(glob.glob(os.path.join(manual_dir, "word-inclusions_*.csv"))),
    }


@st.cache_data(show_spinner="Running preprocessing pipeline…")
def load_and_preprocess(cat_excl_paths: tuple, word_excl_paths: tuple, word_incl_paths: tuple):

    paths_df = process_data_folder(DATA_DIR)
    dfs_dict = load_sample_dfs(paths_df)

    sample_name = list(dfs_dict.keys())[0]
    sample_dfs = dfs_dict[sample_name]

    mcdi_ibi = sample_dfs["mcdi_ibi_df"]
    mcdi_ibi = mcdi_ibi_setup(mcdi_ibi)

    for path in cat_excl_paths:
        mcdi_ibi = exclude_cats(mcdi_ibi, path)

    mcdi_ibi = exclude_words(
        mcdi_ibi,
        exclusion_funcs=[exclude_proper_nouns],
        csv_paths=list(word_excl_paths),
    )
    mcdi_ibi = strip_syntax(mcdi_ibi)

    mcdi_active = mcdi_ibi[
        mcdi_ibi["excl_reason"].isna()
        & ~mcdi_ibi["alt"].str.contains(" ", na=False)
    ].copy()

    alt_forms_dict = create_alt_form_dict(mcdi_active)
    for path in word_incl_paths:
        alt_forms_dict = manual_inclusions(alt_forms_dict, path)
    alt_forms_dict = grammatically_generated_inclusions(alt_forms_dict)

    mcdi_info = build_mcdi_info(mcdi_active)

    childes_df = sample_dfs["childes_df"]
    childes_transcripts_dict = childes_cleaner(childes_df, language_column="gloss", identifier_col="id")
    childes_counts = count_words_in_childes(childes_df, language_column="gloss")

    return alt_forms_dict, childes_transcripts_dict, childes_counts, mcdi_info, mcdi_active


def _scatter_layout(x_title, x_range_log, y_range):
    return dict(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font_color="white",
        xaxis=dict(
            type="log",
            showgrid=False, zeroline=False,
            title=x_title, title_font=dict(size=11),
            range=x_range_log,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            title="Production score", title_font=dict(size=11),
            range=y_range,
        ),
        margin=dict(l=50, r=20, t=20, b=50),
        height=320,
        showlegend=False,
    )


def _linreg_trace_log(xs, ys):
    """Fit y ~ log10(x) and return a Scatter trace for a log-x plot."""
    xs_arr, ys_arr = np.array(xs, dtype=float), np.array(ys, dtype=float)
    mask = xs_arr > 0
    if mask.sum() < 2:
        return None
    m, b = np.polyfit(np.log10(xs_arr[mask]), ys_arr[mask], 1)
    x_line = np.geomspace(xs_arr[mask].min(), xs_arr[mask].max(), 100)
    y_line = m * np.log10(x_line) + b
    return go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color="#99c4e8", width=1.5, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
    )


def _linreg_stats(xs, ys):
    xs_arr, ys_arr = np.array(xs, dtype=float), np.array(ys, dtype=float)
    mask = xs_arr > 0
    x_log = np.log10(xs_arr[mask])
    y = ys_arr[mask]
    m, b = np.polyfit(x_log, y, 1)
    y_hat = m * x_log + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return dict(slope=round(m, 4), intercept=round(b, 4), r2=round(r2, 4))


def build_scatter_r_count(all_bases, mcdi_info, childes_counts, x_range_log, y_range, highlight=None):
    xs, ys, texts = [], [], []
    hx, hy = None, None

    for w in all_bases:
        prod = mcdi_info.get(w, {}).get("prod")
        if prod is None:
            continue

        x = childes_counts.get(w, 0)
        y = prod

        if highlight and w == highlight:
            hx, hy = x, y
        else:
            xs.append(x)
            ys.append(y)
            texts.append(w)

    traces = [
        go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(size=5, color="#5B9BD5", opacity=0.5),
            text=texts,
            hovertemplate="<b>%{text}</b><br>prod: %{y:.2f}<br>r_count: %{x:,}<extra></extra>",
        )
    ]

    # highlighted point (RED + bigger)
    if hx is not None:
        traces.append(
            go.Scatter(
                x=[hx], y=[hy],
                mode="markers+text",
                marker=dict(size=12, color="red"),
                text=[highlight],
                textposition="top center",
                name="highlight",
            )
        )

    reg = _linreg_trace_log(xs, ys)
    if reg:
        traces.append(reg)

    fig = go.Figure(traces)
    fig.update_layout(**_scatter_layout("log CHILDES count (base form)", x_range_log, y_range))
    return fig


def build_scatter_c_count(
    all_bases,
    mcdi_info,
    alt_forms_dict,
    childes_counts,
    x_range_log,
    y_range,
    highlight=None,
):
    xs, ys, texts = [], [], []
    hx, hy = None, None

    for w in all_bases:
        prod = mcdi_info.get(w, {}).get("prod")
        if prod is None:
            continue

        c = sum(childes_counts.get(alt, 0) for alt in alt_forms_dict.get(w, {}))

        # skip zeros for log-scale plotting
        if c <= 0:
            continue

        # check highlight (case-insensitive)
        if highlight and w.lower() == highlight.lower():
            hx, hy = c, prod
        else:
            xs.append(c)
            ys.append(prod)
            texts.append(w)

    traces = [
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=5, color="#70AD47", opacity=0.5),
            text=texts,
            hovertemplate="<b>%{text}</b><br>prod: %{y:.2f}<br>c_count: %{x:,}<extra></extra>",
        )
    ]

    # highlighted point (red, larger, labeled)
    if hx is not None:
        traces.append(
            go.Scatter(
                x=[hx],
                y=[hy],
                mode="markers+text",
                marker=dict(size=12, color="red"),
                text=[highlight],
                textposition="top center",
                name="highlight",
                hovertemplate="<b>%{text}</b><br>prod: %{y:.2f}<br>c_count: %{x:,}<extra></extra>",
            )
        )

    # regression line (only on non-highlighted points to avoid bias)
    reg = _linreg_trace_log(xs, ys)
    if reg:
        traces.append(reg)

    fig = go.Figure(traces)
    fig.update_layout(
        **_scatter_layout("log CHILDES count (all forms)", x_range_log, y_range)
    )
    return fig

def draw_graph_plotly(G: nx.DiGraph, base: str) -> go.Figure | None:
    if len(G.nodes) == 0:
        return None

    pos = hierarchical_layout(G, base)

    r_counts_all = [G.nodes[n].get("r_count", 0) for n in G.nodes]
    max_count = max(r_counts_all) if any(c > 0 for c in r_counts_all) else 1

    edge_x, edge_y = [], []
    annotations = []

    edge_transforms: dict[tuple, list[str]] = defaultdict(list)
    for u, v, data in G.edges(data=True):
        edge_transforms[(u, v)].append(data.get("transform", ""))

    for (u, v), transforms in edge_transforms.items():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        label = " / ".join(sorted(set(t for t in transforms if t)))
        annotations.append(dict(
            x=mx, y=my,
            text=label,
            showarrow=False,
            font=dict(size=9, color="black"),
            bgcolor="white",
            borderpad=2,
            opacity=0.9,
        ))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#777777", width=1.5),
        hoverinfo="none",
        showlegend=False,
    )

    gramm_to_nodes = defaultdict(list)
    for node in G.nodes:
        gramm_to_nodes[G.nodes[node].get("gramm") or "unknown"].append(node)

    node_traces = []
    for gramm, group in gramm_to_nodes.items():
        xs = [pos[n][0] for n in group]
        ys = [pos[n][1] for n in group]
        sizes = [20 + 45 * (G.nodes[n].get("r_count", 0) / max_count) for n in group]

        customdata = [
            [
                (G.nodes[n].get("gramm") or "").replace("_", " "),
                G.nodes[n].get("category", ""),
                G.nodes[n].get("r_count", 0),
                f"{G.nodes[n].get('prod'):.2f}" if G.nodes[n].get("prod") is not None else "N/A",
            ]
            for n in group
        ]

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=GRAMM_COLORS.get(gramm, "#888888"),
                line=dict(width=1.5, color="white"),
            ),
            text=group,
            textposition="top center",
            textfont=dict(color="white", size=10),
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "gramm: %{customdata[0]}<br>"
                "category: %{customdata[1]}<br>"
                "CHILDES count: %{customdata[2]:,}<br>"
                "production: %{customdata[3]}"
                "<extra></extra>"
            ),
            name=gramm.replace("_", " "),
            legendgroup=gramm,
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        annotations=annotations,
        showlegend=True,
        legend=dict(
            font=dict(color="white"),
            bgcolor="#2a2a4a",
            bordercolor="#555555",
            borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_color="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=30, b=20),
        height=560,
    )
    return fig


def build_sankey(G: nx.DiGraph, base: str) -> go.Figure | None:
    nodes = list(G.nodes)
    if not nodes:
        return None

    node_idx = {n: i for i, n in enumerate(nodes)}
    sources, targets, values, link_labels = [], [], [], []

    edge_map: dict[tuple, list[str]] = defaultdict(list)
    for u, v, data in G.edges(data=True):
        edge_map[(u, v)].append(data.get("transform", ""))

    for (u, v), transforms in edge_map.items():
        sources.append(node_idx[u])
        targets.append(node_idx[v])
        values.append(max(G.nodes[v].get("r_count", 0), 1))
        link_labels.append(" / ".join(sorted(set(t for t in transforms if t))))

    if not sources:
        return None

    node_colors = [
        GRAMM_COLORS.get(G.nodes[n].get("gramm") or "singular_noun", "#888888")
        for n in nodes
    ]

    node_customdata = [
        [
            (G.nodes[n].get("gramm") or "").replace("_", " "),
            G.nodes[n].get("category", ""),
            G.nodes[n].get("r_count", 0),
            G.nodes[n].get("prod"),
        ]
        for n in nodes
    ]
    node_hovertemplate = (
        "<b>%{label}</b><br>"
        "gramm: %{customdata[0]}<br>"
        "category: %{customdata[1]}<br>"
        "CHILDES count: %{customdata[2]:,}<br>"
        "production: %{customdata[3]:.2f}"
        "<extra></extra>"
    )

    link_customdata = [
        [link_labels[i], nodes[sources[i]], nodes[targets[i]]]
        for i in range(len(sources))
    ]
    link_hovertemplate = (
        "transform: <b>%{customdata[0]}</b><br>"
        "%{customdata[1]} → %{customdata[2]}<br>"
        "CHILDES count: %{value:,}"
        "<extra></extra>"
    )

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15, thickness=20,
            label=nodes,
            color=node_colors,
            customdata=node_customdata,
            hovertemplate=node_hovertemplate,
        ),
        link=dict(
            source=sources, target=targets, value=values,
            label=link_labels,
            color="rgba(160,160,160,0.3)",
            customdata=link_customdata,
            hovertemplate=link_hovertemplate,
        ),
    ))
    fig.update_layout(
        title_text=f"Transform flow for '{base}' (link width = CHILDES count)",
        font_size=12,
        paper_bgcolor=BG_COLOR,
        font_color="white",
    )
    return fig


def node_attributes_table(G: nx.DiGraph) -> pd.DataFrame:
    rows = [
        {
            "word": n,
            "gramm": attrs.get("gramm", ""),
            "category": attrs.get("category", ""),
            "r_count": attrs.get("r_count", 0),
            "prod": attrs.get("prod"),
            "is_base": attrs.get("is_base", False),
        }
        for n, attrs in G.nodes(data=True)
    ]
    return pd.DataFrame(rows).sort_values("r_count", ascending=False).reset_index(drop=True)

st.set_page_config(page_title="Word Graphs", layout="wide")
st.title("Counts and Forms for CHILDES")
st.caption("American-English Corpus from 0-24 months with MCDI production data")

available = discover_manual_files(MANUAL_DIR)

def _fnames(paths):
    return [os.path.basename(p) for p in paths]

def _paths_for(names, all_paths):
    name_to_path = {os.path.basename(p): p for p in all_paths}
    return [name_to_path[n] for n in names if n in name_to_path]

with st.sidebar:
    with st.expander("Preprocessing files", expanded=False):
        sel_cat_names = st.multiselect(
            "Category exclusions",
            options=_fnames(available["category_exclusions"]),
            default=_fnames(available["category_exclusions"]),
        )
        sel_word_excl_names = st.multiselect(
            "Word exclusions",
            options=_fnames(available["word_exclusions"]),
            default=_fnames(available["word_exclusions"]),
        )
        sel_word_incl_names = st.multiselect(
            "Word inclusions",
            options=_fnames(available["word_inclusions"]),
            default=_fnames(available["word_inclusions"]),
        )

cat_excl_paths  = tuple(_paths_for(sel_cat_names,       available["category_exclusions"]))
word_excl_paths = tuple(_paths_for(sel_word_excl_names, available["word_exclusions"]))
word_incl_paths = tuple(_paths_for(sel_word_incl_names, available["word_inclusions"]))

alt_forms_dict, childes_transcripts_dict, childes_counts, mcdi_info, mcdi_active = load_and_preprocess(
    cat_excl_paths, word_excl_paths, word_incl_paths
)

all_bases = sorted(alt_forms_dict.keys())

prods = [v["prod"] for v in mcdi_info.values() if v.get("prod") is not None]
prod_min = float(min(prods)) if prods else 0.0
prod_max = float(max(prods)) if prods else 1.0

if "_prod_lo" not in st.session_state:
    st.session_state["_prod_lo"] = prod_min
if "_prod_hi" not in st.session_state:
    st.session_state["_prod_hi"] = prod_max

with st.sidebar:
    st.header("Search & Filters")

    all_categories = sorted({
        info.get("category", "unknown") for info in mcdi_info.values()
    })
    selected_categories = st.multiselect("Category", all_categories, default=all_categories)

    st.markdown("**Production score**")
    col_lo, col_hi = st.columns(2)
    with col_lo:
        st.session_state["_prod_lo"] = st.number_input(
            "Min", min_value=prod_min, max_value=prod_max,
            value=st.session_state["_prod_lo"], step=0.01, format="%.2f",
        )
    with col_hi:
        st.session_state["_prod_hi"] = st.number_input(
            "Max", min_value=prod_min, max_value=prod_max,
            value=st.session_state["_prod_hi"], step=0.01, format="%.2f",
        )

    prod_range = (st.session_state["_prod_lo"], st.session_state["_prod_hi"])
    exclude_zero = st.checkbox("Exclude zero-count forms", value=False)

    st.divider()
    st.caption(f"Base words loaded: **{len(alt_forms_dict)}**")
    st.caption(f"CHILDES total tokens: **{sum(childes_counts.values()):,}**")


_all_r_xs, _all_c_xs, _all_ys = [], [], []
for _w in all_bases:
    _prod = mcdi_info.get(_w, {}).get("prod")
    if _prod is None:
        continue
    _all_r_xs.append(childes_counts.get(_w, 0))
    _all_c_xs.append(sum(childes_counts.get(_a, 0) for _a in alt_forms_dict.get(_w, {})))
    _all_ys.append(_prod)

_nz = [x for x in _all_r_xs + _all_c_xs if x > 0]
_x_range_log = [np.log10(min(_nz)) - 0.15, np.log10(max(_nz)) + 0.15] if _nz else [0, 4]
_y_pad = (max(_all_ys or [1]) - min(_all_ys or [0])) * 0.05
_y_range = [min(_all_ys or [0]) - _y_pad, max(_all_ys or [1]) + _y_pad]


_valid = [(r, c, y) for r, c, y in zip(_all_r_xs, _all_c_xs, _all_ys) if r > 0]
_r_valid = [v[0] for v in _valid]
_c_valid = [v[1] for v in _valid]
_y_valid = [v[2] for v in _valid]
_sr = _linreg_stats(_r_valid, _y_valid)
_sc = _linreg_stats(_c_valid, _y_valid)

highlight_word = None

col_scatter_l, col_scatter_r = st.columns(2)

with col_scatter_l:
    st.markdown("**Production vs base-form count**")
    fig_r = build_scatter_r_count(
        all_bases, mcdi_info, childes_counts,
        _x_range_log, _y_range,
        highlight=st.session_state.get("highlight_word")
    )
    st.plotly_chart(fig_r, use_container_width=True)
    _sr = _linreg_stats(_all_r_xs, _all_ys)
    if _sr:
        _a, _b, _c, _d = st.columns(4)
        _a.caption(f"slope `{_sr['slope']:+.4f}`")
        _b.caption(f"intercept `{_sr['intercept']:.4f}`")
        _c.caption(f"R² `{_sr['r2']:.4f}`")

with col_scatter_r:
    st.markdown("**Production vs cumulative count**")
    fig_c = build_scatter_c_count(
    all_bases, mcdi_info, alt_forms_dict, childes_counts,
    _x_range_log, _y_range,
    highlight=st.session_state.get("highlight_word")
    )
    st.plotly_chart(fig_c, use_container_width=True)
    _sc = _linreg_stats(_all_c_xs, _all_ys)
    if _sc:
        _a, _b, _c, _d = st.columns(4)
        _a.caption(f"slope `{_sc['slope']:+.4f}`")
        _b.caption(f"intercept `{_sc['intercept']:.4f}`")
        _c.caption(f"R² `{_sc['r2']:.4f}`")


filtered_bases = [
    w for w in all_bases
    if mcdi_info.get(w, {}).get("category", "unknown") in selected_categories
    and (
        mcdi_info.get(w, {}).get("prod") is None
        or prod_range[0] <= mcdi_info.get(w, {}).get("prod", 0) <= prod_range[1]
    )
    and (not exclude_zero or childes_counts.get(w, 0) > 0)
]

st.subheader(f"All MCDI base words ({len(filtered_bases)} shown)")



def _word_row(w):
    r = childes_counts.get(w, 0)
    c = sum(childes_counts.get(alt, 0) for alt in alt_forms_dict.get(w, {}))
    log_dist = round(np.log1p(c) - np.log1p(r), 3)
    return {
        "word": w,
        "category": mcdi_info.get(w, {}).get("category", ""),
        "prod": mcdi_info.get(w, {}).get("prod"),
        "alt_forms": sum(1 for f in alt_forms_dict.get(w, {}) if childes_counts.get(f, 0) > 0),
        "r_count": r,
        "c_count": c,
        "log_distance": log_dist,
    }

word_table = pd.DataFrame([_word_row(w) for w in filtered_bases])

selection = st.dataframe(
    word_table,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

selected_base = None
if selection.selection.rows:
    selected_base = word_table.iloc[selection.selection.rows[0]]["word"]

highlight_word = None
if selection.selection.rows:
    highlight_word = word_table.iloc[selection.selection.rows[0]]["word"]

if selected_base:
    G = build_word_graph(
        selected_base, alt_forms_dict, childes_counts, mcdi_info,
        skip_compounds=True,
    )

    if exclude_zero:
        zero_nodes = [
            n for n in G.nodes
            if G.nodes[n].get("r_count", 0) == 0 and not G.nodes[n].get("is_base")
        ]
        G.remove_nodes_from(zero_nodes)

    st.subheader(f"**{selected_base}**")

    c_count = sum(G.nodes[n].get("r_count", 0) for n in G.nodes)

    info = mcdi_info.get(selected_base, {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Category", info.get("category", "N/A"))
    prod_val = info.get("prod")
    c2.metric("Production score", f"{prod_val:.2f}" if prod_val is not None else "N/A")
    c3.metric("Alt forms in graph", len(G.nodes) - 1)
    c4.metric("CHILDES count", f"{c_count:,}")

    tab_graph, tab_sankey, tab_table = st.tabs(
        ["Transformation Graph", "Sankey Diagram", "Node Attributes"]
    )

    with tab_graph:
        fig = draw_graph_plotly(G, selected_base)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No graph data for this word.")

    with tab_sankey:
        sankey = build_sankey(G, selected_base)
        if sankey:
            st.plotly_chart(sankey, use_container_width=True)
        else:
            st.info("No edges to display in Sankey.")

    with tab_table:
        attr_df = node_attributes_table(G)
        st.dataframe(attr_df, use_container_width=True, hide_index=True)
