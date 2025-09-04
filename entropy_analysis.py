# %%
import os
from pathlib import Path
from typing import Literal

from plotly.subplots import make_subplots
import torch
import transformer_lens
from transformer_lens.utils import composition_scores
from transformer_lens import FactoredMatrix, HookedTransformer
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd

import einops as e


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"

# %%


def imshow(array, return_fig=False, xaxis="x", yaxis="y", title=""):
    fig = px.imshow(
        array,
        color_continuous_scale="Viridis",
        aspect="auto",
        labels={"x": xaxis, "y": yaxis},
        title=title,
    )
    if return_fig:
        return fig
    fig.show()


def scatter(x, y, return_fig=False, title="", xaxis="Component", yaxis="Composition Score"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers"))
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis)
    if return_fig:
        return fig
    fig.show()


def make_even(u, s, v):
    return FactoredMatrix(
        u * s.sqrt()[..., None, :],
        s.sqrt()[..., :, None] * transformer_lens.utils.transpose(v),
    )


def re_get_single_component(u, s, v, i):
    news = s.clone()
    newu = u  # .clone()
    newv = v  # .clone()
    news[:i] = 0
    # newu[:, :i] = 0
    # newv[:, :i] = 0
    if i != len(s) - 1:
        news[i + 1 :] = 0
        # newu[:, i+1:] = 0
        # newv[:, i+1:] = 0
    return make_even(newu, news, newv)


# create a heatmap of all heads in all layers
def very_exhaustive_heatmap(layer_weights, is_moeut=True, all_to_all=True):
    if is_moeut:
        n_experts = layer_weights[0]["n_experts"]["v"]
        assert (
            n_experts == layer_weights[0]["n_experts"]["o"]
        ), "Expected n_experts to be the same for v and o"
    else:
        n_experts = 1
    n_heads = layer_weights[0]["n_heads"]
    d_head = layer_weights[0]["d_head"]

    heatmap = np.zeros(
        (len(layer_weights), len(layer_weights), n_heads * n_experts, n_heads, d_head)
    )
    with torch.no_grad():
        for i, layer_from in enumerate(layer_weights):
            for j, layer_to in enumerate(layer_weights):
                if not all_to_all and i >= j:
                    # if not universal transformer, only compute for
                    # layers after layer_from
                    continue
                # compute composition scores from ov in layer i to qk in layer j
                for ov_idx, ov in enumerate(layer_from["ov"]):
                    # ov = FactoredMatrix(ov.A.to("cuda"), ov.B.to("cuda"))
                    src = ov.svd()
                    for qk_idx, qk in enumerate(layer_to["qk"]):
                        print(f"Computing heatmap for {i}.{ov_idx} -> {j}.{qk_idx}")
                        # qk = FactoredMatrix(qk.A.to("cuda"), qk.B.to("cuda"))
                        # ov.A, ov.B = ov.A.to("cuda"), ov.B.to("cuda")
                        # qk.A, qk.B = qk.A.to("cuda"), qk.B.to("cuda")
                        # right = qk.svd()
                        for k in range(layer_from["d_head"]):
                            src_comp = re_get_single_component(*src, k)
                            s = composition_scores(src_comp, qk).item()
                            heatmap[i, j, ov_idx, qk_idx, k] = s
                            # src_comp.svd.cache_clear()
                            del src_comp, s
                        # qk.svd.cache_clear()
                    # ov.svd.cache_clear()
                    del src
                torch.cuda.empty_cache()
                FactoredMatrix.svd.cache_clear()

    return heatmap


def get_weights_and_heatmap_from_path(layer_weights_path, all_to_all=False):
    weights = torch.load(layer_weights_path, weights_only=False, map_location=device)
    # heatmap is saved next to the layer_weights
    heatmap_path = Path(layer_weights_path).with_name("heatmap.pkl")
    if not heatmap_path.exists():
        heatmap = very_exhaustive_heatmap(
            weights, is_moeut="moeut" in layer_weights_path, all_to_all=all_to_all
        )
        with open(heatmap_path, "wb") as f:
            torch.save(heatmap, f)
    else:
        heatmap = torch.load(heatmap_path, weights_only=False)

    return weights, heatmap


# %%
# Find average composition scores for random matrices of the same size
def random_composition_scores(d_embed, d_head, n_runs=10):
    maxes, means = [], []
    for _ in range(n_runs):
        # Generate random matrices
        qk = FactoredMatrix(
            torch.randn(d_embed, d_head, device=device), torch.randn(d_head, d_embed, device=device)
        )
        ov = FactoredMatrix(
            torch.randn(d_embed, d_head, device=device), torch.randn(d_head, d_embed, device=device)
        )
        layer_weights = {
            "qk": [qk],
            "ov": [ov],
            "d_head": d_head,
            "n_heads": 1,
            "n_experts": {"v": 1, "o": 1},
        }
        # Compute composition scores
        heatmap = very_exhaustive_heatmap(
            [layer_weights], is_moeut=True, all_to_all=True
        )  # (1,1,1,1,d_head)
        heatmap = heatmap.squeeze((0, 1, 2, 3))
        _max = heatmap.max().item()
        _mean = heatmap.mean().item()
        maxes.append(_max)
        means.append(_mean)
    return maxes, means


# %%
def compute_entropy(a, axis=-1):
    a = np.array(a) / (np.sum(a, axis=axis, keepdims=True) + 1e-10)

    return -np.sum(a * np.log(a + 1e-10), axis=axis)


def plot_nd_heatmap_grid(
    heatmap: np.ndarray,
    all_to_all: bool = True,
    cmin=None,
    cmax=None,
    layout_dict: dict = None,
    subplot_title_func=None,
    showtext=False
):
    # if heatmap 5d, make subplots
    # otherwise, make one subplot
    if layout_dict is None:
        layout_dict = dict()
    if subplot_title_func is None:
        subplot_title_func = lambda h, i, j: f"{i}.OV -> {j}.QK"  # noqa: E731

    if heatmap.ndim == 4:
        n_layers = heatmap.shape[0]
        assert n_layers == heatmap.shape[1]
        subplot_titles = np.empty((n_layers, n_layers), dtype=object)
        # the following garbage is necessary because plotly indexes subplots
        # starting from origin at top-left, but plotting heatmaps uses bottom-left origin.
        # all transpose-related machinations are to account for this discrepancy
        # smh
        subplot_titles.fill("")
        for i in range(n_layers):
            for j in range(n_layers):
                if all_to_all or i < j:
                    subplot_titles[i, n_layers - j - 1] = subplot_title_func(heatmap, i, j)
        subplot_titles = subplot_titles.flatten(order="F").tolist()

    else:
        assert heatmap.ndim == 2
        assert all_to_all
        n_layers = 1
        heatmap = np.expand_dims(heatmap, axis=(0, 1))
        subplot_titles = [""]

    fig = make_subplots(
        rows=max(n_layers-1, 1),
        cols=n_layers,
        subplot_titles=subplot_titles,
    )

    for from_i in range(n_layers):
        for to_j in range(n_layers):
            if not all_to_all and from_i >= to_j:
                continue
            fig.add_trace(
                go.Heatmap(
                    z=heatmap[from_i, to_j].T,
                    x=np.arange(heatmap.shape[2]),
                    y=np.arange(heatmap.shape[3]),
                    colorscale="Viridis",
                    showscale=False,
                    name=subplot_title_func(heatmap, from_i, to_j),
                    zmin=cmin,
                    zmax=cmax,
                    # hovertext=h_argmax[i, j],
                    texttemplate="%{z:.2f}" if showtext else None,
                    # hovertemplate="<b>OV Head: %{x}</b><br>"
                    # "QK Head: %{y}<br>"
                    # "Composition Score: %{z}<br>"
                    # "Component: %{text}<extra></extra>",
                    # colorbar=dict(
                    #     title="Composition Score",
                    #     # titleside="right",
                    #     len=0.5,
                    #     thickness=10,
                    #     x=1.05,
                    #     y=0.5,
                    # ),
                ),
                row=n_layers - to_j,
                col=from_i + 1,
            )
    fig.update_layout(**layout_dict)
    # , xaxis_title="OV Head", yaxis_title="QK Head")
    fig.data[-1].update(colorbar=dict(x=-0.1, y=0.5, thickness=20), showscale=True)
    return fig


# %%
layer_weights_path_moeut = (
    "analysis_out/dump_slimpajama_moeut_small_matched_rope_noln_long/layer_weights.pth"
)
layer_weights_path_baseline = (
    "analysis_out/dump_slimpajama_baseline_small_rope_long_nodrop_3/layer_weights.pth"
)
layer_weights_path_moeut_g16 = "analysis_out/dump_slimpajama_moeut_small_g16/layer_weights.pth"
layer_weights_path_baseline_20heads = (
    "analysis_out/dump_slimpajama_baseline_small_20heads/layer_weights.pth"
)
# %%

# layer_weights, heatmap = get_weights_and_heatmap_from_path(layer_weights_path_moeut)
layer_weights_path = layer_weights_path_baseline
ALL_TO_ALL = False
layer_weights, heatmap = get_weights_and_heatmap_from_path(
    layer_weights_path, all_to_all=ALL_TO_ALL
)
# IS_UT = True
# layer_weights_baseline, heatmap_baseline = get_weights_and_heatmap_from_path(layer_weights_path_baseline)

# %%
# randoms
d_embed = layer_weights[0]["d_embed"]
d_head = layer_weights[0]["d_head"]
maxes, means = random_composition_scores(d_embed, d_head, n_runs=10)
mean_max = np.mean(maxes)
mean_mean = np.mean(means)
print(f"Mean of max composition score: {mean_max} ± {np.std(maxes)}")
print(f"Mean of mean composition score: {mean_mean} ± {np.std(means)}")


# %%
# Entropy analyses
def normalize_heatmap(heatmap):
    """Normalize heatmap along the last axis (components) to sum to 1."""
    h = heatmap / (np.sum(heatmap, axis=-1, keepdims=True) + 1e-10)
    return h


norm_heatmap = normalize_heatmap(heatmap)

# %%
h_entropy = compute_entropy(norm_heatmap, axis=-1)
heatmap_entropy_grid_fig = plot_nd_heatmap_grid(
    h_entropy,
    all_to_all=ALL_TO_ALL,
    cmin=2,
    cmax=h_entropy.max(),
    layout_dict=dict(
        title="Composition Scores Heatmap Grid (entropy over component comp scores)",
        autosize=False,
        width=1800,
        height=1500,
    ),
)
_save_path = Path(layer_weights_path).with_name("heatmap_entropy.png")
heatmap_entropy_grid_fig.write_image(_save_path, "png", scale=2, width=1800, height=1500)
heatmap_entropy_grid_fig.show()

# %%
def compute_entropy_overlap(heatmap, all_to_all=True):
    # heatmap: (n_layers, n_layers, n_ov, n_qk, n_components)
    # output should be (n_layers, n_layers, n_ov, n_qk, n_ov, n_qk)
    # TODO: vectorize
    n_layers = heatmap.shape[0]
    overlap = -np.empty(
        (n_layers, n_layers, heatmap.shape[2], heatmap.shape[3], heatmap.shape[2], heatmap.shape[3])
    )
    for i in range(n_layers):
        for j in range(n_layers):
            if not all_to_all and i >= j:
                continue
            h = heatmap[i, j]
            n_ov = h.shape[0]
            n_qk = h.shape[1]

            # get pairs of OV-QK heads
            for ov in range(n_ov):
                for qk in range(n_qk):
                    scores = h[ov, qk]
                    # top_indices = np.argsort(scores)[-top_n:][::-1]
                    # top_scores = scores[top_indices]

                    # plot overlap with other OV-QK heads
                    for ov2 in range(n_ov):
                        for qk2 in range(n_qk):
                            scores2 = h[ov2, qk2]
                            o = compute_entropy((scores + scores2) / 2, axis=-1)

                            overlap[i, j, ov, qk, ov2, qk2] = o
    return overlap


pairwise_entropy_overlap = compute_entropy_overlap(norm_heatmap, all_to_all=ALL_TO_ALL)
pairwise_entropy_overlap = e.rearrange(pairwise_entropy_overlap, "i j ov1 qk1 ov2 qk2 -> i j (ov1 qk1) (ov2 qk2)")
entropy_overlap_fig = plot_nd_heatmap_grid(
    pairwise_entropy_overlap,
    all_to_all=ALL_TO_ALL,
    cmin=pairwise_entropy_overlap.min(),
    cmax=pairwise_entropy_overlap.max(),
    subplot_title_func=lambda h, i, j: f"{i}.OV -> {j}.QK (avg: {h[i, j].mean():.2f}↑)",
    layout_dict=dict(
        title="Component Weighting Overlap Heatmap (entropy over component composition scores). "
        "For a given OV-QK pairing, to what extent do different heads pay "
        "attention to the same components?",
        autosize=False,
        width=3200,
        height=1500,
        font=dict(size=12),  # Reduce overall font size including subplot titles
    ),
)
_save_path = Path(layer_weights_path).with_name("entropy_component_overlap.jpg")
# entropy_overlap_fig.write_image(_save_path, "jpg", scale=1, width=3200, height=1500)
entropy_overlap_fig.show()

# %%
# avg_layer_entropy_fig = plot_average_entropy(norm_heatmap)
# avg_layer_entropy_fig.update_layout(autosize=False, width=800, height=600)
avg_layer_entropy = compute_entropy(norm_heatmap.mean(axis=(2, 3)), axis=-1) # average over OV and QK heads

avg_layer_entropy_fig = plot_nd_heatmap_grid(
    avg_layer_entropy,
    all_to_all=True,
    cmin=2,
    cmax=5,
    layout_dict=dict(
        title="Average Entropy of Component Weightings Between All OV-QK Pairs in Each Layer Pair (↑)",
        autosize=False,
        width=1000,
        height=1000,
        font=dict(size=12),  # Reduce overall font size including subplot titles
    ),
    showtext=True
)
_save_path = Path(layer_weights_path).with_name("avg_layer_entropy.jpg")
avg_layer_entropy_fig.write_image(_save_path, "jpg", scale=1)
avg_layer_entropy_fig.show()

