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

# %%

# # %%
# model = HookedTransformer.from_pretrained(
#     "gpt2-small", fold_value_biases=True, refactor_factored_attn_matrices=True
# )

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


# def get_qk(l, h):
#     return model.blocks[l].attn.QK[h]


def simple_heatmap(dest_layer, right):
    # only look at the layers and heads, not all components
    heatmap = np.zeros((dest_layer, n_heads))
    for layer in range(dest_layer):
        print(layer)
        for head in range(n_heads):
            src = model.blocks[layer].attn.OV[head]
            scores = []
            # src = re_get_single_component(*src_usv, comp_idx)
            scores.append(composition_scores(src, right).item())
            heatmap[layer, head] = max(scores)
    return heatmap


# create a heatmap of all heads in all layers for a given destination layer and matrix
def exhaustive_heatmap(dest_layer, right):
    heatmap = np.zeros((dest_layer, n_heads))
    for layer in range(dest_layer):
        print(layer)
        for head in range(n_heads):
            src_usv = model.blocks[layer].attn.OV[head].svd()
            scores = []
            for i in range(d_head):
                src = re_get_single_component(*src_usv, i)
                scores.append(composition_scores(src, right).item())
            heatmap[layer, head] = max(scores)
    return heatmap


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
        heatmap = very_exhaustive_heatmap(weights, is_moeut="moeut" in layer_weights_path, all_to_all=all_to_all)
        with open(heatmap_path, "wb") as f:
            torch.save(heatmap, f)
    else:
        heatmap = torch.load(heatmap_path, weights_only=False)

    return weights, heatmap


# # %%

# heatmap.shape  # (n_layers, n_layers, n_heads*n_experts, n_heads, d_head)

# imshow(
#     heatmap.max(axis=-1)[0, 1],
#     return_fig=True,
#     xaxis="OV Head",
#     yaxis="QK Head",
#     title="Composition Scores from 0.OV to 1.QK (max over components)",
# )


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
        heatmap = very_exhaustive_heatmap([layer_weights], is_moeut=True, all_to_all=True) # (1,1,1,1,d_head)
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

# plot on a 2d grid of heatmaps, x=layer from, y=layer to
def plot_heatmap_grid(heatmap, all_to_all=True, subtract=None, func: Literal["max", "mean", "entropy"]="max"):
    n_layers = heatmap.shape[0]
    pairs = [(i, j) for i in range(n_layers) for j in range(n_layers)]
    fig = make_subplots(
        rows=n_layers,
        cols=n_layers,
        subplot_titles=[f"{i}.OV -> {j}.QK" if (all_to_all or i < j) else "" for i, j in pairs],
    )

    if func == "mean":
        h = heatmap.mean(axis=-1)
    elif func == "max":
        h = heatmap.max(axis=-1)
    elif func == "entropy":
        h = compute_entropy(heatmap, axis=-1)

    if subtract is not None:
        h = h - subtract
    h_argmax = heatmap.argmax(axis=-1)
    cmin = h.min()
    cmax = h.max()

    for i in range(n_layers):
        for j in range(n_layers):
            if not all_to_all and i >= j:
                continue
            fig.add_trace(
                go.Heatmap(
                    z=h[i, j],
                    x=np.arange(heatmap.shape[2]),
                    y=np.arange(heatmap.shape[3]),
                    colorscale="Viridis",
                    showscale=False,
                    name=f"{i}.OV -> {j}.QK",
                    zmin=cmin,
                    zmax=cmax,
                    hovertext=h_argmax[i, j],
                    # texttemplate="%{text}",
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
                row=i + 1,
                col=j + 1,
            )
    fig.update_layout(title="Composition Scores Heatmap Grid (max over components)" + " Subtracted Mean Max" if subtract is not None else "")
    # , xaxis_title="OV Head", yaxis_title="QK Head")

    fig.data[-1].update(colorbar=dict(x=1.05, y=0.5, thickness=20), showscale=True)
    return fig


# %%
# for a single layer, plot comp scores for all components by head
def plot_composition_scores_by_component(heatmap, layer_from, layer_to, hline: float = None):
    h = heatmap[layer_from, layer_to]
    n_ov = h.shape[0]
    n_qk = h.shape[1]

    # subplot by qk, plot component scores for all ov heads colored by ov
    # subplots = n_ov * n_qk
    fig = make_subplots(
        rows=n_qk,#n_ov,
        cols=1,
        # subplot_titles=[f"{layer_from}.OV.{ov} -> {layer_to}.QK.{qk}" for ov in range(n_ov) for qk in range(n_qk)],
        subplot_titles=[f"Composition scores from {layer_from}.OV.x to {layer_to}.QK.{qk}" for qk in range(n_qk)],
    )
    for ov in range(n_ov):
        for qk in range(n_qk):
            scores = h[ov, qk]
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(scores)),
                    y=scores,
                    mode="markers",
                    name=f".{ov} -> .{qk}",
                    # color by ov
                    marker=dict(
                        color=px.colors.qualitative.Alphabet[ov % len(px.colors.qualitative.Alphabet)],
                    )
                    # marker=dict(color=f"rgba({ov * 50}, {qk * 50}, 150, 0.8)"),
                ),
                col=1,
                row=qk + 1,
            )
    fig.update_layout(
        title=f"Composition Scores from {layer_from}.OV to {layer_to}.QK (by component)",
        xaxis_title="Component",
        yaxis_title="Composition Score",
    )
    if hline is not None:
        fig.add_hline(y=hline, line_color="red", line_dash="dash", annotation_text="Random", annotation_position="top left")

    # fig.update_xaxes(tickvals=np.arange(len(scores)), ticktext=[f"Component {i}" for i in range(len(scores))])
    # fig.update_yaxes(range=[0, 1], dtick=0.1)
    return fig


# # %%
# # For each OV-QK pair, plot percentage of components that have a composition score above threshold
# def plot_percentage_above_threshold(heatmap, layer_from, layer_to, threshold=0):
#     h = heatmap[layer_from, layer_to]
#     n_ov = h.shape[0]
#     n_qk = h.shape[1]

#     percentages = []
#     for ov in range(n_ov):
#         for qk in range(n_qk):
#             scores = h[ov, qk]
#             percentage = (scores > threshold).sum() / len(scores) * 100
#             percentages.append((str(ov), str(qk), percentage))

#     df = pd.DataFrame(percentages, columns=["OV Head", "QK Head", "Percentage"])
#     fig = px.bar(
#         df,
#         x="OV Head",
#         y="Percentage",
#         color="QK Head",
#         title=f"Percentage of Components with Composition Score > {threshold} from {layer_from}.OV to {layer_to}.QK",
#         labels={"OV Head": "OV Head", "Percentage": "Percentage (%)"},
#         barmode="group",  # Side by side bars
#     )
#     print(f"Mean percentage of components above threshold {threshold}: {df['Percentage'].mean():.2f}%")
#     return fig


# %%
# For all OV-QK pairs, plot the indices of top-3 components by composition score.
# For each subplot: x-axis is QK-OV pair, y-axis is component index, color is composition score.
def plot_top_components(heatmap, top_n=3, all_to_all=True):
    n_layers = heatmap.shape[0]
    pairs = [(i, j) for i in range(n_layers) for j in range(n_layers)]
    fig = make_subplots(
        rows=n_layers,
        cols=n_layers,
        subplot_titles=[f"{i}.OV -> {j}.QK" if (all_to_all or i < j) else "" for i, j in pairs],
    )
    for i in range(n_layers):
        for j in range(n_layers):
            if not all_to_all and i >= j:
                continue
            h = heatmap[i, j]
            n_ov = h.shape[0]
            n_qk = h.shape[1]

            # For each OV-QK pair, get top_n components by composition score
            for ov in range(n_ov):
                for qk in range(n_qk):
                    scores = h[ov, qk]
                    top_indices = np.argsort(scores)[-top_n:][::-1]
                    top_scores = scores[top_indices]

                    fig.add_trace(
                        go.Scatter(
                            x=[f"{i}.OV.{ov} -> {j}.QK.{qk}"] * top_n,
                            y=top_indices,
                            mode="markers",
                            text=top_scores,
                            textposition="top center",
                            name=f"{i}.{ov} -> {j}.{qk}",
                            marker=dict(
                                color=top_scores,
                                colorscale="Viridis",
                                size=10,
                                showscale=False,
                            ),
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
    fig.update_layout(
        title=f"Top {top_n} Components by Composition Score (max over components)",
        xaxis_title="OV-QK Pair",
        yaxis_title="Component Index",
    )
    return fig

# %%
# Plot percentage of components that have a composition score above threshold for each OV-QK pair
def plot_percentage_above_threshold(heatmap, threshold=0, all_to_all=True):
    n_layers = heatmap.shape[0]
    pairs = [(i, j) for i in range(n_layers) for j in range(n_layers)]
    fig = make_subplots(
        rows=n_layers,
        cols=n_layers,
        subplot_titles=[f"{i}.OV -> {j}.QK" if (all_to_all or i < j) else "" for i, j in pairs],
    )
    for i in range(n_layers):
        for j in range(n_layers):
            if not all_to_all and i >= j:
                continue
            h = heatmap[i, j]
            n_ov = h.shape[0]
            n_qk = h.shape[1]

            # For each OV-QK pair, get top_n components by composition score
            p = np.zeros((n_ov, n_qk))
            for ov in range(n_ov):
                for qk in range(n_qk):
                    scores = h[ov, qk]
                    percentage = (scores > threshold).sum() / len(scores) * 100
                    p[ov, qk] = percentage
            fig.add_trace(
                go.Heatmap(
                    z=p,
                    colorscale="Viridis",
                    zmin=0,
                    zmax=100,
                    showscale=False,
                    # colorbar=dict(title="Percentage (%)"),
                ),
                row=i + 1,
                col=j + 1,
            )
    fig.data[-1].update(colorbar=dict(x=1.05, y=0.5, thickness=20), showscale=True)
    fig.update_layout(
        title=f"Percentage of Components with Composition Score > {threshold}",
        xaxis_title="QK Head",
        yaxis_title="OV Head",
    )
    return fig
# %%

def kl_divergence(p, q):
    """Compute KL divergence between two probability distributions."""
    p = np.array(p)
    q = np.array(q)
    # Avoid division by zero and log(0) by adding a small constant
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return np.sum(p * np.log(p / q))
# def overlap(a, b):
#     """Compute cosine similarity between two vectors."""
#     a = np.array(a)
#     b = np.array(b)
#     dot_product = np.dot(a, b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     if norm_a == 0 or norm_b == 0:
#         return 0.0  # Avoid division by zero
#     return dot_product / (norm_a * norm_b)
# def compute_overlap(a, b):
#     """Norm a and b and compute kl divergence between them."""
#     a = np.array(a)
#     b = np.array(b)
#     # a = np.exp(a) / np.sum(np.exp(a))
#     # b = np.exp(b) / np.sum(np.exp(b))
#     a = a / np.sum(a)
#     b = b / np.sum(b)
#     return kl_divergence(a, b)
def compute_overlap(a, b, top_n=3):
    a = np.array(a)
    b = np.array(b)
    top_a_indices = np.argsort(a)[-top_n:][::-1]
    # top_a_scores = a[top_a_indices]
    top_b_indices = np.argsort(b)[-top_n:][::-1]
    # top_b_scores = b[top_b_indices]

    # return IOU
    intersection = len(set(top_a_indices) & set(top_b_indices))
    union = len(set(top_a_indices) | set(top_b_indices))
    return intersection / union if union > 0 else 0


# %%
# For a given pair of layers, take pairs of OV-QK heads. Plot overlap between (top) components
def plot_exhaustive_component_overlap(heatmap, all_to_all=True, overlap_metric_fn=None, want_lower=True):
    n_layers = heatmap.shape[0]

    overlap = -np.empty((n_layers, n_layers, heatmap.shape[2], heatmap.shape[3], heatmap.shape[2], heatmap.shape[3]))
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
                            o = overlap_metric_fn(scores, scores2)

                            overlap[i, j, ov, qk, ov2, qk2] = o

    pairs = [(i, j) for i in range(n_layers) for j in range(n_layers)]
    sign = "↓" if want_lower else "↑"
    fig = make_subplots(
        rows=n_layers,
        cols=n_layers,
        subplot_titles=[f"{i}.OV -> {j}.QK (avg: {overlap[i, j].mean():.2f}{sign})" if (all_to_all or i < j) else "" for i, j in pairs],
    )
    for i in range(n_layers):
        for j in range(n_layers):
            if not all_to_all and i >= j:
                continue
            # create a heatmap of overlaps
            cmin = overlap.min()
            cmax = overlap.max()
            o = overlap[i, j]
            n_ov = o.shape[0]
            n_qk = o.shape[1]

            o = o.reshape(n_ov * n_qk, n_ov * n_qk)

            fig.add_trace(
                go.Heatmap(
                    z=o,
                    x=np.arange(o.shape[0]),
                    y=np.arange(o.shape[1]),
                    colorscale="Viridis",
                    showscale=False,
                    # name=f"{i}.OV -> {j}.QK (average: {o.mean():.2f}↑)",
                    zmin=cmin,
                    zmax=cmax,
                ),
                row=i + 1,
                col=j + 1,
            )

    fig.data[-1].update(colorbar=dict(x=1.05, y=0.5, thickness=20), showscale=True)

    fig.update_layout(
        title=f"Component Weighting Overlap Heatmap.",
        font=dict(size=12),  # Reduce overall font size including subplot titles
    )

    return fig, overlap

# %%
def plot_average_pairwise_overlap(overlaps, cmin=-1, cmax=1):
    n_layers = overlaps.shape[0]
    avg_overlap = overlaps.mean(axis=(2, 3, 4, 5))  # Average over OV and QK heads

    # plot avg_cosine_sims as a heatmap using plotly express
    fig = px.imshow(
        avg_overlap,
        x=[f"{i}" for i in range(n_layers)],
        y=[f"{i}" for i in range(n_layers)],
        zmin=cmin,
        zmax=cmax,
        color_continuous_scale="Viridis",
        labels={"x": "Layer To", "y": "Layer From", "color": "Average Overlap"},
        title="Average Pairwise Overlap of Component Weightings Between All OV-QK Pairs in Each Layer Pair",
        aspect="auto",
        text_auto=".2f",
    )
    # fig.update_layout(
    #     xaxis_title="Layer From",
    #     yaxis_title="Layer To",
    #     xaxis=dict(tickvals=np.arange(n_layers), ticktext=[f"{i}" for i in range(n_layers)]),
    #     yaxis=dict(tickvals=np.arange(n_layers), ticktext=[f"{i}" for i in range(n_layers)]),
    # )
    return fig

# %%
def plot_average_entropy(heatmap, cmin=2, cmax=5):
    # heatmap: (n_layers, n_layers, n_ov, n_qk, n_components)
    n_layers = heatmap.shape[0]
    # entropy = compute_entropy(heatmap, axis=-1)
    avg_dist = heatmap.mean(axis=(2, 3))  # Average over OV and QK heads
    entropy = compute_entropy(avg_dist, axis=-1)
    assert entropy.shape == (n_layers, n_layers)

    # plot avg_cosine_sims as a heatmap using plotly express
    fig = px.imshow(
        entropy,
        x=[f"{i}" for i in range(n_layers)],
        y=[f"{i}" for i in range(n_layers)],
        zmin=cmin,
        zmax=cmax,
        color_continuous_scale="Viridis",
        labels={"x": "Layer To", "y": "Layer From", "color": "Average Entropy"},
        title="Average Entropy of Component Weightings Between All OV-QK Pairs in Each Layer Pair (↑)",
        aspect="auto",
        text_auto=".2f",
    )
    # fig.update_layout(
    #     xaxis_title="Layer From",
    #     yaxis_title="Layer To",
    #     xaxis=dict(tickvals=np.arange(n_layers), ticktext=[f"{i}" for i in range(n_layers)]),
    #     yaxis=dict(tickvals=np.arange(n_layers), ticktext=[f"{i}" for i in range(n_layers)]),
    # )
    return fig

# %%
layer_weights_path_moeut = "analysis_out/dump_slimpajama_moeut_small_matched_rope_noln_long/layer_weights.pth"
layer_weights_path_baseline = (
    "analysis_out/dump_slimpajama_baseline_small_rope_long_nodrop_3/layer_weights.pth"
)
layer_weights_path_moeut_g16 = "analysis_out/dump_slimpajama_moeut_small_g16/layer_weights.pth"
layer_weights_path_baseline_20heads = (
    "analysis_out/dump_slimpajama_baseline_small_20heads/layer_weights.pth"
)
# %%

# layer_weights, heatmap = get_weights_and_heatmap_from_path(layer_weights_path_moeut)
layer_weights_path = layer_weights_path_moeut_g16
ALL_TO_ALL = False
layer_weights, heatmap = get_weights_and_heatmap_from_path(layer_weights_path, all_to_all=ALL_TO_ALL)
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
heatmap_grid_fig = plot_heatmap_grid(heatmap, all_to_all=ALL_TO_ALL, subtract=mean_max, func="max")
heatmap_grid_fig.update_layout(autosize=False, width=1800, height=1500,
    title=f"Composition Scores Heatmap Grid (max over components), minus expected max comp. score of two random matrices ({mean_max:.2f})")
_save_path = Path(layer_weights_path).with_name("heatmap_sub_meanmax.png")
heatmap_grid_fig.write_image(_save_path, "png", scale=2, width=1800, height=1500)
heatmap_grid_fig.show()


# # %%
# # plot for a single layer, e.g. 0.OV to 1.QK
# layer_from = 0
# layer_to = 1
# comp_scores_fig = plot_composition_scores_by_component(heatmap_moeut, layer_from, layer_to, hline=mean_max_moeut)
# comp_scores_fig.update_layout(autosize=False, width=1200, height=800)
# comp_scores_fig.show()

# # %%
# # plot for baseline: 4.OV to 5.QK
# layer_from = 4
# layer_to = 5
# comp_scores_fig = plot_composition_scores_by_component(heatmap_baseline, layer_from, layer_to, hline=mean_max_baseline)
# comp_scores_fig.update_layout(autosize=False, width=1200, height=800)
# comp_scores_fig.show()

# # %%
# layer_from = 0
# layer_to = 1
# threshold = mean_max_moeut
# percentage_fig_moeut = plot_percentage_above_threshold(heatmap_moeut, layer_from, layer_to, threshold)
# percentage_fig_moeut.update_layout(autosize=False, width=1200, height=800)
# percentage_fig_moeut.show()


# %%
# percentage of components above threshold for moeut
for i in range(heatmap.shape[0]):
    for j in range(0 if ALL_TO_ALL else i+1, heatmap.shape[1]):
        print(f"Layer {i} to {j}: {heatmap[i, j].shape}")
        print(f"Mean composition score: {heatmap[i, j].mean().item()}")
        print(f"Max composition score: {heatmap[i, j].max().item()}")
        num_above = (heatmap[i, j] > mean_mean).sum()
        num_total = heatmap[i, j].size
        # percentage of components above threshold
        percentage = num_above / num_total * 100
        print(f"Components above mean-mean random threshold {mean_mean}: {num_above}/{num_total} ({percentage:.2f}%)")

# # %%
# top_components_fig_moeut = plot_top_components(heatmap_moeut, top_n=3, ut=True)
# top_components_fig_moeut.update_layout(autosize=False, width=1200, height=800)
# top_components_fig_moeut.show()

# # %%
# top_components_fig_baseline = plot_top_components(heatmap_baseline, top_n=3, ut=True)
# top_components_fig_baseline.update_layout(autosize=False, width=1800, height=1500)
# top_components_fig_baseline.show()


# %%
percentage_above_threshold_fig = plot_percentage_above_threshold(heatmap, threshold=mean_max, all_to_all=ALL_TO_ALL)
percentage_above_threshold_fig.update_layout(autosize=False, width=1200, height=800,
   title=f"Percentage of Components Above Expected Max Composition Score for Random Matrices ({mean_max:.2f})")
_save_path = Path(layer_weights_path).with_name("pct_above_threshold.png")
percentage_above_threshold_fig.write_image(_save_path, "png", scale=2, width=1800, height=1500)
percentage_above_threshold_fig.show()

# # %%
# TOP_N=1
# top_component_overlap_fig, overlaps = plot_exhaustive_component_overlap(heatmap, all_to_all=ALL_TO_ALL, overlap_metric_fn=lambda x, y: compute_overlap(x, y, top_n=TOP_N), want_lower=True)
# top_component_overlap_fig.update_layout(autosize=False, width=3200, height=1500,
#     title=f"Component Weighting Overlap Heatmap (IOU of top-{TOP_N} component indices). For a given OV-QK pairing, to what extent do different heads pay attention to the same top {TOP_N} components?",
#     font=dict(size=12),  # Reduce overall font size including subplot titles
# )
# _save_path = Path(layer_weights_path).with_name(f"top_{TOP_N}_component_overlap.jpg")
# top_component_overlap_fig.write_image(_save_path, "jpg", scale=1, width=3200, height=1500)
# top_component_overlap_fig.show()


# %%
# Entropy analyses
def normalize_heatmap(heatmap):
    """Normalize heatmap along the last axis (components) to sum to 1."""
    h = heatmap / (np.sum(heatmap, axis=-1, keepdims=True) + 1e-10)
    return h

norm_heatmap = normalize_heatmap(heatmap)

heatmap_entropy_grid_fig = plot_heatmap_grid(norm_heatmap, all_to_all=ALL_TO_ALL, func="entropy")
heatmap_entropy_grid_fig.update_layout(autosize=False, width=1800, height=1500,
    title="Composition Scores Heatmap Grid (entropy over component comp scores)")
_save_path = Path(layer_weights_path).with_name("heatmap_entropy.png")
heatmap_entropy_grid_fig.write_image(_save_path, "png", scale=2, width=1800, height=1500)
heatmap_entropy_grid_fig.show()

# %%
entropy_overlap_fig, _ = plot_exhaustive_component_overlap(norm_heatmap, all_to_all=ALL_TO_ALL, overlap_metric_fn=lambda x, y: compute_entropy((x+y)/2, axis=-1), want_lower=False)
entropy_overlap_fig.update_layout(autosize=False, width=3200, height=1500,
    title="Component Weighting Entropy Overlap. For a given OV-QK pairing, to what extent do different heads pay attention to the same components?",
    font=dict(size=12),  # Reduce overall font size including subplot titles
)
_save_path = Path(layer_weights_path).with_name("entropy_component_overlap.jpg")
# entropy_overlap_fig.write_image(_save_path, "jpg", scale=1, width=3200, height=1500)
entropy_overlap_fig.show()

# %%
avg_layer_entropy_fig = plot_average_entropy(norm_heatmap)
avg_layer_entropy_fig.update_layout(autosize=False, width=800, height=600)
_save_path = Path(layer_weights_path).with_name("avg_layer_entropy.jpg")
avg_layer_entropy_fig.write_image(_save_path, "jpg", scale=1)
avg_layer_entropy_fig.show()

# %%
# MISC

## What we want to check is whether two heads in the same OV-QK pairing are communicating along orthogonal
# information channels. We get at this by checking whether the subspace to which the OV writes and the QK reads is
# orthogonal to those of the other OV-QK pairs.

layer_from = 0
layer_to = 1

# plot weightings of components across all pairs of OV-QK
h_from_to = heatmap[layer_from, layer_to]  # (n_ov, n_qk, d_head)
ov=0
qk=1

# plot line graph of head_from_to[ov, qk]
fig = go.Figure()
fig.add_trace(go.Scatter(y=h_from_to[ov, qk], mode='lines+markers'))
fig.update_layout(title=f"Head {layer_from}.{ov} to {layer_to}.{qk} Weightings",
                  xaxis_title="Component",
                  yaxis_title="Weight")
# also draw mean random horizontal line from mean_mean
fig.add_trace(go.Scatter(y=[mean_mean] * h_from_to[ov, qk].shape[0], mode='lines', name='Mean Random'))
fig.add_trace(go.Scatter(y=[mean_max] * h_from_to[ov, qk].shape[0], mode='lines', name='Max Random'))

# %%
# count number of times each component is attended to across all OV-QK pairs
attended_more_than_mean = h_from_to > mean_mean
attended_count = attended_more_than_mean.sum(axis=(0, 1))
print(attended_count)

# %%
print(heatmap.mean(), heatmap.std())
