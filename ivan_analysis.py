# %%
import os

from plotly.subplots import make_subplots
import torch
import transformer_lens
from transformer_lens.utils import composition_scores
from transformer_lens import FactoredMatrix, HookedTransformer
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

# %%

# # %%
# model = HookedTransformer.from_pretrained(
#     "gpt2-small", fold_value_biases=True, refactor_factored_attn_matrices=True
# )

layer_weights_path = "analysis_out/dump_slimpajama_moeut_small_matched_rope_noln_long/layer_weights.pth"
layer_weights = torch.load(layer_weights_path, weights_only=False)

n_heads = layer_weights[0]["n_heads"]
n_layers = len(layer_weights) # although for moeut we have to do l0 -> l1 and l1 -> l0
d_head = layer_weights[0]["d_head"]

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
def very_exhaustive_heatmap(layer_weights):
    n_experts = layer_weights[0]["n_experts"]["v"]
    assert n_experts == layer_weights[0]["n_experts"]["o"], "Expected n_experts to be the same for v and o"
    n_heads = layer_weights[0]["n_heads"]
    d_head = layer_weights[0]["d_head"]

    heatmap=np.zeros((len(layer_weights), len(layer_weights), n_heads*n_experts, n_heads, d_head))
    for i, layer_from in enumerate(layer_weights):
        for j, layer_to in enumerate(layer_weights):
            # compute composition scores from ov in layer i to qk in layer j
            for ov_idx, ov in enumerate(layer_from["ov"]):
                for qk_idx, qk in enumerate(layer_to["qk"]):
                    print(f"Computing heatmap for {i}.{ov_idx} -> {j}.{qk_idx}")
                    src = ov.svd()
                    # right = qk.svd()
                    for k in range(layer_from["d_head"]):
                        src_comp = re_get_single_component(*src, k)
                        s = composition_scores(src_comp, qk).item()
                        heatmap[i, j, ov_idx, qk_idx, k] = s

    return heatmap


# %%
heatmap = very_exhaustive_heatmap(layer_weights)

# %%
heatmap.shape # (n_layers, n_layers, n_heads*n_experts, n_heads, d_head)

imshow(heatmap.max(axis=-1)[0, 1], return_fig=True, xaxis="OV Head", yaxis="QK Head", title="Composition Scores from 0.OV to 1.QK (max over components)")

# %%
# plot on a 2d grid of heatmaps, x=layer from, y=layer to
def plot_heatmap_grid(heatmap):
    n_layers = heatmap.shape[0]
    fig = make_subplots(
        rows=n_layers, cols=n_layers,
        subplot_titles=[f"{i}.OV -> {j}.QK" for i in range(n_layers) for j in range(n_layers)],
    )

    h = heatmap.max(axis=-1)
    cmin = h.min()
    cmax = h.max()

    for i in range(n_layers):
        for j in range(n_layers):
            fig.add_trace(
                go.Heatmap(
                    z=h[i,j],
                    x=np.arange(heatmap.shape[2]),
                    y=np.arange(heatmap.shape[3]),
                    colorscale="Viridis",
                    showscale=False,
                    name=f"{i}.OV -> {j}.QK",
                    zmin=cmin,
                    zmax=cmax,
                    # colorbar=dict(
                    #     title="Composition Score",
                    #     # titleside="right",
                    #     len=0.5,
                    #     thickness=10,
                    #     x=1.05,
                    #     y=0.5,
                    # ),
                    # labels={
                    #     "x": "OV Head",
                    #     "y": "QK Head",
                    # },
                ),
                row=i + 1,
                col=j + 1,
            )
    fig.update_layout(title="Composition Scores Heatmap Grid (max over components)")
                      #, xaxis_title="OV Head", yaxis_title="QK Head")

    fig.data[-1].update(colorbar = dict(x=1.05, y=0.5, thickness=20), showscale=True)
    return fig

heatmap_grid_fig = plot_heatmap_grid(heatmap)
heatmap_grid_fig.show()

# # %%
# src_layer, src_head = 8, 6  # gpt 2 small
# src_usv = model.blocks[src_layer].attn.OV[src_head].svd()
# dest_layer, dest_head = 9, 9
# decomp = True
# dest_usv = model.blocks[dest_layer].attn.QK[dest_head].svd()

# comp_idx = 0  # gpt2 small
# if decomp:
#     print("decomposing right side to comp", comp_idx)
#     right = re_get_single_component(*dest_usv, comp_idx)
# #


# scores = []

# exh = True
# if exh:
#     heatmap = exhaustive_heatmap(dest_layer, right)
# else:
#     heatmap = simple_heatmap(dest_layer, right)
# # %%
# title = f"Composition Scores to {dest_layer}.{dest_head}"
# if decomp:
#     title = f"Composition Scores to {dest_layer}.{dest_head}.{comp_idx}"
# heatmap = imshow(heatmap, return_fig=True, xaxis="Head", yaxis="Layer", title=title)

# heatmap.show()
# pio.write_image(heatmap, f"out", format="jpg")

# %%
