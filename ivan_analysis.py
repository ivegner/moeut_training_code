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

layer_weights_path_moeut = "analysis_out/dump_slimpajama_moeut_small_matched_rope_noln_long/layer_weights.pth"
layer_weights_path_baseline = (
    "analysis_out/dump_slimpajama_baseline_small_rope_long_nodrop_3/layer_weights.pth"
)
layer_weights_moeut = torch.load(layer_weights_path_moeut, weights_only=False)
layer_weights_baseline = torch.load(layer_weights_path_baseline, weights_only=False)


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
def very_exhaustive_heatmap(layer_weights, ut=True):
    if ut:
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
                if not ut and i >= j:
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


# %%
if not os.path.exists("heatmap_moeut.pkl"):
    heatmap_moeut = very_exhaustive_heatmap(layer_weights_moeut, ut=True)
    with open("heatmap_moeut.pkl", "wb") as f:
        torch.save(heatmap_moeut, f)
else:
    heatmap_moeut = torch.load("heatmap_moeut.pkl", weights_only=False)

# %%
if not os.path.exists("heatmap_baseline.pkl"):
    heatmap_baseline = very_exhaustive_heatmap(layer_weights_baseline, ut=False)
    with open("heatmap_baseline.pkl", "wb") as f:
        torch.save(heatmap_baseline, f)
else:
    heatmap_baseline = torch.load("heatmap_baseline.pkl", weights_only=False)

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
# plot on a 2d grid of heatmaps, x=layer from, y=layer to
def plot_heatmap_grid(heatmap, ut=True):
    n_layers = heatmap.shape[0]
    pairs = [(i, j) for i in range(n_layers) for j in range(n_layers)]
    fig = make_subplots(
        rows=n_layers,
        cols=n_layers,
        subplot_titles=[f"{i}.OV -> {j}.QK" if (ut or i < j) else "" for i, j in pairs],
    )

    h = heatmap.max(axis=-1)
    cmin = h.min()
    cmax = h.max()

    for i in range(n_layers):
        for j in range(n_layers):
            if not ut and i >= j:
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
    # , xaxis_title="OV Head", yaxis_title="QK Head")

    fig.data[-1].update(colorbar=dict(x=1.05, y=0.5, thickness=20), showscale=True)
    return fig

# %%
heatmap_grid_fig = plot_heatmap_grid(heatmap_moeut, ut=True)
# heatmap_grid_fig.update_layout(autosize=False, width=1800, height=1500)
heatmap_grid_fig.show()

# %%
heatmap_grid_fig = plot_heatmap_grid(heatmap_baseline, ut=False)
heatmap_grid_fig.update_layout(autosize=False, width=1800, height=1500)
heatmap_grid_fig.show()

# %%
# Find average composition scores for random matrices of the same size
def random_composition_scores(d_embed, d_head, n_runs=10):
    maxes, means = [], []
    for _ in range(n_runs):
        # Generate random matrices
        qk = FactoredMatrix(
            torch.randn(d_embed, d_head, device="cuda"), torch.randn(d_head, d_embed, device="cuda")
        )
        ov = FactoredMatrix(
            torch.randn(d_embed, d_head, device="cuda"), torch.randn(d_head, d_embed, device="cuda")
        )
        layer_weights = {
            "qk": [qk],
            "ov": [ov],
            "d_head": d_head,
            "n_heads": 1,
            "n_experts": {"v": 1, "o": 1},
        }
        # Compute composition scores
        heatmap = very_exhaustive_heatmap([layer_weights], ut=True) # (1,1,1,1,d_head)
        heatmap = heatmap.squeeze((0, 1, 2, 3))
        _max = heatmap.max().item()
        _mean = heatmap.mean().item()
        maxes.append(_max)
        means.append(_mean)
    return maxes, means

# %%
# randoms for moeut
d_embed = layer_weights_moeut[0]["d_embed"]
d_head = layer_weights_moeut[0]["d_head"]
maxes_moeut, means_moeut = random_composition_scores(d_embed, d_head, n_runs=10)
mean_max_moeut = np.mean(maxes_moeut)
mean_mean_moeut = np.mean(means_moeut)
print(f"Mean of max composition score for random MoeUT: {mean_max_moeut} ± {np.std(maxes_moeut)}")
print(f"Mean of mean composition score for random MoeUT: {mean_mean_moeut} ± {np.std(means_moeut)}")

# %%
# randoms for baseline
d_embed = layer_weights_baseline[0]["d_embed"]
d_head = layer_weights_baseline[0]["d_head"]
maxes_baseline, means_baseline = random_composition_scores(d_embed, d_head, n_runs=10)
mean_max_baseline = np.mean(maxes_baseline)
mean_mean_baseline = np.mean(means_baseline)
print(f"Mean of max composition score for random Baseline: {mean_max_baseline} ± {np.std(maxes_baseline)}")
print(f"Mean of mean composition score for random Baseline: {mean_mean_baseline} ± {np.std(means_baseline)}")

# %%


# %%


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
