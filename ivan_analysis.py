# %%
import os

import transformer_lens
from transformer_lens.utils import composition_scores
from transformer_lens import FactoredMatrix, HookedTransformer
import numpy as np
import plotly.express as px
import plotly.graph_objs as go


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


# %%
model = HookedTransformer.from_pretrained(
    "gpt2-small", fold_value_biases=True, refactor_factored_attn_matrices=True
)

n_heads = model.cfg.n_heads
n_lays = model.cfg.n_layers
d_head = model.cfg.d_head


# def get_qk(l, h):
#     return model.blocks[l].attn.QK[h]



# create a heatmap of all heads before layer 7
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


# %%

# %%
src_layer, src_head = 8, 6  # gpt 2 small
src_usv = model.blocks[src_layer].attn.OV[src_head].svd()
dest_layer, dest_head = 9, 9
decomp = True
dest_usv = model.blocks[dest_layer].attn.QK[dest_head].svd()

comp_idx = 0  # gpt2 small
if decomp:
    print("decomposing right side to comp", comp_idx)
    right = re_get_single_component(*dest_usv, comp_idx)
#


scores = []

exh = True
if exh:
    heatmap = exhaustive_heatmap(dest_layer, right)
else:
    heatmap = simple_heatmap(dest_layer, right)
# %%
title = f"Composition Scores to {dest_layer}.{dest_head}"
if decomp:
    title = f"Composition Scores to {dest_layer}.{dest_head}.{comp_idx}"
heatmap = imshow(heatmap, return_fig=True, xaxis="Head", yaxis="Layer", title=title)

heatmap.show()
