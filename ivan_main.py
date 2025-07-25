import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ['TORCH_LOGS'] = "+dynamo"
# os.environ['TORCHDYNAMO_VERBOSE'] = "1"

from pathlib import Path
from typing import Dict, Optional

import einops
import framework
from framework.task import task_db
import torch
import json
from framework import dataset
import tasks
from transformer_lens import FactoredMatrix

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False


def register_args(parser: framework.helpers.ArgumentParser):
    task_db.register_args(parser)
    parser.add_argument("-state_size", default=128)
    parser.add_argument("-task", default="tuple")
    parser.add_argument("-dropout", default=0.0)
    parser.add_argument("-embedding_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-transformer.n_heads", default=4)
    parser.add_argument("-transformer.variant", default="standard")
    parser.add_argument("-transformer.ff_multiplier", default=2.0)
    parser.add_argument("-transformer.encoder_n_layers", default=3)
    parser.add_argument("-transformer.attention_dropout", default=0.0)
    parser.add_argument("-load_pretrained_model", type=str)
    parser.add_argument("-test_pretrained", default=1)
    parser.add_argument(
        "-train_baseline",
        default=False,
        help="Train the model on easy task and test on hard," "no masking",
    )
    parser.add_argument("-test_only", default=False)
    parser.add_argument("-nan_detect", default=False)
    parser.add_argument("-fs_cache_pattern", default="*", parser=parser.str_or_none_parser)


def initialize(restore: Optional[str] = None):
    helper = framework.helpers.TrainingHelper(
        wandb_project_name="lm",
        register_args=register_args,
        extra_dirs=["export", "model_weights", "tmp"],
        log_async=True,
        restore=restore,
    )

    dataset.init_fs_cache(helper.args.fs_cache_pattern)
    task = task_db.get_task(helper.args.task)

    task = task(helper)
    return helper, task


def main():
    helper, task = initialize()
    if helper.args.nan_detect:
        torch.autograd.set_detect_anomaly(True)

    if helper.args.load_pretrained_model:
        assert not helper.args.train_baseline

        print("Loading pretrained model...")

        pretrained = os.path.expanduser(helper.args.load_pretrained_model)
        if not helper.args.load_pretrained_model.endswith(".pth"):
            pretrained = os.path.join(
                pretrained, str(helper.args.sweep_id_for_grid_search), "model.pth"
            )

        assert os.path.isfile(
            pretrained
        ), f"Failed to load pretrained weights. File {pretrained} not found."

        if helper.dist_env.is_master():
            task.load_weights(pretrained)

        helper.distibute_model_weights()
        print("Done.")

    # model lives on task.model
    print(task.model)
    if helper.args.transformer.variant == "preln_rope":
        layer_weights = get_baseline_layer_weights(task.model, helper)
    else:
        layer_weights = get_moeut_layer_weights(task.model, helper)

    # Save the layer weights to a torch dump
    layer_weights_path = Path("analysis_out", helper.args.name, "layer_weights.pth")
    os.makedirs(os.path.dirname(layer_weights_path), exist_ok=True)
    torch.save(layer_weights, layer_weights_path)
    print(f"Layer weights saved to {layer_weights_path}")


def QK(W_Q, W_K):
    return FactoredMatrix(W_Q, W_K.transpose(-2, -1))


def OV(W_O, W_V):
    return FactoredMatrix(W_V, W_O)

def get_moeut_layer_weights(model, helper):
    layer_weights = []

    for i, block in enumerate(model.unique_layers):
        proj = block.self_attn.projections
        d_head = helper.args.transformer.head_projection_size
        d_embed = helper.args.state_size
        n_heads = block.self_attn.get_n_copies("k")
        assert (
            n_heads
            == block.self_attn.get_n_copies("q")
            == block.self_attn.get_n_copies("v")
            == block.self_attn.get_n_copies("o")
        ), f"Expected n_heads to be the same for all projections, got {n_heads}, {block.self_attn.get_n_copies('q')}, {block.self_attn.get_n_copies('v')}, {block.self_attn.get_n_copies('o')}"

        # k and q don't have experts
        k = proj.k.view(n_heads, d_head, d_embed).permute(0, 2, 1) # not sure why moeut uses this arrangement of axes
        q = proj.q.view(n_heads, d_head, d_embed).permute(0, 2, 1)

        # v and o have experts
        v = proj.v.view(n_heads, block.self_attn.n_experts["v"], d_embed, d_head)
        o = proj.o.view(n_heads, block.self_attn.n_experts["o"], d_head, d_embed)
        # ...but I don't think that's actually important for TalkingHeads? TODO
        v = v.reshape(-1, d_embed, d_head)
        o = o.reshape(-1, d_head, d_embed)

        l = {}
        # none of the self attn blocks have biases, so we don't need refactor_...
        l["qk"] = [QK(q[h], k[h]) for h in range(n_heads)]
        l["ov"] = [OV(_o, _v) for _o, _v in zip(o, v)]

        l["n_heads"] = n_heads
        l["d_head"] = d_head
        l["d_embed"] = d_embed
        l["n_experts"] = {"v": block.self_attn.n_experts["v"], "o": block.self_attn.n_experts["o"]}

        layer_weights.append(l)

        # there's also .pkm, which has the FFN experts?
    return layer_weights

def get_baseline_layer_weights(model, helper):
    layer_weights = []

    for i, block in enumerate(model.unique_layers):
        proj = block.self_attn
        d_head = helper.args.transformer.head_projection_size
        d_embed = helper.args.state_size
        n_heads = block.self_attn.n_heads

        # proj.data_to_kv.weight: [2 * d_head * n_heads, d_embed]
        # proj.data_to_q.weight: [d_head * n_heads, d_embed]
        # proj.out_proj.weight: [d_embed, d_head * n_heads]
        k, v = proj.data_to_kv.weight.split(n_heads * d_head, dim=0) # [d_head * n_heads, d_embed] each
        q = proj.data_to_q.weight
        o = proj.out_proj.weight

        k = k.view(n_heads, d_head, d_embed).permute(0, 2, 1) # not sure why moeut uses this arrangement of axes
        q = q.view(n_heads, d_head, d_embed).permute(0, 2, 1)
        v = v.view(n_heads, d_head, d_embed).permute(0, 2, 1)
        o = o.view(d_embed, n_heads, d_head).permute(1, 2, 0) # [n_heads, d_head, d_embed]

        l = {}
        # none of the self attn blocks have biases, so we don't need refactor_...
        l["qk"] = [QK(q[h], k[h]) for h in range(n_heads)]
        l["ov"] = [OV(_o, _v) for _o, _v in zip(o, v)]

        l["n_heads"] = n_heads
        l["d_head"] = d_head
        l["d_embed"] = d_embed
        # l["n_experts"] = {"v": block.self_attn.n_experts["v"], "o": block.self_attn.n_experts["o"]}

        layer_weights.append(l)

    return layer_weights


def refactor_factored_attn_matrices(state_dict: Dict[str, torch.Tensor], n_layers: int):
    """Experimental method for managing queries, keys and values.
    Stolen from HookedTransformer

        As argued in [A Mathematical Framework for Transformer
        Circuits](https://transformer-circuits.pub/2021/framework/index.html), queries, keys and
        values are somewhat arbitrary intermediate terms when computing with the low rank factored
        matrices W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O, and these matrices are the only thing
        determining head behaviour. But there are many ways to find a low rank factorization to a
        given matrix, and hopefully some of these are more interpretable than others! This method is
        one attempt, which makes all of the matrices have orthogonal rows or columns, W_O into a
        rotation and W_Q and W_K having the nth column in each having the same norm. The formula is
        $W_V = U @ S,W_O=Vh.T,W_Q=U@S.sqrt(),W_K=Vh@S.sqrt()$.

        More details:

        If W_OV = U @ S @ Vh.T in its singular value decomposition, (where S is in R^d_head not
        R^d_model, as W_OV is low rank), W_OV = (U @ S) @ (Vh.T) is an equivalent low rank
        factorisation, where rows/columns of each matrix are orthogonal! So setting $W_V=US$ and
        $W_O=Vh.T$ works just as well. I *think* this is a more interpretable setup, because now
        $W_O$ is just a rotation, and doesn't change the norm, so $z$ has the same norm as the
        result of the head.

        For $W_QK = W_Q @ W_K.T$ we use the refactor $W_Q = U @ S.sqrt()$ and $W_K = Vh @ S.sqrt()$,
        which is also equivalent ($S==S.sqrt() @ S.sqrt()$ as $S$ is diagonal). Here we keep the
        matrices as having the same norm, since there's not an obvious asymmetry between the keys
        and queries.

        Biases are more fiddly to deal with. For OV it's pretty easy - we just need (x @ W_V + b_V)
        @ W_O + b_O to be preserved, so we can set b_V' = 0. and b_O' = b_V @ W_O + b_O (note that
        b_V in R^{head_index x d_head} while b_O in R^{d_model}, so we need to sum b_V @ W_O along
        the head_index dimension too).

        For QK it's messy - we need to preserve the bilinear form of (x @ W_Q + b_Q) * (y @ W_K +
        b_K), which is fairly messy. To deal with the biases, we concatenate them to W_Q and W_K to
        simulate a d_model+1 dimensional input (whose final coordinate is always 1), do the SVD
        factorization on this effective matrix, then separate out into final weights and biases.
    """

    # assert (
    #     self.cfg.positional_embedding_type != "rotary"
    # ), "You can't refactor the QK circuit when using rotary embeddings (as the QK matrix depends on the position of the query and key)"

    for l in range(n_layers):
        # W_QK = W_Q @ W_K.T
        # Concatenate biases to make a d_model+1 input dimension
        W_Q_eff = torch.cat(
            [
                state_dict[f"blocks.{l}.attn.W_Q"],
                state_dict[f"blocks.{l}.attn.b_Q"][:, None, :],
            ],
            dim=1,
        )
        W_K_eff = torch.cat(
            [
                state_dict[f"blocks.{l}.attn.W_K"],
                state_dict[f"blocks.{l}.attn.b_K"][:, None, :],
            ],
            dim=1,
        )

        W_Q_eff_even, W_K_eff_even_T = (
            FactoredMatrix(W_Q_eff, W_K_eff.transpose(-1, -2)).make_even().pair
        )
        W_K_eff_even = W_K_eff_even_T.transpose(-1, -2)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q_eff_even[:, :-1, :]
        state_dict[f"blocks.{l}.attn.b_Q"] = W_Q_eff_even[:, -1, :]
        state_dict[f"blocks.{l}.attn.W_K"] = W_K_eff_even[:, :-1, :]
        state_dict[f"blocks.{l}.attn.b_K"] = W_K_eff_even[:, -1, :]

        # W_OV = W_V @ W_O
        W_V = state_dict[f"blocks.{l}.attn.W_V"]
        W_O = state_dict[f"blocks.{l}.attn.W_O"]

        # Factors the bias to be consistent.
        b_V = state_dict[f"blocks.{l}.attn.b_V"]
        b_O = state_dict[f"blocks.{l}.attn.b_O"]

        # Add singleton dimension for broadcasting
        b_V_expanded = einops.rearrange(b_V, "head_index d_head -> head_index d_head 1")

        # Element-wise multiplication of b_V and W_O
        b_V_times_W_O = b_V_expanded * W_O

        # Sum over d_head and head_index dimensions
        b_V_contribution = b_V_times_W_O.sum(1).sum(0)

        effective_bias = b_O + b_V_contribution
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros_like(b_V)
        state_dict[f"blocks.{l}.attn.b_O"] = effective_bias

        # Helper class to efficiently deal with low rank factored matrices.
        W_OV = FactoredMatrix(W_V, W_O)
        U, S, Vh = W_OV.svd()
        state_dict[f"blocks.{l}.attn.W_V"] = U @ S.diag_embed()
        state_dict[f"blocks.{l}.attn.W_O"] = Vh.transpose(-1, -2)

    return state_dict


if __name__ == "__main__":
    main()
