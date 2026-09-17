"""Load NeuCodec from any repo, including a mirror its own loader refuses.

`NeuCodec._from_pretrained` asserts the repo id is one of the two upstream ones, so a
mirror kept against upstream disappearing could not actually be loaded — which would make
the mirror useless as insurance. This reproduces what that loader does (build the module,
load `pytorch_model.bin`, drop the keys the released checkpoints don't match) for an
arbitrary repo id or local path.

Note the module's constructor also downloads `facebook/w2v-bert-2.0` for its semantic
encoder, so surviving without the Hub means having that cached (or mirrored) too.
"""

import os

import torch

UPSTREAM = ("neuphonic/neucodec", "neuphonic/distill-neucodec")
IGNORE_KEYS = ("fc_post_s", "SemanticDecoder")


def load(repo="neuphonic/neucodec", token=None, map_location="cpu"):
    from neucodec import NeuCodec

    if repo in UPSTREAM:
        return NeuCodec.from_pretrained(repo, token=token)

    if os.path.isdir(repo):
        ckpt = os.path.join(repo, "pytorch_model.bin")
    else:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo, "pytorch_model.bin", token=token)

    model = NeuCodec(24_000, 480)
    state = torch.load(ckpt, map_location=map_location)
    state = {k: v for k, v in state.items()
             if not any(i in k for i in IGNORE_KEYS)}
    model.load_state_dict(state, strict=False)
    return model
