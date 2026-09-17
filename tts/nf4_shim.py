"""Provide the one symbol `torchtune` imports from a module torchao 0.18 removed.

neucodec -> torchtune -> `from torchao.dtypes.nf4tensor import NF4Tensor`, which no longer
exists in torchao 0.18. Pinning torchao back to 0.12 instead breaks against this box's
torch 2.14 (`torch._inductor.kernel.flex_attention` moved), so neither version satisfies
both.

NF4 is 4-bit quantisation for LLM fine-tuning; the audio codec path never touches it. So
register a placeholder class under the expected path purely to satisfy the import, and let
anything that actually tries to USE it fail loudly rather than silently mis-quantise.

Import this before neucodec.
"""
import sys
import types


def install() -> None:
    try:
        from torchao.dtypes.nf4tensor import NF4Tensor  # noqa: F401
        return                                          # real one present; do nothing
    except Exception:
        pass

    mod = types.ModuleType("torchao.dtypes.nf4tensor")

    class NF4Tensor:                                    # noqa: D401 - placeholder only
        """Placeholder. torchtune imports this symbol; the codec path never uses it."""

        def __init__(self, *a, **k):
            raise RuntimeError(
                "NF4Tensor is a shim installed to satisfy a torchtune import "
                "(torchao 0.18 removed torchao.dtypes.nf4tensor). It has no real "
                "implementation - if you hit this, NF4 quantisation is genuinely needed "
                "and torchao/torch versions must be reconciled properly."
            )

    def to_nf4(*a, **k):
        raise RuntimeError("to_nf4 is a shim; see NF4Tensor above.")

    mod.NF4Tensor = NF4Tensor
    mod.to_nf4 = to_nf4
    mod.linear_nf4 = to_nf4
    sys.modules["torchao.dtypes.nf4tensor"] = mod
