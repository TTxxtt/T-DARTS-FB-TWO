"""Deterministic, name-addressed parameter initialisation.

Why this exists
---------------
PyTorch initialises modules in creation order using the global RNG.  The moment
two model variants build their submodules in a different order -- or one variant
has a module the other lacks -- every parameter after that point diverges, even
under the same seed.  Comparing "RF-only" against "operator-only" against a joint
search would then be comparing initialisations as much as architectures.

The fix is to derive each parameter's seed from its *name* rather than its
position:

    seed(param) = crc32(f"{run_seed}:{param_name}")

Consequences this buys:

* two variants that share a parameter name get bit-identical initial values,
  regardless of module creation order or of what else the model contains;
* a different ``run_seed`` gives different values;
* the result is stable across processes and machines, which
  :func:`hash` is not -- Python randomises ``str`` hashing per process unless
  ``PYTHONHASHSEED`` is set, so ``hash()`` must not be used here.

This module is a prepared tool.  Stage 1 does not rewire the official FBNAS
initialisation; it only provides and tests the machinery.
"""

from __future__ import annotations

import zlib
from typing import Iterable, Mapping

import torch
import torch.nn as nn

__all__ = [
    "stable_seed",
    "parameter_seed",
    "make_generator",
    "seeded_",
    "naive_seeded_",
    "initialise_deterministic",
    "parameter_checksum",
]

#: Default distributions, matching common PyTorch conventions.
DEFAULT_WEIGHT_INIT = "kaiming_uniform"
DEFAULT_BIAS_INIT = "zeros"


def stable_seed(run_seed: int, name: str) -> int:
    """Deterministic 32-bit seed for ``(run_seed, name)``.

    Uses CRC32, which is stable across processes, machines and Python versions.
    Python's built-in :func:`hash` is explicitly unsuitable: it is salted per
    process for ``str`` inputs.
    """
    payload = f"{int(run_seed)}:{name}".encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


#: Backwards-compatible alias.
parameter_seed = stable_seed


def make_generator(run_seed: int, name: str, device: torch.device | str = "cpu"):
    """A :class:`torch.Generator` seeded for ``(run_seed, name)``."""
    generator = torch.Generator(device=device)
    generator.manual_seed(stable_seed(run_seed, name))
    return generator


# ----------------------------------------------------------------------
# in-place seeded initialisers
# ----------------------------------------------------------------------
def _draw(
    tensor: torch.Tensor,
    distribution: str,
    generator: torch.Generator,
    **kwargs,
) -> torch.Tensor:
    """Draw values for ``tensor`` from ``distribution``."""
    shape = tuple(tensor.shape)
    if distribution == "zeros":
        return torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    if distribution == "ones":
        return torch.ones(shape, dtype=tensor.dtype, device=tensor.device)
    if distribution in ("normal", "trunc_normal"):
        mean = kwargs.get("mean", 0.0)
        std = kwargs.get("std", 0.02)
        return torch.normal(
            mean=mean,
            std=std,
            size=shape,
            generator=generator,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    if distribution == "uniform":
        low = kwargs.get("low", -0.05)
        high = kwargs.get("high", 0.05)
        return torch.rand(
            shape, generator=generator, dtype=tensor.dtype, device=tensor.device
        ) * (high - low) + low
    if distribution == "kaiming_uniform":
        fan_in = kwargs.get("fan_in")
        if not fan_in:
            # Fall back to a normal draw rather than guessing the fan-in: a
            # wrong fan-in would be a silently different initialisation.
            return torch.normal(
                mean=0.0,
                std=kwargs.get("std", 0.02),
                size=shape,
                generator=generator,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        bound = (3.0 / float(fan_in)) ** 0.5
        return torch.rand(
            shape, generator=generator, dtype=tensor.dtype, device=tensor.device
        ) * (2 * bound) - bound
    raise ValueError(f"unsupported distribution {distribution!r}")


def _fan_in(module: nn.Module, tensor: torch.Tensor) -> int | None:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        return int(module.weight[0].numel())
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return int(tensor.shape[0])
    if isinstance(module, nn.GroupNorm):
        return int(tensor.shape[0])
    return None


def seeded_(
    module: nn.Module,
    run_seed: int,
    name_prefix: str = "",
    weight_init: str = DEFAULT_WEIGHT_INIT,
    bias_init: str = DEFAULT_BIAS_INIT,
    **init_kwargs,
) -> nn.Module:
    """Initialise every parameter of ``module`` from its fully-qualified name.

    Walks the whole module tree, seeding each parameter from the dotted path
    that reaches it.  Because the seed depends only on that path and on
    ``run_seed``, two models that share a parameter name initialise identically
    even if they were built in a different order or contain extra modules.

    Safe to call on a *submodule* too: pass the same ``name_prefix`` the
    parameter would have inside the parent and the values match, which is what
    makes the order-independence property testable.
    """
    # Walk submodules rather than `named_parameters(recurse=True)` so that the
    # owning module is available for fan-in.  `named_modules()` de-duplicates
    # shared modules, so an aliased submodule is not seeded twice through
    # different paths -- and seeding is idempotent anyway.
    for mod_name, submodule in module.named_modules():
        for param_name, param in submodule.named_parameters(recurse=False):
            if param_name is None:  # pragma: no cover - unnamed parameter
                continue

            # Dotted path of this parameter inside `module`.
            parts = [p for p in (mod_name, param_name) if p]
            local_name = ".".join(parts)
            full_name = (
                f"{name_prefix}.{local_name}" if name_prefix else local_name
            )

            generator = make_generator(run_seed, full_name, device=param.device)
            if param_name == "weight":
                values = _draw(
                    param.data,
                    weight_init,
                    generator,
                    fan_in=_fan_in(submodule, param),
                    **init_kwargs,
                )
            elif param_name == "bias":
                values = _draw(param.data, bias_init, generator, **init_kwargs)
            else:
                # Any other parameter (e.g. norm weights): a stable, name-derived
                # draw rather than leaving it at whatever the default was.
                values = _draw(param.data, weight_init, generator, **init_kwargs)
            with torch.no_grad():
                param.data.copy_(values)
    return module


def naive_seeded_(
    module: nn.Module,
    run_seed: int,
    weight_init: str = DEFAULT_WEIGHT_INIT,
    bias_init: str = DEFAULT_BIAS_INIT,
    **init_kwargs,
) -> nn.Module:
    """Order-*dependent* counterpart to :func:`seeded_`, for contrast tests.

    Seeds one RNG from ``run_seed`` and draws parameters in ``named_parameters``
    order.  Two models that differ only in module creation order therefore get
    different values, which is exactly the failure mode :func:`seeded_` avoids.
    """
    generator = torch.Generator()
    generator.manual_seed(int(run_seed) & 0xFFFFFFFF)
    for name, param in module.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "weight":
            values = _draw(
                param.data,
                weight_init,
                generator,
                fan_in=_fan_in(module, param),
                **init_kwargs,
            )
        else:
            values = _draw(param.data, bias_init, generator, **init_kwargs)
        with torch.no_grad():
            param.data.copy_(values)
    return module


def initialise_deterministic(
    module: nn.Module,
    run_seed: int,
    weight_init: str = DEFAULT_WEIGHT_INIT,
    bias_init: str = DEFAULT_BIAS_INIT,
    **init_kwargs,
) -> nn.Module:
    """Name-addressed initialisation over a whole module tree."""
    return seeded_(
        module,
        run_seed,
        name_prefix="",
        weight_init=weight_init,
        bias_init=bias_init,
        **init_kwargs,
    )


# ----------------------------------------------------------------------
# verification helpers
# ----------------------------------------------------------------------
def parameter_checksum(module: nn.Module) -> dict[str, int]:
    """CRC32 of every parameter's raw bytes, keyed by name.

    Lets a test assert bit-identity without materialising full tensors for
    comparison, and makes it obvious *which* parameter differs when it fails.
    """
    sums: dict[str, int] = {}
    for name, param in module.named_parameters():
        raw = param.detach().cpu().contiguous().numpy().tobytes()
        sums[name] = zlib.crc32(raw) & 0xFFFFFFFF
    return sums
