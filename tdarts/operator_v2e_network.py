"""Network for the Expressive-V2 pilot: the V2 network with a different builder.

Deliberately thin.  The three cells, the band split, the 12-channel path and the
unchanged FBNAS backbone are what make a run comparable to the Matched
generation's runs, so they are *inherited* rather than copied -- a second copy
of that layout could drift, and nothing guards the network file the way the
entry point is guarded.  The only thing that differs between the generations is
which operator families the cells are built from.

The upstream backbone is again untouched: it sees the same ``[B, 36, E, T]``
tensor either way, so a score difference between generations is attributable to
the operator families and not to anything downstream of them.
"""

from __future__ import annotations

from tdarts import config as C
from tdarts.operator_v2_network import OperatorV2Net
from tdarts.operator_v2e import (
    WIDE_CONTROL_REGISTRY,
    build_e_capacity_control,
    build_e_operator,
)

__all__ = ["OperatorV2ENet"]


class OperatorV2ENet(OperatorV2Net):
    """Same three cells and the same backbone; only the operator builder differs.

    Kept as a named class rather than calling ``OperatorV2Net(op, builder=...)``
    at each site so that ``config.json``'s ``model.describe()``, the tests and
    any later reporting have one stable place to point at.
    """

    def __init__(
        self,
        op_name: str,
        *,
        target_rf: int = 57,
        n_electrodes: int = C.NUM_ELECTRODES,
        n_classes: int = C.NUM_CLASSES,
    ) -> None:
        # The network builds either kind, because the ablation has to be
        # runnable through the same protocol.  What keeps controls out of the
        # pilot is the CLI: --capacity-control is a separate, mutually exclusive
        # flag, and the strict build_e_operator refuses a control name.  The
        # separation belongs at the boundary a grid is drawn from, not here.
        builder = (
            build_e_capacity_control
            if op_name in WIDE_CONTROL_REGISTRY
            else build_e_operator
        )
        super().__init__(
            op_name,
            target_rf=target_rf,
            n_electrodes=n_electrodes,
            n_classes=n_classes,
            builder=builder,
        )
