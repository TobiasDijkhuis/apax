import logging
from typing import Any, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
)

from apax.config.lr_config import CyclicCosineLR, LinearLR

log = logging.getLogger(__name__)


class OptimizerConfig(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the optimizer.
    Learning rates of 0 will freeze the respective parameters.

    Parameters
    ----------
    name : str, default = "adam"
        Name of the optimizer. Can be any `optax` optimizer.
    emb_lr : NonNegativeFloat, default = 0.001
        Learning rate of the elemental embedding contraction coefficients.
    nn_lr : NonNegativeFloat, default = 0.001
        Learning rate of the neural network parameters.
    scale_lr : NonNegativeFloat, default = 0.0001
        Learning rate of the elemental output scaling factors.
    shift_lr : NonNegativeFloat, default = 0.003
        Learning rate of the elemental output shifts.
    zbl_lr : NonNegativeFloat, default = 0.0001
        Learning rate of the ZBL correction parameters.
    rep_scale_lr : NonNegativeFloat, default = 0.001
        LR for the length scale of these exponential repulsion potential.
    rep_prefactor_lr : NonNegativeFloat, default = 0.0001
        LR for the strength of the exponential repulsion potential.
    gradient_clipping: NonNegativeFloat, default = 1000.0
        Per element Gradient clipping value.
        Default is so high that it effectively disabled.
    schedule : LRSchedule = LinearLR
        Learning rate schedule.
    kwargs : dict, default = {}
        Optimizer keyword arguments. Passed to the `optax` optimizer.
    """

    name: str = "adam"

    gradient_clipping: NonNegativeFloat = 1000.0

    schedule: Union[LinearLR, CyclicCosineLR] = Field(
        LinearLR(name="linear"), discriminator="name"
    )
    kwargs: Optional[dict[str, Any]] = Field(default_factory=dict)


class ModelOptimizerConfig(OptimizerConfig, frozen=True, extra="forbid"):
    emb_lr: NonNegativeFloat = 0.001
    nn_lr: NonNegativeFloat = 0.001
    scale_lr: NonNegativeFloat = 0.0001
    shift_lr: NonNegativeFloat = 0.003
    zbl_lr: NonNegativeFloat = 0.0001
    rep_scale_lr: NonNegativeFloat = 0.001
    rep_prefactor_lr: NonNegativeFloat = 0.0001


class GeometryOptimizerConfig(OptimizerConfig, frozen=True, extra="forbid"):
    lr: NonNegativeFloat = 0.1
