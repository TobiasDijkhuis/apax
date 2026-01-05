from dataclasses import field
from typing import Any, Callable, List, Optional

import flax.linen as nn
import jax.numpy as jnp

from apax.layers.activation import swish
from apax.layers.ntk_linear import NTKLinear
from apax.utils.convert import str_to_dtype


class AtomisticReadout(nn.Module):
    units: List[int] = field(default_factory=lambda: [32, 32])
    activation_fn: Callable = swish
    w_init: str = "normal"
    b_init: str = "zeros"
    use_ntk: bool = True
    n_shallow_ensemble: int = 0
    is_feature_fn: bool = False
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    dtype: Any = jnp.float32

    def setup(self):
        units = list(self.units)
        if not self.is_feature_fn:
            readout_unit = [1]
            if self.n_shallow_ensemble > 0:
                readout_unit = [self.n_shallow_ensemble]
            units += readout_unit

        dtype = str_to_dtype(self.dtype)

        dense = []
        for ii, n_hidden in enumerate(units):
            layer = NTKLinear(
                n_hidden,
                w_init=self.w_init,
                b_init=self.b_init,
                use_ntk=self.use_ntk,
                dtype=dtype,
                name=f"dense_{ii}",
            )
            dense.append(layer)
            if ii < len(units) - 1:
                dense.append(swish)

        #     dense[-1] = nn.Dropout(rate=self.dropout_rate)(
        #          dense[-1], deterministic=deterministic
        #     )
        self.dense = dense
        # self.sequential = nn.Sequential(dense, name="readout")

    def __call__(self, x, deterministic: Optional[bool] = None):
        # https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/arguments.html
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)

        dense_dropout = self.dense.copy()
        for ii, layer in enumerate(self.dense):
            if ii % 2 == 0:
                continue
            dense_dropout[ii] = nn.Dropout(rate=self.dropout_rate)(
                layer, deterministic=deterministic
            )
        h = nn.Sequential(dense_dropout, name="readout")(x)
        # TODO should we move aggregation here?
        return h
