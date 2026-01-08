import importlib
from apax.utils.helpers import get_attr_from_module
import pytest
from ase.calculators.orca import ORCA
from jax import value_and_grad
from matplotlib.pyplot import scatter

test_attribute_import_data = [
    ("ase.calculators.orca", "ORCA", ORCA),
    ("jax", "value_and_grad", value_and_grad),
    ("matplotlib.pyplot", "scatter", scatter),
]


@pytest.mark.parametrize(
    "module_name, attribute_name, expected_object", test_attribute_import_data
)
def test_get_attribute_from_module(module_name, attribute_name, expected_object):
    assert get_attr_from_module(module_name, attribute_name) == expected_object
