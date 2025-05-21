#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:18:16 2025

@author: talasunna
"""

import pytest
import math

from projects.migrating.parameters import (
    RangedParameter,
    ChoiceParameter,
    LinearConstraint,
    ParameterSpace,
)
from CADETProcess import CADETProcessError


# # ────────────────────────────  RangedParameter  ──────────────────────────────
# def test_ranged_parameter_accepts_value_inside_bounds():
#     p = RangedParameter(lb=0, ub=10, parameter_type=int)
#     # Should not raise
#     p.validate(5)


# def test_ranged_parameter_rejects_wrong_type():
#     p = RangedParameter(lb=0, ub=10, parameter_type=int)
#     with pytest.raises(TypeError):
#         p.validate(3.14)  # float instead of int


# def test_ranged_parameter_rejects_out_of_bounds():
#     p = RangedParameter(lb=0, ub=10, parameter_type=int)
#     with pytest.raises(ValueError):
#         p.validate(42)  # outside upper bound


# def test_ranged_parameter_bounds_order_checked():
#     with pytest.raises(ValueError):
#         RangedParameter(lb=5, ub=0, parameter_type=int)


# # ───────────────────────────── ChoiceParameter  ──────────────────────────────
# def test_choice_parameter_accepts_valid_choice():
#     p = ChoiceParameter(valid_values=["red", "green", "blue"])
#     # Should not raise
#     p.validate("green")


# def test_choice_parameter_rejects_invalid_choice():
#     p = ChoiceParameter(valid_values=["red", "green", "blue"])
#     with pytest.raises(ValueError):
#         p.validate("purple")




#%%


# ---------- Fixtures ----------

@pytest.fixture
def int_param():
    return RangedParameter(name="int_param", parameter_type=int, lb=0, ub=10)

@pytest.fixture
def float_param():
    return RangedParameter(name="float_param", parameter_type=float, lb=0.0, ub=5.0)

@pytest.fixture
def choice_param():
    return ChoiceParameter(name="mode", valid_values=["fast", "slow", "medium"])

# ---------- Tests for RangedParameter ----------

@pytest.mark.parametrize("value", [0, 5, 11])
def test_ranged_param_valid(value, int_param):
    if value == 11:
        # should raise error for exclusive upper bound test
        with pytest.raises(ValueError):
            int_param.validate(value)
    else:
        int_param.validate(value)

@pytest.mark.parametrize("value", [-1, 11])
def test_ranged_param_out_of_bounds(value, int_param):
    with pytest.raises(ValueError):
        int_param.validate(value)

@pytest.mark.parametrize("value", [3.5, "five", None])
def test_ranged_param_type_error(value, int_param):
    with pytest.raises(TypeError):
        int_param.validate(value)

# ---------- Tests for ChoiceParameter ----------

@pytest.mark.parametrize("value", ["fast", "slow", "medium"])
def test_choice_param_valid(value, choice_param):
    choice_param.validate(value)

@pytest.mark.parametrize("value", ["ultra", "", 5])
def test_choice_param_invalid(value, choice_param):
    with pytest.raises(ValueError):
        choice_param.validate(value)

# ---------- Optional edge case ----------

def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        _ = RangedParameter(name="bad", parameter_type=int, lb=10, ub=5)


if __name__ == "__main__":
    pytest.main([__file__])
