"""Shared economics helpers for discounting and revenue calculations."""

from __future__ import annotations

import h5py
import numpy as np


def compute_real_discount_rate(economics: dict) -> float:
    if "REAL_DISCOUNT_RATE" in economics:
        real_rate = float(economics["REAL_DISCOUNT_RATE"])
    elif "NOMINAL_DISCOUNT_RATE" in economics and "INFLATION_RATE" in economics:
        nominal_rate = float(economics["NOMINAL_DISCOUNT_RATE"])
        inflation_rate = float(economics["INFLATION_RATE"])
        if nominal_rate <= -1.0:
            raise ValueError(f"NOMINAL_DISCOUNT_RATE must be > -1.0, got {nominal_rate}")
        if inflation_rate <= -1.0:
            raise ValueError(f"INFLATION_RATE must be > -1.0, got {inflation_rate}")
        real_rate = (1.0 + nominal_rate) / (1.0 + inflation_rate) - 1.0
    else:
        raise ValueError(
            "Missing discount rate terms. Provide NOMINAL_DISCOUNT_RATE and INFLATION_RATE "
            "or REAL_DISCOUNT_RATE."
        )

    if real_rate <= -1.0:
        raise ValueError(f"REAL_DISCOUNT_RATE must be > -1.0, got {real_rate}")

    return real_rate


def resolve_real_discount_rate_from_attrs(attrs: h5py.AttributeManager) -> float:
    if "target_graph_discounted_net_revenue_real_discount_rate" in attrs:
        real_rate = float(attrs["target_graph_discounted_net_revenue_real_discount_rate"])
    elif (
        "target_graph_discounted_net_revenue_nominal_discount_rate" in attrs
        and "target_graph_discounted_net_revenue_inflation_rate" in attrs
    ):
        nominal_rate = float(attrs["target_graph_discounted_net_revenue_nominal_discount_rate"])
        inflation_rate = float(attrs["target_graph_discounted_net_revenue_inflation_rate"])
        if nominal_rate <= -1.0:
            raise ValueError(f"Invalid nominal discount rate in dataset attrs: {nominal_rate}")
        if inflation_rate <= -1.0:
            raise ValueError(f"Invalid inflation rate in dataset attrs: {inflation_rate}")
        real_rate = (1.0 + nominal_rate) / (1.0 + inflation_rate) - 1.0
    else:
        raise KeyError("Missing real discount rate attrs for discounted revenue")

    if real_rate <= -1.0:
        raise ValueError(f"Invalid real discount rate in dataset attrs: {real_rate}")

    return real_rate


def discounted_revenue_from_rates(
    prod_rate: np.ndarray,
    inj_rate: np.ndarray,
    energy_price_kwh: float,
    discount_rate: float,
) -> float:
    prod = np.asarray(prod_rate, dtype=np.float64).reshape(-1)
    inj = np.asarray(inj_rate, dtype=np.float64).reshape(-1)
    net = prod - inj
    years = np.arange(1, len(net) + 1, dtype=np.float64)
    discount = (1.0 / (1.0 + discount_rate)) ** years
    energy_price_kj = energy_price_kwh / 3600.0
    return float(np.sum(net * energy_price_kj * discount))


def compute_discounted_net_energy_revenue(
    energy_prod_rate: np.ndarray,
    energy_inj_rate: np.ndarray,
    economics: dict,
) -> float:
    """Compute discounted net energy revenue from annual field energy rates.

    This target intentionally excludes CAPEX and OPEX so geometry-linked costs can be
    assembled later in differentiable optimization.
    """
    prod = np.asarray(energy_prod_rate, dtype=np.float64).reshape(-1)
    inj = np.asarray(energy_inj_rate, dtype=np.float64).reshape(-1)
    if prod.shape != inj.shape:
        raise ValueError(
            f"Shape mismatch for energy rates: prod {prod.shape} vs inj {inj.shape}"
        )

    discount_rate = compute_real_discount_rate(economics)

    # ENERGY_PRICE is in currency/kWh; convert to currency/kJ.
    energy_price_kj = float(economics["ENERGY_PRICE"]) / 3600.0

    net_energy_rate = prod - inj
    years = np.arange(1, len(net_energy_rate) + 1, dtype=np.float64)
    discount = (1.0 / (1.0 + discount_rate)) ** years
    discounted_revenue = np.sum(net_energy_rate * energy_price_kj * discount)
    return float(discounted_revenue)
