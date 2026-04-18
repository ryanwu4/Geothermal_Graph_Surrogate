# NPV Objective Differences

This note documents how the current proxy optimization objective differs from the
full NPV calculation in GeologicalSimulationWrapper.

## 1) Surrogate Target Used by Inference

- Target name: `graph_discounted_net_revenue`
- Defined in preprocessing as discounted net energy revenue from annual field rates:
  - `net_energy_t = FieldEnergyProductionRate_t - FieldEnergyInjectionRate_t`
  - `revenue_t = net_energy_t * (ENERGY_PRICE / 3600.0)`
  - `discount_t = (1 / (1 + DISCOUNT_FACTOR))^t`, `t = 1..T`
  - `graph_discounted_net_revenue = sum_t(revenue_t * discount_t)`

## 2) Current Inference NPV Proxy (No Facility Terms, No Gate)

The optimization script now uses:

- `NPV_proxy = -CAPEX_wells + discounted_revenue - discounted_opex`
- `CAPEX_wells = sum(well_distance_i) * WELL_COST_PER_DISTANCE`
- `discounted_opex = sum_t(annual_opex * discount_t)`
- `annual_opex = fixed_opex + active_water_opex`

with:

- `fixed_opex = OPEX_WATER_INJECTOR * N_inj + OPEX_WATER_PRODUCER * N_prod`
- `active_water_opex = OPEX_ACTIVE_INJECTOR_PER_M3_WATER * injector_rate_const * ANNUAL_WATER_RATE_SCALE * N_inj`
  `+ OPEX_ACTIVE_PRODUCER_PER_M3_WATER * |producer_rate_const| * ANNUAL_WATER_RATE_SCALE * N_prod`

## 3) Julia Assumption Check (Variable-by-Variable)

Variables aligned with GeologicalSimulationWrapper `ECONOMICS` keys:

- `ENERGY_PRICE` (euros/kWh)
- `DISCOUNT_FACTOR`
- `WELL_COST_PER_DISTANCE`
- `OPEX_WATER_INJECTOR`
- `OPEX_WATER_PRODUCER`
- `OPEX_ACTIVE_INJECTOR_PER_M3_WATER`
- `OPEX_ACTIVE_PRODUCER_PER_M3_WATER`

Deliberate differences from full Julia economics:

- Excluded: `CAPEX_SURFACE_FACILITIES`
- Excluded: `CAPEX_FLOWLINES_BETWEEN_LOCATIONS_PER_DISTANCE`
- Excluded: `CAPEX_FLOWLINES_WELL_TO_LOCATION_BETWEEN_SURFACE_FACILITIES_PER_DISTANCE`
- Excluded: `OPEX_RATE_FROM_CAPEX_SURFACE_FACILITIES` contribution
- Excluded: OPERATED gate (`revenue > opex`) during optimization

Proxy-only assumptions (not direct Julia config keys):

- `INJECTOR_RATE_CONSTANT`
- `PRODUCER_RATE_CONSTANT`
- `ANNUAL_WATER_RATE_SCALE`
- `PLANNING_YEARS`
- `DEPTH_TO_DISTANCE_SCALE`

These are used to approximate annual water-related OPEX from a constant pumping-rate assumption,
instead of using full simulator annual water time-series per geology.

## 4) Interpretation

The current objective is intentionally smoother and simpler than full Julia NPV:

- includes well CAPEX + fixed/active-water OPEX,
- excludes facility/flowline economics,
- excludes operated/shut-in gating.

This is the intended setup for differentiable coordinate optimization under the current scope.
