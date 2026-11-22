# Final figure execution order

Run scripts from their own directory. All paths here are under
`13_CMIP6_GridPath_Figures/`. The numbered directories are retained to preserve
the archived code's relative paths.

## Physical states and Main Figures 1 and 2

Start from the audited physical reference inputs in `docs/DATA.md`.

1. Rebuild the four CMIP6 monthly cores with `repair_cmip6_monthly_routes.py`.
2. Build physical-edge member masks with `prepare_physical_monthly_ice.py`.
3. Run `rebuild_mainfig1_latest_network.py prepare`, `route` and `collect`.
4. Run `prepare_mainfig1_trade_weights.py` and
   `prepare_mainfig1_density_region_sankey.py`.
5. Plot Main Figure 1 using `plot_mainfig1_latest_network.py`.
6. Run `rebuild_mainfig2_latest_network.py`,
   `calculate_mainfig2_cost_uncertainty.py`, then `plot_mainfig2_latest_network.py`.

The baseline uses accepted actual AIS geometry and documented water-network
fallbacks. NWP, NEP, NSR and TSR candidates require a complete supported voyage.

## Main Figure 3 and associated supplements

1. Compute the 24-gate one-to-five closure bank using
   `recompute_mainfig3_five_closures.py`.
2. Run `58_*/prepare_coastal_masks.py` and `58_*/calculate_thresholds.py`.
3. Run `60_*/analyze_two_parts.py`.
4. Plot `60_*/plot_partI_extreme_mainfig3.py` and `60_*/plot_auxiliary.py`.
5. From repository root, run `python tools/final_artwork.py --prepare-main3`
   to relabel the existing vector panels, stage the unchanged numerical series
   and add the final dual-axis response panels.

The same scripts create the all-disturbance gaps and selected vessel response
curves used for Supplementary Figure 3 and Extended Data Figure 3.

## Main Figure 4 and Supplementary Figures 4 and 5

Run `60_*/select_dependent_closures.py`, `replay_rank_routes.py`,
`calculate_new_country_ranks.py`, `plot_new_mainfig4.py` and
`plot_auxiliary.py`. These use dependence of newly diverted ODs to select gate
sets; they are not the earlier unconstrained dominance-selection figure.

## Main Figure 5 and Supplementary Figures 6 and 7

Run `62_*/recalculate_economics.py` then `plot_aligned_economics.py`.
The route selection is held fixed to Main Figure 4. The plotter now produces
only its economic panels; historical copying of other main figures is removed.

## Extended Data Figures 1 and 2

- Events: `rebuild_portwatch_event_appendix.py`.
- Regional/gate dependence: `40_*/build_bipartite_by_vessel.py`.

## Supplementary Figures 1 and 2

- Climate scenario/windows: `44_*/compute_seasonality.py`,
  `compute_missingness_sensitivity.py`, then `validate_and_plot.py`.
- Admission threshold: `45_*/compute_sensitivity.py`, then
  `validate_and_plot.py`.

## Supplementary Figures 8 and 9

Prepare the S24 market inputs with `71_*/prepare_inputs.py`. Score standing
buyer/exporter policies using `71_*/rank_market.py`, then run `analyse.py`
and `plot_benefits.py`.

For the direct supplier-switch experiment:

```sh
cd 13_CMIP6_GridPath_Figures/72_Supplier_Switch_Experiment_20261111
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python switch_market.py \
  --model qwen3 --model-dir /path/to/qwen3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python switch_market.py \
  --model qwen25 --model-dir /path/to/qwen25
python analyse_switch.py
python plot_switch.py
```

## Final numbering

After generating the underlying PDFs, use
`python tools/final_artwork.py --collect`. This copies or extracts the
appropriate PDF page into the final five main, three Extended Data and nine
Supplementary figure filenames under `final_figures/`. No figure captions,
manuscript prose or document-generation code is run.
