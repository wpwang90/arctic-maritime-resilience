# Arctic maritime resilience

Calculation and plotting code for the **final five main figures, three
Extended Data figures and nine Supplementary figures**.

Only final figure entry points and their required local helper modules are
included. Earlier figure designs, unrelated repair workflows, manuscript
writing, Word/PPT/Overleaf code, raw data and model weights are excluded.

## Figure guide

| Figure | Calculation and plotting directory |
| --- | --- |
| Main 1 | Root AIS-first network and trade-weighted dependence scripts |
| Main 2 | Root physical Arctic comparison and cost-sensitivity scripts |
| Main 3 | `58_*/`, `60_*/`, then the vector-label and dual-axis helpers in `64_*/`, `66_*/` |
| Main 4 | Dependency-guided closure, route-replay and country-rank scripts in `60_*/` |
| Main 5 | Route-aligned economic calculation and plotting in `62_*/` |
| Extended Data 1 | `rebuild_portwatch_event_appendix.py` |
| Extended Data 2 | `40_Chokepoint_Bipartite_Dependency/` |
| Extended Data 3 | `60_*/plot_auxiliary.py` |
| Supplementary 1-2 | CMIP6/window and admission sensitivities in `44_*/`, `45_*/` |
| Supplementary 3-5 | Disturbance gaps, gate dependencies and coastal-radius ranks in `60_*/` |
| Supplementary 6-7 | Economic geography and monthly output exposure in `62_*/` |
| Supplementary 8-9 | Multiagent procurement in `71_*/`, `72_*/` |

Paths abbreviated above are under `13_CMIP6_GridPath_Figures/`.
See the [full figure-to-code map](docs/final_figures.json) and
[execution order](docs/PIPELINE.md).

## Checks

Use Python 3.12:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python tools/check_release.py
python -m unittest discover -s tests -v
```

The data-free numerical fixtures check accounting and solver properties; they
are not research observations or an empirical figure reproduction.

For figure calculations, install `requirements-analysis.txt`, restore the
authorized inputs in [DATA.md](docs/DATA.md), and run the relevant entry points.
`python tools/run_stage.py --list` lists selected stages. A stage without
`--run` only checks inputs and prints commands.

## Scope and provenance

The original directory names preserve relative imports. Helper modules are
retained where final scripts import shared geometry, palettes or labels, not as
additional analysis entry points. A historical directory name containing
`Manuscript` does not mean manuscript-writing code is included.

The [source manifest](docs/source_manifest.csv) records original and release
SHA-256 hashes. Changes are limited to configurable external paths, removal of
unused report/caption/document routines, and technical wording.

Complete figure calculation still requires the study's AIS-derived graph,
monthly CMIP6 fields, trade weights and economic tables. These are not uploaded.
Some vector composition steps require earlier generated panels, as documented.
No end-to-end empirical reproduction from a data-free clone is claimed.

The primary 5% Arctic premium is a scenario setting. Coastal rank is not welfare,
structural output exposure is not measured GDP loss, and language-model
procurement responses are conditional simulations. See [METHODS.md](docs/METHODS.md).

No open-source license is granted. Source datasets and dependencies retain their
own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
