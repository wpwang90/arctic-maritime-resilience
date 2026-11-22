# Third party sources and rights

This release does not vendor third-party source archives, model weights or
research datasets. Public accessibility is not permission to redistribute a
dataset. Users must check the terms of the version they obtain.

| Source | Use | Upstream reference |
| --- | --- | --- |
| CMIP6 / ESGF | Sea-ice concentration, thickness or mass; model-specific calendars | https://esgf.llnl.gov/ |
| Natural Earth | Land and country boundaries | https://www.naturalearthdata.com/ |
| IMF PortWatch | Daily chokepoint event observations | https://portwatch.imf.org/ |
| CEPII BACI | Directed country-commodity trade values | https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37 |
| UNCTAD / World Bank | Mode-of-transport fractions | https://unctadstat.unctad.org/EN/TransportCost.html |
| OECD ITIC | International transport and insurance margins | https://www.oecd.org/en/data/datasets/international-transport-and-insurance-costs-of-merchandise-trade-itic.html |
| EXIOBASE | Input-output structure and concordances | https://doi.org/10.5281/zenodo.20051562 |
| Maritime chokepoint study | Reference port network and comparison diagnostics | https://doi.org/10.5281/zenodo.19663059 |
| USACE and other navigation authorities | Documented waterway geometry and access checks | Refer to source URLs in the relevant collection scripts |
| Qwen / MLX community | Two local inference checkpoints | https://huggingface.co/mlx-community |

Representative AIS trajectories and port-call records were supplied separately;
they are not distributed here. Earlier scripts may also expect GTAP, shipping
lane, LNG or regional AIS sources. Their acquisition pointers remain in the
source code; their use is not implied to be unrestricted.

Selected methodological references:

- Eyring et al. (2016), CMIP6 experimental design.
  https://doi.org/10.5194/gmd-9-1937-2016
- Verschuur et al. (2023), systemic risks from port disruptions.
  https://doi.org/10.1038/s41558-023-01754-w
- Verschuur et al. (2025), systemic impacts of maritime chokepoint disruptions.
  https://doi.org/10.1038/s41467-025-65403-w
- Akata et al. (2025), repeated games with language models.
  https://doi.org/10.1038/s41562-025-02172-y
- Antras, Fort and Tintelnot (2017), global sourcing.
  https://doi.org/10.1257/aer.20141685
- Barrot and Sauvagnat (2016), input specificity.
  https://doi.org/10.1093/qje/qjw018
