"""Export the Figure 4 aligned inputs for a four-market trade experiment."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import duckdb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT = ROOT / "62_FigureAligned_Manuscript_20261101/results"
BRIDGE = RESULT / "OD_industry_weight_bridge.parquet"
LOSSES = RESULT / "paired_od_losses.parquet"
RANKS = ROOT / "67_Nature_Manuscripts_20261106/source_data/MainFig4_country_rank_source.csv"
LABELS = ROOT / "32_NCC2023_IndirectExposure/results/industry_aggregation.csv"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    c = duckdb.connect()
    c.execute("create temp table losses as select * from read_parquet(?) where primary_case", [str(LOSSES)])
    c.execute("create temp table commodity as select * from read_parquet(?) where supported_sector_weight_usd>0 and origin_iso3<>dest_iso3", [str(BRIDGE)])
    assert c.execute("select count(*) from (select od_id from losses group by od_id having count(*)<>2)").fetchone()[0] == 0
    assert c.execute("""select count(*) from commodity b join losses r using(od_id)
         where b.shiptype<>r.shiptype or b.month<>r.month
            or b.origin_iso3<>r.origin_iso3 or b.dest_iso3<>r.dest_iso3
            or abs(b.trade_weight_usd-r.trade_weight_usd)>0.01""").fetchone()[0] == 0
    relation = c.execute("""
      with x as (
        select b.shiptype, b.month, b."group" as industry, b.origin_iso3 as seller,
            b.dest_iso3 as buyer, r.case as state,
            b.supported_sector_weight_usd/1e6 as weight,
            isfinite(r.selected_distance_km) and r.effective_delay_days<30 as available,
            r.transport_insurance_usd / (r.trade_weight_usd*30/r.days_in_month)
                + 0.003*r.effective_delay_days as wedge,
            r.case='with_arctic' and r.selected_B as arctic
        from commodity b join losses r using(od_id)
      ), totals as (
        select shiptype, industry, sum(weight) as total from x where state='without_arctic'
        group by shiptype, industry
        qualify row_number() over(partition by shiptype order by total desc, industry)=1
      )
      select x.shiptype, x.month, x.industry, seller, buyer, state, sum(weight) as reference,
             sum(case when available then weight else 0 end) as reachable,
             sum(case when available then weight*wedge else 0 end) as wedge_weighted,
             sum(case when available and arctic then weight else 0 end) as arctic_reachable
      from x join totals t on x.shiptype=t.shiptype and x.industry=t.industry
      group by all order by x.shiptype, seller, buyer, state
    """)
    names = [x[0] for x in relation.description]
    rows = relation.fetchall()
    with (HERE / "market_inputs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerows(rows)
    with LABELS.open() as f:
        labels = {int(r["group"]): r["group_label"] for r in csv.DictReader(f)}
    market_info = {}
    for ship, month, industry, *_ in rows:
        market_info[ship] = {"month": month, "industry": industry, "label": labels[industry]}
    shutil.copyfile(RANKS, HERE / "rank_inputs.csv")
    (HERE / "input_provenance.json").write_text(json.dumps({
        "source_sha256": {p.name: digest(p) for p in (BRIDGE, LOSSES, RANKS, LABELS)},
        "market_inputs_sha256": digest(HERE / "market_inputs.csv"),
        "rank_inputs_sha256": digest(HERE / "rank_inputs.csv"),
        "markets": market_info, "group_selection": "Largest supported international trade value within each primary vessel-month",
        "delay_wedge_per_day": 0.003, "availability_rule": "Finite route and additional delay <30 days",
    }, indent=2))
    print(market_info, len(rows))


if __name__ == "__main__":
    main()
