"""Check stage inputs and run only explicitly requested calculations."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
B = "13_CMIP6_GridPath_Figures/"
S24 = B+"71_Rank_Trade_Benefit_Experiment_20261111/"
S25 = B+"72_Supplier_Switch_Experiment_20261111/"
STAGES = {
    "climate": {
        "requires": ["12_CMIP6_Download/data/cmip6_future_model_inventory.csv",
                     B+"data/public/naturalearth_10m/ne_10m_land.shp"],
        "commands": [[B+"repair_cmip6_monthly_routes.py"]],
    },
    "disturbance": {
        "requires": [B+"data/physical_arctic_embedding/oxmartrans_1390/physical_edges.parquet",
                     B+"data/chokepoints24/registry.csv",
                     B+"28_MainFig3_FiveClosures/month_01/scenario_metrics.parquet"],
        "commands": [[B+"58_FourClass_Threshold_Analysis_20261101/calculate_thresholds.py"]],
    },
    "dependency-closures": {
        "requires": [B+"58_FourClass_Threshold_Analysis_20261101/results/month_01/od_states.npz",
                     B+"28_MainFig3_FiveClosures/scenarios.json"],
        "commands": [[B+"60_TwoPart_Disturbance_Chokepoint_Analysis_20261101/analyze_two_parts.py"],
                     [B+"60_TwoPart_Disturbance_Chokepoint_Analysis_20261101/select_dependent_closures.py"]],
    },
    "economics": {
        "requires": [B+"60_TwoPart_Disturbance_Chokepoint_Analysis_20261101/results/partII_mainfig4_fixed_month_optimal_closures.csv",
                     B+"32_NCC2023_IndirectExposure/results/mrio2024_aggregated.npz"],
        "commands": [[B+"62_FigureAligned_Manuscript_20261101/recalculate_economics.py"]],
    },
    "supplier-switch": {
        "requires": [S24+"market_inputs.csv",S24+"rank_inputs.csv",S24+"input_provenance.json",
                     S24+"policy_logits_qwen3.jsonl.gz",S24+"policy_logits_qwen25.jsonl.gz",
                     S24+"policy_provenance_qwen3.json",S24+"policy_provenance_qwen25.json",
                     S25+"buyer_logits_qwen3.jsonl.gz",S25+"buyer_logits_qwen25.jsonl.gz",
                     S25+"buyer_provenance_qwen3.json",S25+"buyer_provenance_qwen25.json"],
        "commands": [[S25+"analyse_switch.py"],[S25+"plot_switch.py"]],
    },
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list",action="store_true")
    parser.add_argument("--stage",choices=STAGES)
    parser.add_argument("--run",action="store_true",help="Execute, rather than only check prerequisites")
    args=parser.parse_args()
    if args.list:
        print("\n".join(STAGES))
        return 0
    if args.stage is None:
        parser.error("Select --stage or --list")
    spec=STAGES[args.stage]
    print("Minimum input check; additional source-specific inputs may be needed.")
    missing=[p for p in spec["requires"] if not (ROOT/p).is_file()]
    for cmd in spec["commands"]:
        path=ROOT/cmd[0]
        print(f"cwd: {path.parent.relative_to(ROOT)}")
        print(f"command: python {path.name}"+"".join(" "+s for s in cmd[1:]))
    if missing:
        print("\nMissing inputs:",file=sys.stderr)
        print("\n".join(missing),file=sys.stderr)
        return 2
    if args.run:
        for cmd in spec["commands"]:
            path=ROOT/cmd[0]
            subprocess.run([sys.executable,path.name,*cmd[1:]],cwd=path.parent,check=True)
    else:
        print("Inputs found. Add --run to execute.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
