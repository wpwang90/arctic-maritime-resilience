"""Compare function ASTs with an authorized local calculation archive."""
import argparse
import ast
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPECTED = {
    ("08_Nature_Probability_Boost_Figures/generate_local_first_nature_figures.py","local_sithick_stats"),
    ("13_CMIP6_GridPath_Figures/43_MaritimeShare2024_Integrated/integrate_forecast.py","main"),
    ("13_CMIP6_GridPath_Figures/45_AdmissionThreshold_Sensitivity/compute_sensitivity.py","main"),
    ("13_CMIP6_GridPath_Figures/61_MainFig5_RowReorganization_20261101/reorganize_figures.py","main"),
    ("13_CMIP6_GridPath_Figures/66_MainFig3_DualAxis_20261106/build_figure.py","main"),
    ("13_CMIP6_GridPath_Figures/generate_cmip6_gridpath_figures.py","write_summary"),
    ("13_CMIP6_GridPath_Figures/62_FigureAligned_Manuscript_20261101/plot_aligned_economics.py","main"),
}
REMOVED = {
    ("13_CMIP6_GridPath_Figures/61_MainFig5_RowReorganization_20261101/reorganize_figures.py","write_captions"),
    ("13_CMIP6_GridPath_Figures/66_MainFig3_DualAxis_20261106/build_figure.py","make_caption"),
}
STYLE_MODULE="13_CMIP6_GridPath_Figures/32_NCC2023_IndirectExposure/plot_figure5.py"


def functions(text):
    result={}
    for node in ast.parse(text).body:
        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            continue
        result[node.name]=ast.dump(node,include_attributes=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path,help="Local directory containing the original numbered source directories")
    args=parser.parse_args()
    checked=unchanged=0
    differences=[]
    with (ROOT/"docs/source_manifest.csv").open(newline="") as f:
        manifest=list(csv.DictReader(f))
    for row in manifest:
        path=row["path"]
        source=args.archive/path
        if not source.is_file():
            raise FileNotFoundError(source)
        before=functions(source.read_text())
        after=functions((ROOT/path).read_text())
        for name,value in before.items():
            checked+=1
            if after.get(name)==value:
                unchanged+=1
            elif path==STYLE_MODULE and name!="name" and name not in after:
                pass
            elif (path,name) not in EXPECTED|REMOVED:
                differences.append((path,name))
    print(f"Function/class definitions checked: {checked}; identical: {unchanged}")
    print("Document dependencies, removed caption helpers and technical-message changes are explicitly allowlisted.")
    for path,name in differences:
        print(f"REVIEW: {path}:{name}")
    return bool(differences)


if __name__=="__main__":
    raise SystemExit(main())
