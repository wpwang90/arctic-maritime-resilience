"""Collect final numbered artwork without document or caption generation."""
import argparse
import csv
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"13_CMIP6_GridPath_Figures"
MAPPING={
    "MainFig1.pdf": ("25_MainFig1_LatestNetwork/figures/MainFig1_AISfirst_24Chokepoints.pdf",None),
    "MainFig2.pdf": ("26_MainFig2_LatestNetwork/figures/MainFig2_Monthly_Arctic_Reconnection.pdf",None),
    "MainFig3.pdf": ("66_MainFig3_DualAxis_20261106/figures/MainFig3.pdf",None),
    "MainFig4.pdf": ("60_TwoPart_Disturbance_Chokepoint_Analysis_20261101/figures/MainFig4_ODDependent_CountryRank.pdf",None),
    "MainFig5.pdf": ("62_FigureAligned_Manuscript_20261101/figures/MainFig5.pdf",None),
    "ExtendedDataFig01.pdf": ("35_PortWatch_Event_Appendix/figures/SupplementaryFig8_PortWatch_Events.pdf",None),
    "ExtendedDataFig02.pdf": ("40_Chokepoint_Bipartite_Dependency/five_panels/SuppFig9_Bipartite_All_and_FourVesselTypes.pdf",None),
    "ExtendedDataFig03.pdf": ("60_TwoPart_Disturbance_Chokepoint_Analysis_20261101/figures/SI_PartI_FourClass_Shiptype_ShareCurves.pdf",None),
    "SupplementaryFig01.pdf": ("44_CMIP6_Monthly_SSP_Audit/figures/SupplementaryFig11_CMIP6_Monthly_ice_capable.pdf",None),
    "SupplementaryFig02.pdf": ("45_AdmissionThreshold_Sensitivity/figures/SupplementaryFig15_Monthly_Threshold_Admission.pdf",None),
    "SupplementaryFig03.pdf": ("60_TwoPart_Disturbance_Chokepoint_Analysis_20261101/figures/SI_PartI_All960_DominanceCases.pdf",None),
    "SupplementaryFig04.pdf": ("60_TwoPart_Disturbance_Chokepoint_Analysis_20261101/figures/SI_PartII_OD_to_Chokepoint_Dependencies.pdf",None),
    "SupplementaryFig05.pdf": ("60_TwoPart_Disturbance_Chokepoint_Analysis_20261101/figures/SI_MainFig4_CoastalRadius_Sensitivity.pdf",0),
    "SupplementaryFig06.pdf": ("62_FigureAligned_Manuscript_20261101/figures/SI_EconomicGeography_and_NationalExposure.pdf",None),
    "SupplementaryFig07.pdf": ("62_FigureAligned_Manuscript_20261101/figures/SI_Monthly_OutputExposure.pdf",None),
    "SupplementaryFig08.pdf": ("71_Rank_Trade_Benefit_Experiment_20261111/Supplementary_Figure_8.pdf",None),
    "SupplementaryFig09.pdf": ("72_Supplier_Switch_Experiment_20261111/Supplementary_Figure_9.pdf",None),
}


def prepare_main3():
    source=BASE/"60_TwoPart_Disturbance_Chokepoint_Analysis_20261101"
    labels=BASE/"64_NetworkTerminology_20261101"
    for rel in ("figures","source_data"):
        (labels/rel).mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location("final_network_labels",labels/"update_figure_labels.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.relabel(source/"figures/MainFig3_FourDisturbance_ExtremeCases.pdf",
                   labels/"figures/MainFig3.pdf",footer_kind="main")
    for name in ("partI_monthly_mechanism_decomposition.csv","partI_extreme_case_selection.csv"):
        shutil.copy2(source/"results"/name,labels/"source_data"/name)
    script=BASE/"66_MainFig3_DualAxis_20261106/build_figure.py"
    subprocess.run([sys.executable,script.name],cwd=script.parent,check=True)


def collect(source_root,output):
    missing=[str(source_root/p) for p,_ in MAPPING.values() if not (source_root/p).is_file()]
    if missing:
        raise FileNotFoundError("Generate the required panels first:\n"+"\n".join(missing))
    from pypdf import PdfReader,PdfWriter
    output.mkdir(parents=True,exist_ok=True)
    for name,(path,page) in MAPPING.items():
        if page is None:
            shutil.copy2(source_root/path,output/name)
        else:
            writer=PdfWriter()
            writer.add_page(PdfReader(source_root/path).pages[page])
            writer.write(output/name)
        print(name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-main3",action="store_true")
    parser.add_argument("--collect",action="store_true")
    parser.add_argument("--source-root",type=Path,default=BASE)
    parser.add_argument("--output",type=Path,default=ROOT/"final_figures")
    args=parser.parse_args()
    if not args.prepare_main3 and not args.collect:
        parser.print_help()
        return
    if args.prepare_main3:
        prepare_main3()
    if args.collect:
        collect(args.source_root,args.output)


if __name__=="__main__":
    main()
