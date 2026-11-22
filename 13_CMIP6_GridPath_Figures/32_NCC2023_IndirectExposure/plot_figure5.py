"""Shared colours, industry groups and labels for the final economic panels."""
import country_converter as coco
from matplotlib.colors import LinearSegmentedColormap

DISPLAY_GROUPS = [
    ('Agri. & food',[0,1,2,7]), ('Fossil energy',[3,4,5,11]),
    ('Ores & minerals',[6,14]), ('Chem. & plastics',[12,13]),
    ('Other manuf.',[8,9,10,19]), ('Metals',[15]), ('Equipment',[16,17,18]),
    ('Utilities',[20]), ('Construction',[21]), ('Trade & hotels',[22,23]),
    ('Transport',[24,25,26,27]), ('Other services',[28,29,30,31,32,33])]
PARTNER_COLORS = ['#2E7296','#D98A4A','#AF6572','#77A0B4','#72997E',
                  '#777FA5','#BEA062','#5FA6A1','#C58C7A','#A5B8C3','#D5D9DC']
ABSOLUTE_CMAP = LinearSegmentedColormap.from_list('exposure_blue',
    ['#F2F7FA','#D9E9F1','#A7CBDC','#69A4C0','#347FA3','#184D70'])
RELATIVE_CMAP = LinearSegmentedColormap.from_list('relief_orange',
    ['#FFF7EC','#FEE4C1','#F7BF84','#EC995E','#D4773C','#A64F26'])


def name(region):
    overrides = {'RU':'Russia','US':'United States','GB':'United Kingdom','KR':'South Korea',
                 'CZ':'Czechia','TR':'Turkiye','WA':'Rest of Asia-Pacific','WL':'Rest of Americas',
                 'WE':'Rest of Europe','WF':'Rest of Africa','WM':'Rest of Middle East'}
    return overrides.get(region) or coco.convert(names=region,src='ISO2',to='name_short')
