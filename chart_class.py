from re import T
from matplotlib import axes
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from matplotlib.gridspec import GridSpec
import os
from numpy.core.arrayprint import str_format
from numpy.lib.function_base import iterable
import pandas as pd
from typing import Union
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import textwrap
import math
import matplotlib.dates as mdates

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["font.serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams.update({"font.size": 16})
mpl.rcParams["hatch.linewidth"] = 0.5
mpl.rcParams["hatch.color"] = "grey"

# sns.set_theme(style="whitegrid")
MYFONT = fm.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
NUM_FONT = {"fontname": "Calibri"}

COLOR_DICT = {
    "缬沙坦": "#44546A",
    "厄贝沙坦": "#6F8DB9",
    "氯沙坦": "#BD2843",
    "替米沙坦": "#ED94B6",
    "坎地沙坦": "#FAA53A",
    "奥美沙坦酯": "#2B9B33",
    "阿利沙坦酯": "Deepskyblue",
    "培哚普利": "Saddlebrown",
    "贝那普利": "Purple",
    "高血压": "#44546A",
    "冠心病": "#6F8DB9",
    "糖尿病": "#ED94B6",
    "血脂异常": "#BD2843",
    "慢性肾病": "#FAA53A",
    "卒中": "#2B9B33",
    "高尿酸": "Deepskyblue",
    "心力衰竭": "Saddlebrown",
    "无以上适应症": "Purple",
    "单纯高血压": "#44546A",
    "高血压+冠心病": "#6F8DB9",
    "高血压+糖尿病": "#ED94B6",
    "高血压+血脂异常": "#BD2843",
    "高血压+慢性肾病": "#FAA53A",
    "高血压+卒中": "#2B9B33",
    "高血压+高尿酸": "Deepskyblue",
    "高血压+心力衰竭": "Saddlebrown",
    "心内科": "#44546A",
    "肾内科": "#6F8DB9",
    "老干科": "#BD2843",
    "神内科": "#ED94B6",
    "内分泌科": "#FAA53A",
    "普内科": "#2B9B33",
    "其他": "Deepskyblue",
    #     "销售基恩（万元）": "teal",
    #     "标准盒数（万盒）": "crimson",
    #     "单价": "navy",
    #     "心内科": "teal",
    #     "肾内科": "crimson",
    #     "老干科": "navy",
    #     "神内科": "darkorange",
    #     "内分泌科": "darkgreen",
    #     "普内科": "saddlebrown",
    #     "其他科室": "grey",
    #     "等级医院": "navy",
    #     "社区医院": "crimson",
    #     "D9-D10": "navy",
    #     "D6-D8": "royalblue",
    #     "D3-D5": "dodgerblue",
    #     "D1-D2": "deepskyblue",
    #     "旗舰社区": "crimson",
    #     "普通社区": "pink",
    #     "西格列汀": "navy",
    #     "沙格列汀": "teal",
    #     "维格列汀": "crimson",
    #     "利格列汀": "darkorange",
    #     "阿格列汀": "darkgreen",
    #     "阿卡波糖片剂": "navy",
    #     "阿卡波糖咀嚼片": "deepskyblue",
    #     "伏格列波糖": "teal",
    #     "米格列醇": "crimson",
    #     "卡格列净": "navy",
    #     "恩格列净": "crimson",
    #     "达格列净": "teal",
    #     "埃格列净": "darkorange",
    #     "比索洛尔": "navy",
    #     "美托洛尔缓释剂型": "crimson",
    #     "美托洛尔常释剂型": "teal",
    #     "阿罗洛尔": "darkorange",
    #     "地舒单抗": "deepskyblue",
    #     "带量品种": "crimson",
    #     "非带量品种": "teal",
    #     "拜阿司匹灵": "navy",
    #     "波立维": "crimson",
    #     "泰嘉": "darkgreen",
    #     "泰仪": "olivedrab",
    #     "帅信": "darkgreen",
    #     "帅泰": "olivedrab",
    #     "帅信/帅泰": "saddlebrown",
    #     "硫酸氢氯吡格雷片": "saddlebrown",
    #     "倍林达": "darkorange",
    #     "阿司匹林": "navy",
    #     "氯吡格雷": "teal",
    #     "吲哚布芬": "crimson",
    #     "替格瑞洛": "darkorange",
    #     "倍利舒": "purple",
    #     "美洛林": "deepskyblue",
    #     "国产阿司匹林": "grey",
    #     "华东区": "navy",
    #     "华西区": "crimson",
    #     "华南区": "teal",
    #     "华北区": "darkgreen",
    #     "华中区": "darkorange",
    #     "一线城市": "navy",
    #     "二线城市": "crimson",
    #     "三线城市": "teal",
    #     "四线城市": "darkgreen",
    #     "五线城市": "darkorange",
    #     "25MG10片装": "darkgreen",
    #     "25MG20片装": "olivedrab",
    #     "75MG7片装": "darkorange",
    #     "10MG": "slateblue",
    #     "15MG": "rebeccapurple",
    #     "20MG": "indigo",
    #     "吸入性糖皮质激素(ICS)": "navy",
    #     "短效β2受体激动剂(SABA)": "crimson",
    #     "长效β2受体激动剂(LABA)": "tomato",
    #     "抗白三烯类药物(LTRA)": "teal",
    #     "黄嘌呤类": "darkorange",
    #     "长效抗胆碱剂(LAMA)": "darkgreen",
    #     "短效抗胆碱剂(SAMA)": "olivedrab",
    #     "LABA+ICS固定复方制剂": "purple",
    #     "SAMA+SABA固定复方制剂": "deepskyblue",
    #     "非类固醇类呼吸道消炎药": "saddlebrown",
    #     "其他": "grey",
    #     "其他品种": "grey",
    #     "氨氯地平": "teal",
    #     "硝苯地平": "crimson",
    #     "左旋氨氯地平": "darkorange",
    #     "非洛地平": "navy",
    #     "贝尼地平": "darkgreen",
    #     "布地奈德": "navy",
    #     "丙酸倍氯米松": "crimson",
    #     "丙酸氟替卡松": "darkorange",
    #     "环索奈德": "darkgreen",
    #     "异丙肾上腺素": "grey",
    #     "特布他林": "navy",
    #     "沙丁胺醇": "crimson",
    #     "丙卡特罗": "navy",
    #     "福莫特罗": "crimson",
    #     "班布特罗": "darkorange",
    #     "妥洛特罗": "teal",
    #     "环仑特罗": "darkgreen",
    #     "茚达特罗": "purple",
    #     "孟鲁司特": "navy",
    #     "普仑司特": "crimson",
    #     "多索茶碱": "navy",
    #     "茶碱": "crimson",
    #     "二羟丙茶碱": "tomato",
    #     "氨茶碱": "darkorange",
    #     "复方胆氨": "darkgreen",
    #     "二羟丙茶碱氯化钠": "teal",
    #     "复方妥英麻黄茶碱": "olivedrab",
    #     "复方茶碱麻黄碱": "purple",
    #     "茶碱,盐酸甲麻黄碱,暴马子浸膏": "saddlebrown",
    #     "ARB": "teal",
    #     "ACEI": "crimson",
    #     "CCB": "navy",
    #     "Beta Blocker": "olivedrab",
    #     "Diuretics": "darkgreen",
    #     "RAAS FDC": "darkorange",
    #     "厄贝沙坦": "navy",
    #     "缬沙坦": "crimson",
    #     "缬沙坦,氨氯地平": "maroon",
    #     "缬沙坦氨氯地平": "maroon",
    #     "贝那普利,氨氯地平": "salmon",
    #     "贝那普利氨氯地平": "salmon",
    #     "厄贝沙坦,氢氯噻嗪": "royalblue",
    #     "厄贝沙坦氢氯噻嗪": "royalblue",
    #     "氯沙坦氢氯噻嗪": "gold",
    #     "氯沙坦": "darkorange",
    #     "氯沙坦钾": "darkorange",
    #     "替米沙坦": "darkgreen",
    #     "奥美沙坦": "deepskyblue",
    #     "奥美沙坦（BW.）": "deepskyblue",
    #     "坎地沙坦": "olivedrab",
    #     "阿利沙坦": "teal",
    #     "代文": "navy",
    #     "代文（NVR）": "navy",
    #     "代文（NVU）": "navy",
    #     "洛汀新": "olivedrab",
    #     "洛汀新（NBJ）": "olivedrab",
    #     "洛汀新（NVU）": "olivedrab",
    #     "科素亚": "darkorange",
    #     "科素亚（MHU）": "darkorange",
    #     "科素亚（MSG）": "darkorange",
    #     "安博维": "crimson",
    #     "安博维（SG9）": "crimson",
    #     "安博维（SA9）": "crimson",
    #     "雅施达": "saddlebrown",
    #     "雅施达（TSV）": "saddlebrown",
    #     "雅施达（SVU）": "saddlebrown",
    #     "百安新": "darkgreen",
    #     "倍博特": "maroon",
    #     "傲坦": "deepskyblue",
    #     "美卡素": "purple",
    #     "美卡素（B.I）": "purple",
    #     "必洛斯": "darkgreen",
    #     "信立坦": "purple",
    #     "信立坦（SI6）": "teal",
    #     "倍悦": "purple",
    #     "缓宁": "gold",
    #     "安来": "gold",
    #     "安来（ZJ5）": "gold",
    #     "搏力高": "mediumslateblue",
    #     "搏力高（ZYG）": "mediumslateblue",
    #     "平欣": "mediumslateblue",
    #     "依苏": "coral",
    #     "科苏": "darkgreen",
    #     "穗悦": "olivedrab",
    #     "迪之雅": "olive",
    #     "伊达力": "olive",
    #     "伊达力（ZUP）": "olive",
    #     "伊达力（ZHI）": "olive",
    #     "兰沙": "c",
    #     "倍怡": "pink",
    #     "倍怡（ZJ5）": "pink",
    #     "吉加": "orchid",
    #     "吉加（JSH）": "orchid",
    #     "吉加（H9R）": "orchid",
    #     "卡托普利片": "grey",
    #     "华法林": "darkorange",
    #     "利伐沙班": "rebeccapurple",
    #     "阿哌沙班": "darkgreen",
    #     "依度沙班": "deepskyblue",
    #     "达比加群酯": "dodgerblue",
    #     "肝素": "crimson",
    #     "比伐芦定": "teal",
    #     "万脉舒（H2C）": "navy",
    #     "克赛（AVS）": "crimson",
    #     "立迈青（AHK）": "darkorange",
    #     "泰加宁（SI6）": "teal",
    #     "希弗全（A1L）": "darkgreen",
    #     "尤尼舒（J/G）": "olivedrab",
    #     "速避凝（A3N）": "purple",
    #     "赛博利（S1E）": "gold",
    #     "注射用那屈肝素钙（DG+）": "deepskyblue",
    #     "赛倍畅（JJY）": "saddlebrown",
    #     "齐征（QLU）": "pink",
    #     "普洛静（GTW）": "pink",
    #     "法安明（PHA）": "rebeccapurple",
    #     "那屈肝素钙注射液（JJY）": "saddlebrown",
    #     "泰加宁": "teal",
    #     "泽朗": "crimson",
    #     "泰加宁\n深圳信立泰药业股份有限公司": "teal",
    #     "泽朗\n江苏豪森药业集团有限公司": "crimson",
    #     "比伐芦定\n海南双成药业股份有限公司": "darkorange",
    #     "比伐芦定\n齐鲁制药集团": "deepskyblue",
    #     "阿托伐他汀": "navy",
    #     "瑞舒伐他汀": "crimson",
    #     "匹伐他汀": "teal",
    #     "氟伐他汀": "darkgreen",
    #     "普伐他汀": "olivedrab",
    #     "辛伐他汀": "darkorange",
    #     "洛伐他汀": "saddlebrown",
    #     "立普妥": "navy",
    #     "可定": "crimson",
    #     "阿乐": "darkorange",
    #     "尤佳": "teal",
    #     "瑞旨": "darkgreen",
    #     "京诺": "olivedrab",
    #     "托妥": "purple",
    #     "赛博利": "gold",
    #     "优力平": "gold",
    #     "注射用那屈肝素钙": "deepskyblue",
    #     "齐征": "pink",
    #     "立普妥（PFZ）": "navy",
    #     "立普妥（VI/）": "navy",
    #     "信立明（SI6）": "teal",
    #     "信立明": "teal",
    #     "可定（AZN）": "crimson",
    #     "可定（A5Z）": "crimson",
    #     "阿乐（SDS）": "darkorange",
    #     "尤佳（TOF）": "darkgreen",
    #     "瑞旨（S6B）": "olivedrab",
    #     "瑞旨（S7O）": "olivedrab",
    #     "托妥（NJ2）": "purple",
    #     "京诺（ZXJ）": "pink",
    #     "阿托伐他汀钙分散片（G6B）": "gold",
    #     "冠爽（BJP）": "crimson",
    #     "冠爽（SHN）": "crimson",
    #     "阿托伐他汀（ZLU）": "deepskyblue",
    #     "美百乐镇（DSC）": "coral",
    #     "美百乐镇（DCG）": "coral",
    #     "富利他之（ZHI）": "rebeccapurple",
    #     "邦之（JBI）": "saddlebrown",
    #     "邦之（SFO）": "saddlebrown",
    #     "舒降之（MHU）": "dodgerblue",
    #     "来适可XL（NVR）": "orchid",
    #     "来适可XL（NVU）": "orchid",
    #     "辛可（GXN）": "olive",
    #     "优力平（ZLU）": "saddlebrown",
    #     "阿托伐他汀（GI2）": "olive",
    #     "阿托伐他汀（FXG）": "olive",
    #     "瑞舒伐他汀钙片（ZHI）": "orchid",
    #     "瑞舒伐他汀（C2T）": "dodgerblue",
    #     "瑞舒伐他汀（NJ2）": "dodgerblue",
    #     "瑞舒伐他汀（LEK）": "olive",
    #     "瑞舒伐他汀（NVU）": "olive",
    #     "阿托伐他汀（HNQ）": "deepskyblue",
    #     "阿托伐他汀（QIL）": "deepskyblue",
    #     "阿托伐他汀（FJ.）": "pink",
    #     "京必舒新（ZXJ）": "darkorange",
    #     "冠爽": "navy",
    #     "邦之": "crimson",
    #     "力清之": "darkorange",
    #     "京可新": "darkgreen",
    #     "匹伐他汀钙片": "olivedrab",
    #     "京可新（ZXJ）": "darkgreen",
    #     "力清之（KW.）": "darkorange",
    #     "匹伐他汀（JN.）": "dodgerblue",
    #     "匹伐他汀钙片（S1Q）": "navy",
    #     "匹伐他汀钙片（SI6）": "olivedrab",
    #     "G03J0\n选择性雌激素\n受体调节剂": "navy",
    #     "SERM": "navy",
    #     "H04A0\n降钙素": "crimson",
    #     "降钙素": "crimson",
    #     "H04E0\n甲状旁腺激素\n及类似物": "darkorange",
    #     "PTH": "darkorange",
    #     "M05B3\n治疗骨质疏松\n和骨钙失调\n的二膦酸盐类": "darkgreen",
    #     "双膦酸盐类": "darkgreen",
    #     "唑来膦酸": "navy",
    #     "唑来膦酸（SHR）": "pink",
    #     "唑来膦酸（YAZ）": "darkorange",
    #     "达芬盖（SZA）": "dodgerblue",
    #     "伊疏（S.I）": "saddlebrown",
    #     "普罗力（AAI）": "pink",
    #     "普罗力": "pink",
    #     "唑来膦酸注射液": "navy",
    #     "鲑鱼降钙素": "crimson",
    #     "鲑降钙素": "crimson",
    #     "阿仑膦酸钠": "darkorange",
    #     "阿仑膦酸钠,维生素D3": "darkgreen",
    #     "阿仑膦酸钠维生素D3": "darkgreen",
    #     "依降钙素": "olivedrab",
    #     "利塞膦酸钠": "purple",
    #     "特立帕肽": "pink",
    #     "雷洛昔芬": "gold",
    #     "依替膦酸二钠": "teal",
    #     "密固达": "navy",
    #     "密固达（NVR）": "navy",
    #     "密固达（NVU）": "navy",
    #     "欣复泰": "teal",
    #     "欣复泰（XIL）": "teal",
    #     "依固": "crimson",
    #     "依固（CTA）": "crimson",
    #     "依固（C2T）": "crimson",
    #     "密盖息": "darkorange",
    #     "密盖息（NVR）": "darkorange",
    #     "密盖息（NVU）": "darkorange",
    #     "福美加": "darkgreen",
    #     "福美加（MSD）": "darkgreen",
    #     "福美加（MSG）": "darkgreen",
    #     "福善美": "olivedrab",
    #     "福善美（MSD）": "olivedrab",
    #     "福善美（MSG）": "olivedrab",
    #     "金尔力": "purple",
    #     "金尔力（B-Y）": "purple",
    #     "斯迪诺": "saddlebrown",
    #     "利塞膦酸钠片": "gold",
    #     "利塞膦酸钠片（YKJ）": "gold",
    #     "益盖宁": "teal",
    #     "斯迪诺（S-Y）": "teal",
    #     "斯迪诺（LU6）": "teal",
    #     "复泰奥": "deepskyblue",
    #     "复泰奥（LYG）": "deepskyblue",
    #     "珍固": "crimson",
    #     "珍固（S60）": "crimson",
    #     "益盖宁（ASC）": "pink",
    #     "唑来膦酸注射液（SK4）": "saddlebrown",
    #     "唑来膦酸注射液（KEU）": "saddlebrown",
    #     "丙戊酸钠": "navy",
    #     "左乙拉西坦": "crimson",
    #     "奥卡西平": "darkorange",
    #     "硫酸镁": "darkgreen",
    #     "普瑞巴林": "olivedrab",
    #     "拉莫三嗪": "purple",
    #     "苯巴比妥": "pink",
    #     "加巴喷丁": "gold",
    #     "托吡酯": "teal",
    #     "氯硝西泮": "deepskyblue",
    #     "开浦兰（UCB）": "navy",
    #     "左乙拉西坦片（CQU）": "crimson",
    #     "左乙拉西坦片（ZXJ）": "darkorange",
    #     "左乙拉西坦（SI6）": "teal",
    #     "左乙拉西坦（ZJO）": "darkgreen",
    #     "依那普利": "saddlebrown",
    #     "依那普利（JJJ）": "saddlebrown",
    #     "依那普利（YAZ）": "saddlebrown",
    #     "依那普利拉": "olivedrab",
    #     "卡托普利": "darkorange",
    #     "咪达普利": "coral",
    #     "喹那普利": "saddlebrown",
    #     "培哚普利": "purple",
    #     "福辛普利": "teal",
    #     "贝那普利": "coral",
    #     "赖诺普利": "pink",
    #     "雷米普利": "deepskyblue",
    #     "傲坦（DSC）": "navy",
    #     "傲坦（DCG）": "navy",
    #     "兰沙（B4W）": "crimson",
    #     "希佳（NJ2）": "darkorange",
    #     "希佳（C2T）": "darkorange",
    #     "奥美沙坦酯片（S6N）": "darkgreen",
    #     "天泉乐宁（FTQ）": "olivedrab",
    #     "奥美沙坦（FTQ）": "deepskyblue",
    #     "奥美沙坦（GY0）": "saddlebrown",
    #     "奥美沙坦（SI6）": "teal",
    #     "250MG": "navy",
    #     "500MG": "crimson",
    #     "恩存": "deepskyblue",
    #     "托平（ZHT）": "deepskyblue",
    #     "诺欣妥": "crimson",
    #     "口服片剂": "navy",
    #     "口服溶液": "deepskyblue",
    #     "注射剂": "crimson",
    #     "A+C": "navy",
    #     "A+D": "darkorange",
    #     "B03A 补血药，铁剂": "navy",
    #     "B03C 红细胞生成素": "crimson",
    #     "B03D HIF-PH抑制剂": "teal",
    #     "中标仿制": "crimson",
    #     "未中标原研": "navy",
    #     "未中标仿制": "dodgerblue",
}

COLOR_LIST = [
    "#44546A",
    "#6F8DB9",
    "#BD2843",
    "#ED94B6",
    "#FAA53A",
    "#2B9B33",
    "Deepskyblue",
    "Saddlebrown",
    "Purple",
    "Olivedrab",
    "Pink",
]

# COLOR_LIST = [
#     "teal",
#     "crimson",
#     "navy",
#     "tomato",
#     "darkorange",
#     "darkgreen",
#     "olivedrab",
#     "purple",
#     "deepskyblue",
#     "saddlebrown",
#     "grey",
#     "cornflowerblue",
#     "magenta",
#     "teal",
#     "crimson",
#     "navy",
#     "tomato",
#     "darkorange",
#     "darkgreen",
#     "olivedrab",
#     "purple",
#     "deepskyblue",
#     "saddlebrown",
#     "grey",
#     "cornflowerblue",
#     "magenta",
# ]


class UnequalDataGridError(Exception):
    def __init__(self, message):
        super().__init__(message)


def data_to_list(data):
    if isinstance(data, dict):
        list_df = []
        for k, v in data.items():
            list_df.append(v)
    elif isinstance(data, pd.DataFrame):
        list_df = [data]
    elif isinstance(data, pd.Series):
        list_df = [data.to_frame()]
    elif isinstance(data, tuple):
        list_df = list(data)
    elif isinstance(data, list):
        list_df = data
    else:
        list_df = data

    return list_df


def check_data_with_axes(data: list, axes: axes):
    if len(data) != len(axes):
        message = "Got %s pieces of data, while %s axes existed." % (
            len(data),
            len(axes),
        )
        raise UnequalDataGridError(message)


class GridFigure(Figure):
    """
    一个matplotlib图表基本类，主要实现:
    数据预处理，
    grid,
    宽高设置，
    字体大小，
    总标题
    保存
    """

    def __init__(
        self,
        data,  # 原始数
        savepath: str = "./plots/",  # 保存位置
        width: int = 15,  # 宽
        height: int = 6,  # 高
        fontsize: int = 14,  # 字体大小
        gs: GridSpec = None,  # GridSpec
        fmt: list = [",.0f"],  # 每个grid的数字格式
        style: dict = None,  # 风格字典
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.savepath = savepath
        self.width = width
        self.height = height
        self.fontsize = fontsize
        self.gs = gs
        self.fmt = ["{:%s}" % f for f in fmt]
        self.style = style

        # 所有数据处理成列表格式
        self.data = data_to_list(data)

        # 宽高
        self.set_size_inches(self.width, self.height)

        # Grid
        if gs is not None:
            for axes in gs:
                ax = self.add_subplot(axes)
        else:
            ax = self.add_subplot(111)

        # 检查grid大小和数据是否匹配
        check_data_with_axes(self.data, self.axes)

        # 检查grid大小与数字格式是否匹配
        check_data_with_axes(self.fmt, self.axes)

    def set_default_style(self):
        if self.style is not None:
            # 总标题
            if "title" in self.style:
                self.suptitle(self.style["title"], fontsize=self.fontsize * 1.5)

            if "ytitle" in self.style:
                self.supylabel(self.style["ytitle"])

            for i, ax in enumerate(self.axes):
                ax.tick_params(axis="x", labelsize=self.fontsize)  # 设置x轴刻度标签字体大小
                ax.tick_params(axis="y", labelsize=self.fontsize)  # 设置x轴刻度标签字体大小

                # y轴标签
                # yticklabels = [
                #     label.get_text().split("（")[0] for label in ax.get_yticklabels()
                # ]  # 去除y轴标签括号内内容
                # ax.set_yticklabels(yticklabels)

                # 添加grid标题
                if "gs_title" in self.style:
                    check_data_with_axes(self.style["gs_title"], self.axes)
                    try:
                        ax.set_title(self.style["gs_title"][i], fontsize=self.fontsize)
                    except:
                        continue
                    # box = ax.get_position()
                    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])

                # 旋转x轴标签
                if "xlabel_rotation" in self.style:
                    ax.tick_params(
                        axis="x", labelrotation=self.style["xlabel_rotation"]
                    )

                # 旋转y轴标签
                if "ylabel_rotation" in self.style:
                    ax.tick_params(
                        axis="y", labelrotation=self.style["ylabel_rotation"]
                    )

                # 去除x轴ticks
                if "remove_xticks" in self.style:
                    if self.style["remove_xticks"]:
                        ax.get_xaxis().set_ticks([])

                # 去除y轴ticks
                if "remove_yticks" in self.style:
                    if self.style["remove_yticks"]:
                        ax.get_yaxis().set_ticks([])

                # 添加x轴标签
                if "xlabel" in self.style:
                    ax.set_xlabel(self.style["xlabel"], fontsize=self.fontsize)

                # 添加y轴标签
                if "ylabel" in self.style:
                    ax.set_ylabel(self.style["ylabel"], fontsize=self.fontsize)

                # 多个子图情况下只显示最下方图片的x轴label
                if "last_xticks_only" in self.style:
                    gs = self.axes[0].get_gridspec()
                    ncols = gs.ncols

                    if self.style["last_xticks_only"]:
                        if i < len(self.axes) - ncols:
                            ax.get_xaxis().set_ticks([])

                # 多个子图情况下只显示最左边图片的x轴label
                if "first_yticks_only" in self.style:
                    gs = self.axes[0].get_gridspec()
                    ncols = gs.ncols

                    if self.style["first_yticks_only"]:
                        if (i % ncols) != 0:
                            ax.get_yaxis().set_ticks([])

                # 隐藏上/右边框
                if "hide_top_right_spines" in self.style:
                    if self.style["hide_top_right_spines"]:
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        ax.yaxis.set_ticks_position("left")
                        ax.xaxis.set_ticks_position("bottom")

                # # x轴显示lim
                # if "xlim" in self.style:
                #     ax.set_xlim(self.style["xlim"][i][0], self.style["xlim"][i][1])

                # y轴显示lim，如果有多个y轴需要注意传参的个数
                if "ylim" in self.style:
                    ax.set_ylim(self.style["ylim"][i][0], self.style["ylim"][i][1])
                    try:
                        ax2 = ax.get_shared_x_axes().get_siblings(ax)[0]
                    except:
                        pass
                    if ax2 is not None:
                        ax2.set_ylim(self.style["ylim"][i][0], self.style["ylim"][i][1])

                # # 次坐标y轴显示lim
                # if "y2lim" in self.style:
                #     ax2 = ax.get_shared_x_axes().get_siblings(ax)[0]
                #     ax2.set_ylim(self.style["y2lim"][i][0], self.style["y2lim"][i][1])

                # 主网格线
                if "major_grid" in self.style:
                    ax.grid(
                        which="major",
                        color=self.style["major_grid"],
                        axis="both",
                        linestyle="-",
                        linewidth=0.5,
                    )

                # 次网格线
                if "minor_grid" in self.style:
                    plt.minorticks_on()
                    ax.grid(
                        which="minor",
                        color=self.style["minor_grid"],
                        axis="both",
                        linestyle=":",
                        linewidth=0.2,
                    )

    def save(self):

        # 设置一些基本格式
        self.set_default_style()

        # 保存
        if os.path.exists(self.savepath) is False:
            os.makedirs(self.savepath)

        path = "%s%s.png" % (
            self.savepath,
            "test"
            if "title" not in self.style
            else self.style["title"].replace("/", "_"),
        )
        self.savefig(
            path, format="png", bbox_inches="tight", transparent=True, dpi=600,
        )
        print(path + " has been saved...")

        # Close
        plt.clf()
        plt.cla()
        plt.close()

        return path


# 继承基本类，饼图类
class PlotPie(GridFigure):
    def plot(self, donut: bool = True, donut_title: str = ""):
        for j, ax in enumerate(self.axes):
            df = self.data[j]

            # Prepare the white center circle for Donat shape
            my_circle = plt.Circle((0, 0), 0.7, color="white")

            df = df.transform(lambda x: x / x.sum())
            df_mask = []
            for index, row in df.iterrows():
                df_mask.append(abs(row[0]))

            # Draw the pie chart
            wedges, texts, autotexts = ax.pie(
                df_mask,
                labels=df.index,
                autopct="%1.1f%%",
                pctdistance=0.85,
                wedgeprops={"linewidth": 3, "edgecolor": "white"},
                textprops={"family": "Simhei"},
            )

            for i, pie_wedge in enumerate(wedges):
                # 如果有指定颜色就颜色，否则按预设列表选取
                if pie_wedge.get_label() in COLOR_DICT.keys():
                    color = COLOR_DICT[pie_wedge.get_label()]
                else:
                    color = COLOR_LIST[i]

                pie_wedge.set_facecolor(color)

                if df.iloc[i, 0] < 0:
                    pie_wedge.set_facecolor("white")

            for k, autotext in enumerate(autotexts):
                autotext.set_color("white")
                autotext.set_fontsize(10)
                autotext.set_text(self.fmt[j].format(df.iloc[k, 0]))
                if df.iloc[k, 0] < 0:
                    autotext.set_color("r")

            if donut:
                ax.text(
                    0,
                    0,
                    donut_title,
                    horizontalalignment="center",
                    verticalalignment="center",
                    size=self.fontsize,
                    fontproperties=MYFONT,
                )
                ax.add_artist(my_circle)  # 用白色圆圈覆盖饼图，变成圈图

        return self.save()


# 继承基本类，Histgram分布图类
class PlotHist(GridFigure):
    def __init__(
        self,
        data,  # 原始数
        savepath: str = "./plots/",  # 保存位置
        width: int = 15,  # 宽
        height: int = 6,  # 高
        fontsize: int = 14,  # 字体大小
        title: str = None,  # 图表标题
        ytitle: str = None,  # y轴标题
        gs: GridSpec = None,  # GridSpec
        gs_title: list = None,  # 每个Grid的标题
        text_diff=None,  # 差异数据
        *args,
        **kwargs,
    ):
        super().__init__(
            data,
            savepath,
            width,
            height,
            fontsize,
            title,
            ytitle,
            gs,
            gs_title,
            *args,
            **kwargs,
        )
        self.text_diff = data_to_list(text_diff)
        if self.text_diff is not None:
            check_data_with_axes(self.text_diff, self.axes)

    def plot(
        self,
        bins: int = 100,
        tiles: int = 10,
        show_kde: bool = True,
        show_metrics: bool = True,
        show_tiles: bool = False,
        *args,
        **kwargs,
    ):
        for j, ax in enumerate(self.axes):
            df = self.data[j]
            df.plot(
                kind="hist",
                density=True,
                bins=bins,
                ax=ax,
                color="grey",
                legend=None,
                alpha=0.5,
            )
            if show_kde:
                ax_new = ax.twinx()
                df.plot(kind="kde", ax=ax_new, color="darkorange", legend=None)
                # ax_new.get_legend().remove()
                ax_new.set_yticks([])  # 删除y轴刻度
                ax_new.set_ylabel(None)

            if "xlim" in kwargs:
                ax.set_xlim(kwargs["xlim"][j][0], kwargs["xlim"][j][1])  # 设置x轴显示limit

            # ax.set_title(title)
            # ax.set_xlabel(xlabel)
            # ax.set_yticks([])  # 删除y轴刻度
            # ax.set_ylabel(ylabel)

            # 添加百分位信息
            if show_tiles:

                # 计算百分位数据
                percentiles = []
                for i in range(tiles):
                    percentiles.append(
                        [df.quantile((i) / tiles), "D" + str(i + 1)]
                    )  # 十分位Decile

                # 在hist图基础上绘制百分位
                for i, percentile in enumerate(percentiles):
                    ax.axvline(percentile[0], color="crimson", linestyle=":")  # 竖分隔线
                    ax.text(
                        percentile[0],
                        ax.get_ylim()[1] * 0.97,
                        int(percentile[0]),
                        ha="center",
                        color="crimson",
                        fontsize=self.fontsize,
                    )
                    if i < tiles - 1:
                        ax.text(
                            percentiles[i][0]
                            + (percentiles[i + 1][0] - percentiles[i][0]) / 2,
                            ax.get_ylim()[1],
                            percentile[1],
                            ha="center",
                        )
                    else:
                        ax.text(
                            percentiles[tiles - 1][0]
                            + (ax.get_xlim()[1] - percentiles[tiles - 1][0]) / 2,
                            ax.get_ylim()[1],
                            percentile[1],
                            ha="center",
                        )

            # 添加均值、中位数等信息
            if show_metrics:
                median = np.median(df.values)  # 计算中位数
                mean = np.mean(df.values)  # 计算平均数
                if self.text_diff is not None:
                    median_diff = self.text_diff[j]["中位数"]  # 计算对比中位数
                    mean_diff = self.text_diff[j]["平均数"]  # 计算对比平均数

                if median > mean:
                    yindex_median = 0.95
                    yindex_mean = 0.9
                    pos_median = "left"
                    pos_mean = "right"
                else:
                    yindex_mean = 0.95
                    yindex_median = 0.9
                    pos_median = "right"
                    pos_mean = "left"

                ax.axvline(median, color="crimson", linestyle=":")
                ax.text(
                    median,
                    ax.get_ylim()[1] * yindex_median,
                    "中位数：%s(%s)"
                    % ("{:.0f}".format(median), "{:+.0f}".format(median_diff)),
                    ha=pos_median,
                    color="crimson",
                    fontsize=self.fontsize,
                )

                ax.axvline(mean, color="purple", linestyle=":")
                ax.text(
                    mean,
                    ax.get_ylim()[1] * yindex_mean,
                    "平均数：%s(%s)" % ("{:.1f}".format(mean), "{:+.1f}".format(mean_diff)),
                    ha=pos_mean,
                    color="purple",
                    fontsize=self.fontsize,
                )

            # 去除ticks
            ax.get_yaxis().set_ticks([])

            # 轴标题
            # 命名xlabel
            if "xlabel" in kwargs:
                ax.set_xlabel(kwargs["xlabel"], fontsize=self.fontsize)
            ax.set_ylabel("频次", fontsize=self.fontsize)

        return self.save()


# 继承基本类，算珠图类
class PlotStripDot(GridFigure):
    def __init__(
        self,
        data,  # 原始数
        savepath: str = "./plots/",  # 保存位置
        width: int = 15,  # 宽
        height: int = 6,  # 高
        fontsize: int = 14,  # 字体大小
        gs: GridSpec = None,  # GridSpec
        fmt: list = [",.0f"],  # 每个grid的数字格式
        text_diff=None,  # 差异数据
        *args,
        **kwargs,
    ):
        super().__init__(
            data, savepath, width, height, fontsize, gs, fmt, *args, **kwargs,
        )
        self.text_diff = data_to_list(text_diff)
        if self.text_diff is not None:
            check_data_with_axes(self.text_diff, self.axes)

    def plot(self):
        fmt_diff = [fmt[:2] + "+" + fmt[2:] for fmt in self.fmt]

        for j, ax in enumerate(self.axes):
            df = self.data[j]
            index_range = range(1, len(df.index) + 1)
            ax.hlines(
                y=index_range,
                xmin=df.iloc[:, 0],
                xmax=df.iloc[:, 1],
                color="grey",
                alpha=0.3,
            )  # 连接线
            ax.scatter(
                df.iloc[:, 0],
                index_range,
                color="grey",
                alpha=0.3,
                label=df.columns[0],
            )  # 起始端点
            ax.scatter(
                df.iloc[:, 1],
                index_range,
                color=COLOR_LIST[j],
                alpha=0.4,
                label=df.columns[1],
            )  # 结束端点

            # 添加最新时点的数据标签
            text_gap = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 50
            for i in index_range:
                ax.text(
                    df.iloc[i - 1, 1] + text_gap,
                    i,
                    self.fmt[j].format(df.iloc[i - 1, 1]),
                    ha="left",
                    va="center",
                    color=COLOR_LIST[j],
                    fontsize=self.fontsize,
                    zorder=20,
                    **NUM_FONT,
                )
            # 添加间隔线
            list_range = list(index_range)
            list_range.append(max(list_range) + 1)
            ax.hlines(
                y=[i - 0.5 for i in list_range],
                xmin=ax.get_xlim()[0],
                xmax=ax.get_xlim()[1],
                color="grey",
                linestyle="--",
                linewidth=0.5,
                alpha=0.2,
            )
            ax.set_yticks(index_range)  # 添加y轴标签
            ax.set_yticklabels(df.index)
            ax.tick_params(
                axis="y", which="major", labelsize=self.fontsize
            )  # 调整y轴标签字体大小
            # if j != 0 and j != 2:  # 多图的情况，除第一张图以外删除y轴信息
            #     ax.get_yaxis().set_ticks([])

            if self.text_diff is not None:
                if self.text_diff[j] is not None and self.text_diff[j].empty is False:
                    for i in index_range:

                        idx = df.index[i - 1]
                        try:
                            v_diff = self.text_diff[j].loc[idx].values[0]
                        except:
                            v_diff = 0

                        # 正负色
                        if v_diff < 0:
                            fontcolor = "crimson"
                        else:
                            fontcolor = "black"

                        if v_diff > 0:
                            edgecolor_diff = "green"
                        elif v_diff < 0:
                            edgecolor_diff = "red"
                        else:
                            edgecolor_diff = "darkorange"

                        if v_diff != 0 and math.isnan(v_diff) is False:
                            t = ax.text(
                                ax.get_xlim()[1] * 1.1,
                                i,
                                fmt_diff[j].format(v_diff),
                                ha="center",
                                va="center",
                                color=fontcolor,
                                fontsize=self.fontsize,
                                zorder=20,
                                **NUM_FONT,
                            )
                            # t.set_bbox(
                            #     dict(
                            #         facecolor=edgecolor_diff,
                            #         alpha=0.25,
                            #         edgecolor=edgecolor_diff,
                            #         zorder=20,
                            #     )
                            # )

            ax.invert_yaxis()  # 翻转y轴，最上方显示排名靠前的序列

            # 图例
            ax.legend(
                # df.columns,
                loc="lower right",
                # ncol=4,
                # bbox_to_anchor=(0.5, -0.1),
                # labelspacing=1,
                # frameon=False,
                prop={"family": "SimHei", "size": self.fontsize},
            )

        return self.save()


# 继承基本类，网格热力图类
class PlotHeatGrid(GridFigure):
    def plot(self, cbar: bool = True, cmap: list = ["bwr"], fmt: list = [",.0f"]):
        check_data_with_axes(cmap, self.axes)
        check_data_with_axes(fmt, self.axes)

        for j, ax in enumerate(self.axes):
            df = self.data[j]
            sns.heatmap(
                df,
                ax=ax,
                annot=True,
                cbar=cbar,
                cmap=cmap[j],
                fmt=fmt[j],
                annot_kws={"fontsize": self.fontsize},
            )

            ax.set(ylabel=None)  # 去除y轴标题

        return self.save()


# 继承基本类，线形图
class PlotLine(GridFigure):
    def plot(
        self,
        show_legend: bool = True,
        show_labels: bool = True,
        label_hshift_index: float = 10,
    ):
        for j, ax in enumerate(self.axes):
            # 处理绘图数据
            df = self.data[j].transpose()

            for k, index in enumerate(df.index):
                # 如果有指定颜色就颜色，否则按预设列表选取
                if index in COLOR_DICT.keys():
                    color = COLOR_DICT[index]
                else:
                    color = COLOR_LIST[k]
                ax.plot(
                    df.columns,
                    df.loc[index, :],
                    color=color,
                    marker="o",
                    markerfacecolor="white",
                    label=index,
                )
                # Label Text
                if show_labels:
                    # # 计算标签靠上的间隔
                    # if "ylim" in self.style:
                    #     label_hshift = (
                    #         self.style["ylim"][j][1] - self.style["ylim"][j][0]
                    #     ) / label_hshift_index
                    # else:
                    #     label_hshift = (
                    #         ax.get_ylim()[1] - ax.get_ylim()[0]
                    #     ) / label_hshift_index
                    # print(label_hshift)

                    for i, col in enumerate(df):
                        v = df.loc[index, col]
                        ax.text(
                            col,
                            v,
                            self.fmt[j].format(v),
                            color="white",
                            va="center",
                            ha="center",
                            fontsize=self.fontsize,
                            bbox=dict(facecolor=color, edgecolor=None, alpha=0.7),
                            **NUM_FONT,
                        )
            # 图例
            if show_legend:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))  # 和下放调用.values()/.keys()配合去除重复的图例
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc="center left",
                    ncol=1,
                    bbox_to_anchor=(1, 0.5),
                    labelspacing=1,
                    frameon=False,
                    prop={"family": "SimHei", "size": self.fontsize},
                )

        return self.save()


# 继承基本类，水平柱状图类
class PlotHorizontalBar(GridFigure):
    def __init__(
        self,
        data,  # 原始数
        savepath: str = "./plots/",  # 保存位置
        width: int = 15,  # 宽
        height: int = 6,  # 高
        fontsize: int = 14,  # 字体大小
        gs: GridSpec = None,  # GridSpec
        fmt: list = [",.0f"],  # 每个grid的数字格式
        text_diff=None,  # 差异数据
        *args,
        **kwargs,
    ):
        super().__init__(
            data, savepath, width, height, fontsize, gs, fmt, *args, **kwargs,
        )
        self.text_diff = data_to_list(text_diff)
        if self.text_diff is not None:
            check_data_with_axes(self.text_diff, self.axes)

    def plot(
        self,
        show_label: bool = True,
        show_legend: bool = True,
        threshold: float = 0.02,
        text_diff_pos_index: float = 1.1,
        text_diff_threshold: float = 0.02,
    ):
        fmt_diff = [fmt[:2] + "+" + fmt[2:] for fmt in self.fmt]

        for j, ax in enumerate(self.axes):
            df = self.data[j]
            # index_range = range(1, len(df.index) + 1)

            for k, index in enumerate(df.index):
                left_pos = 0
                left_neg = 0
                max_v = df.values.max()
                min_v = df.values.min()
                for i, col in enumerate(df):
                    v = df.loc[index, col]
                    if v >= 0:
                        left = left_pos
                    else:
                        left = left_neg
                    # 如果有指定颜色就颜色，否则按预设列表选取
                    if col in COLOR_DICT.keys():
                        color = COLOR_DICT[col]
                    elif index in COLOR_DICT.keys():
                        color = COLOR_DICT[index]
                    else:
                        color = COLOR_LIST[i]

                    bar = ax.barh(
                        index, v, color=color, left=left, height=0.5, label=col,
                    )

                    if show_label is True:
                        if df.shape[1] == 1:  # 只有一列数的情况（非堆叠）
                            # 根据数据判断标签是否需要微调
                            if 0 <= v < max_v * 0.2:
                                pos_x = v * 1.1
                                ha = "left"
                                fontcolor = color
                            elif min_v * 0.2 < v < 0:
                                pos_x = v * 0.9
                                ha = "right"
                                fontcolor = color
                            else:
                                pos_x = v / 2
                                ha = "center"
                                fontcolor = "white"

                        else:  # 堆叠的情况
                            pos_x = left + v / 2
                            ha = "center"
                            fontcolor = "white"

                        if abs(v) >= threshold:
                            ax.text(
                                pos_x,
                                index,
                                self.fmt[j].format(v),
                                color=fontcolor,
                                va="center",
                                ha=ha,
                                fontsize=self.fontsize,
                                **NUM_FONT,
                            )
                    if v >= 0:
                        left_pos += v
                    else:
                        left_neg += v

                    patches = ax.patches
                    for rect in patches:
                        height = rect.get_height()
                        # 负数则添加纹理
                        if height < 0:
                            rect.set_hatch("//")

                # 添加间隔线
                ax.axhline(k + 0.5, color="grey", linestyle="--", linewidth=0.5)
            ax.axhline(-0.5, color="grey", linestyle="--", linewidth=0.5)  # 额外多添加一条，上下

            ax.axvline(0, color=color, linewidth=0.5)  # 0值处增加竖线

            if self.text_diff is not None:
                if self.text_diff[j] is not None and self.text_diff[j].empty is False:
                    for i, index in enumerate(df.index):
                        try:
                            v_diff = self.text_diff[j].loc[index].values[0]
                        except:
                            v_diff = 0

                        # 正负色
                        if v_diff < 0:
                            fontcolor = "crimson"
                        else:
                            fontcolor = "darkgreen"

                        if v_diff > 0:
                            edgecolor_diff = "green"
                        elif v_diff < 0:
                            edgecolor_diff = "red"
                        else:
                            edgecolor_diff = "darkorange"

                        if v_diff != 0 and math.isnan(v_diff) is False:
                            if abs(v_diff) >= text_diff_threshold:
                                t = ax.text(
                                    ax.get_xlim()[1] * text_diff_pos_index,
                                    index,
                                    fmt_diff[j].format(v_diff),
                                    ha="center",
                                    va="center",
                                    color=fontcolor,
                                    fontsize=self.fontsize * 0.8,
                                    zorder=20,
                                    **NUM_FONT,
                                )
                                t.set_bbox(
                                    dict(
                                        facecolor=edgecolor_diff,
                                        alpha=0.25,
                                        edgecolor=edgecolor_diff,
                                        zorder=20,
                                    )
                                )
            # y轴倒序
            ax.invert_yaxis()

            # 去除上下多余空白
            ax.margins(y=0)

            # # 添加平均值竖线
            # if vline_value is not None:
            #     ax.axvline(
            #         vline_value[i],
            #         color=COLOR_LIST[i],
            #         linestyle="--",
            #         linewidth=1,
            #         zorder=1,
            #     )
            #     ax.text(
            #         vline_value[i],
            #         ax.get_ylim()[1] + 0.1,
            #         "平均:" + formats[i].format(vline_value[i]),
            #         va="top",
            #         color=COLOR_LIST[i],
            #         fontsize=fontsize,
            #     )

            #     # 添加绝对值数字标签
            #     ax.text(
            #         pos_x,
            #         j,
            #         formats[i].format(v),
            #         ha=ha,
            #         va="center",
            #         color=fontcolor,
            #         fontsize=fontsize,
            #     )
            #     # 添加和pre差值数字标签
            #     if df_pre is not None and df_pre.empty is False:
            #         idx = df_bar.index[j]
            #         v_diff = df_diff.loc[idx]

            #         # 正负色
            #         if v_diff > 0:
            #             edgecolor_diff = "green"
            #         elif v_diff < 0:
            #             edgecolor_diff = "red"
            #         else:
            #             edgecolor_diff = "darkorange"

            #         # # 下列代码将所有各列的格式都在正值前添加+号
            #         # str_list = list(formats[i])
            #         # str_list.insert(2, "+")
            #         # format_diff = "".join(str_list)
            #         # print(format_diff)

            #         if v_diff != 0 and math.isnan(v_diff) is False:
            #             t = ax.text(
            #                 ax.get_xlim()[1] * 1.1,
            #                 j,
            #                 formats_diff[i].format(v_diff),
            #                 ha="center",
            #                 va="center",
            #                 color="black",
            #                 fontsize=fontsize * 0.7,
            #                 zorder=20,
            #                 **NUM_FONT
            #             )
            #             t.set_bbox(
            #                 dict(
            #                     facecolor=edgecolor_diff,
            #                     alpha=0.25,
            #                     edgecolor=edgecolor_diff,
            #                     zorder=20,
            #                 )
            #             )
            #
        return self.save()


# 继承基本类，堆积柱状对比图类
class PlotStackedBar(GridFigure):
    def __init__(
        self,
        data,  # 原始数
        savepath: str = "./plots/",  # 保存位置
        width: int = 15,  # 宽
        height: int = 6,  # 高
        fontsize: int = 14,  # 字体大小
        gs: GridSpec = None,  # GridSpec
        fmt: list = [",.0f"],  # 每个grid的数字格式
        style: dict = {"title": "test"},  # 风格字典
        data_line=None,  # 折线图数据
        fmt_line=None,  # 折线图格式
        *args,
        **kwargs,
    ):
        super().__init__(
            data, savepath, width, height, fontsize, gs, fmt, style, *args, **kwargs,
        )
        self.data_line = data_to_list(data_line)
        if fmt_line is not None:
            self.fmt_line = ["{:%s}" % f for f in fmt_line]
        else:
            self.fmt_line = None

        if self.data_line is not None:
            check_data_with_axes(self.data_line, self.axes)
        if self.fmt_line is not None:
            check_data_with_axes(self.fmt_line, self.axes)

    def plot(
        self,
        stacked: bool = True,
        show_label: bool = True,
        show_total_label: bool = False,
        add_gr_text: bool = False,
        threshold: float = 0.02,
        show_legend: bool = True,
        *args,
        **kwargs,
    ):
        for j, ax in enumerate(self.axes):
            # 处理绘图数据
            df = self.data[j]
            df_gr = self.data[j].pct_change(axis=1)
            if self.data_line is not None:
                df_line = self.data_line[j]

            # 绝对值bar图和增长率标注
            for k, index in enumerate(df.index):
                bottom_pos = 0
                bottom_neg = 0
                bottom_gr = 0
                bbox_props = None
                max_v = df.values.max()
                min_v = df.values.min()

                for i, col in enumerate(df):
                    v = df.loc[index, col]

                    if stacked:
                        if v >= 0:
                            bottom = bottom_pos
                        else:
                            bottom = bottom_neg
                    else:
                        bottom = 0

                    # 如果有指定颜色就颜色，否则按预设列表选取
                    if stacked:
                        if col in COLOR_DICT.keys():
                            color = COLOR_DICT[col]
                        elif index in COLOR_DICT.keys():
                            color = COLOR_DICT[index]
                        else:
                            color = COLOR_LIST[i]
                    else:
                        color = COLOR_LIST[i]

                    # bar宽度
                    bar_width = 0
                    if stacked:
                        if isinstance(
                            df.index, pd.DatetimeIndex
                        ):  # 如果x轴是日期，宽度是以“天”为单位的
                            bar_width = 20
                        else:
                            bar_width = 0.5
                    else:
                        bar_width = 0.8 / df.shape[1]

                    # bar x轴位置
                    if stacked:
                        pos_x = k
                    else:
                        pos_x = k + bar_width * i

                    # 绘制bar图
                    bar = ax.bar(
                        pos_x,
                        v,
                        width=bar_width,
                        color=color,
                        bottom=bottom,
                        label=col,
                    )
                    if show_label is True:
                        if stacked is False or df.shape[1] == 1:  # 非堆叠图或只有一列数的情况（非堆叠）
                            # 根据数据判断标签是否需要微调
                            if 0 <= v < max_v * 0.05:
                                pos_y = v * 1.1
                                va = "bottom"
                                fontcolor = color
                            elif min_v * 0.05 < v < 0:
                                pos_y = v * 0.9
                                va = "top"
                                fontcolor = color
                            else:
                                pos_y = v / 2
                                va = "center"
                                fontcolor = "white"

                        else:  # 堆叠的情况
                            pos_y = bottom + v / 2
                            va = "center"
                            fontcolor = "white"

                        if abs(v) >= threshold:
                            ax.text(
                                pos_x,
                                pos_y,
                                self.fmt[j].format(v),
                                color=fontcolor,
                                va=va,
                                ha="center",
                                fontsize=self.fontsize,
                                **NUM_FONT,
                            )
                    if v >= 0:
                        bottom_pos += v
                    else:
                        bottom_neg += v

                    patches = ax.patches
                    for rect in patches:
                        height = rect.get_height()
                        # 负数则添加纹理
                        if height < 0:
                            rect.set_hatch("//")

                    if add_gr_text:
                        if k > 0:
                            # 各系列增长率标注
                            ax.annotate(
                                "{:+.1%}".format(df_gr.iloc[k, i]),
                                xy=(
                                    0.5,
                                    (
                                        bottom_gr
                                        + df.iloc[k - 1, i] / 2
                                        + df.iloc[k, i] / 2
                                    )
                                    / 2,
                                ),
                                ha="center",
                                va="center",
                                color=color,
                                fontsize=self.fontsize,
                                bbox=bbox_props,
                            )
                            bottom_gr += df.iloc[k - 1, i] + df.iloc[k, i]

                    # 在柱状图顶端添加total值
                    if show_total_label:
                        total = df.sum(axis=1)
                        for p, v in enumerate(total.values):
                            ax.text(
                                x=pos_x,
                                y=v,
                                s=self.fmt[j].format(float(v)),
                                fontsize=self.fontsize,
                                ha="center",
                                va="bottom",
                            )

                    # box = ax.get_position()
                    # ax.set_position([box.x0, box.y0, box.width, box.height * 1.1])

            # 如果是非堆叠图要手动指定x轴ticks
            if stacked is False:
                plt.xticks(np.arange(df.shape[0]) + bar_width / df.shape[1], df.index)
            else:
                plt.xticks(np.arange(df.shape[0]), df.index)

            # y轴标签格式
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: self.fmt[j].format(y))
            )

            ax.axhline(0, color="black", linewidth=0.5)  # y轴为0的横线

            if self.data_line is not None:

                # 增加次坐标轴
                ax2 = ax.twinx()

                if isinstance(df_line, pd.DataFrame):
                    label = df_line.columns[0]
                else:
                    label = df_line.name

                color_line = "darkorange"
                line = ax2.plot(
                    df_line.index,
                    df_line.values,
                    label=label,
                    color=color_line,
                    linewidth=1,
                    linestyle="dashed",
                    marker="o",
                    markersize=3,
                    markerfacecolor="white",
                )
                if "y2lim" in kwargs:
                    ax2.set_ylim(kwargs["y2lim"][0], kwargs["y2lim"][1])

                for i in range(len(df_line)):
                    if float(df_line.values[i]) <= ax2.get_ylim()[1]:
                        t = ax2.text(
                            x=df_line.index[i],
                            y=df_line.values[i],
                            s=self.fmt_line[j].format(float(df_line.values[i])),
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            color="white",
                        )
                        t.set_bbox(
                            dict(facecolor=color_line, alpha=0.7, edgecolor=color_line)
                        )

                # 次坐标轴标签格式
                ax2.yaxis.set_major_formatter(
                    FuncFormatter(lambda y, _: self.fmt_line[j].format(y))
                )

            # 图例
            if show_legend:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                handles, labels = ax.get_legend_handles_labels()
                if self.data_line is not None:
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    by_label = dict(
                        zip(
                            labels[::-1] + labels2[::-1],
                            handles[::-1] + handles2[::-1],
                        )
                    )  # 和下放调用.values()/.keys()配合去除重复的图例，顺便倒序让图例与图表保持一致
                else:
                    by_label = dict(
                        zip(labels[::-1], handles[::-1],)
                    )  # 和下放调用.values()/.keys()配合去除重复的图例，顺便倒序让图例与图表保持一致
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc="center left",
                    ncol=1,
                    bbox_to_anchor=(1, 0.5),
                    labelspacing=1,
                    frameon=False,
                    prop={"family": "SimHei", "size": self.fontsize},
                )

        return self.save()


# 继承基本类，堆积柱状对比图（增强型）类
class PlotStackedBarPlus(GridFigure):
    def plot(self, *args, **kwargs):
        H_INDEX = 1.03  # 外框对比bar的高度系数
        for j, ax in enumerate(self.axes):
            # 处理绘图数据
            df = self.data[j].transpose()
            df_share = self.data[j].apply(lambda x: x / x.sum()).transpose()
            df_gr = self.data[j].pct_change(axis=1).transpose()

            # 绝对值bar图和增长率标注
            for k, index in enumerate(df.index):
                bottom = 0
                bottom_gr = 0
                bbox_props = None
                for i, col in enumerate(df):
                    v = df.loc[index, col]

                    # 如果有指定颜色就颜色，否则按预设列表选取
                    if col in COLOR_DICT.keys():
                        color = COLOR_DICT[col]
                    else:
                        color = COLOR_LIST[i]

                    # 绝对值bar图
                    ax.bar(
                        index, v, width=0.5, color=color, bottom=bottom, label=col,
                    )
                    ax.text(
                        index,
                        bottom + v / 2,
                        "{:,.0f}".format(v)
                        + "("
                        + "{:.1%}".format(df_share.loc[index, col])
                        + ")",
                        color="white",
                        va="center",
                        ha="center",
                        fontsize=self.fontsize,
                    )

                    if math.isnan(v) is False:
                        bottom += v

                    if k > 0:
                        # 各系列增长率标注
                        ax.annotate(
                            "{:+.1%}".format(df_gr.iloc[k, i]),
                            xy=(
                                0.5,
                                (bottom_gr + df.iloc[k - 1, i] / 2 + df.iloc[k, i] / 2)
                                / 2,
                            ),
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=self.fontsize,
                            bbox=bbox_props,
                        )
                        if math.isnan(df.iloc[k - 1, i]) is False:
                            bottom_gr += df.iloc[k - 1, i]
                        if math.isnan(df.iloc[k, i]) is False:
                            bottom_gr += df.iloc[k, i]

                # 绘制总体增长率
                if k > 0:
                    gr = df.iloc[k, :].sum() / df.iloc[k - 1, :].sum() - 1

                    ax.annotate(
                        "{:+.1%}".format(gr),
                        xy=(
                            0.5,
                            (df.iloc[k, :].sum() + df.iloc[k - 1, :].sum())
                            / 2
                            * (H_INDEX + 0.02),
                        ),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=self.fontsize,
                        bbox=bbox_props,
                    )
            # 绘制总体表现外框
            ax.bar(
                df.index,
                df.sum(axis=1) * H_INDEX,
                width=0.6,
                linewidth=1,
                linestyle="--",
                facecolor=(1, 0, 0, 0.0),
                edgecolor=(0, 0, 0, 1),
            )
            for index in df.index:
                ax.text(
                    index,
                    df.loc[index, :].sum() * (H_INDEX + 0.02),
                    "{:,.0f}".format(df.loc[index, :].sum()),
                    ha="center",
                    fontsize=self.fontsize,
                )

            # 因为有总体数量标签，增加一些图表高度
            box = ax.get_position()
            ax.set_position(
                [box.x0, box.y0 - box.height * 0.1, box.width, box.height * 1.1]
            )

            # 图例
            if len(self.axes) > 1:  # 平行多图时的情况
                ax.legend(
                    df.columns,
                    loc="upper center",
                    ncol=4,
                    bbox_to_anchor=(0.5, -0.1),
                    labelspacing=1,
                    frameon=False,
                    prop={"family": "SimHei", "size": self.fontsize},
                )
            else:  # 单图时的情况
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(
                    zip(labels[::-1], handles[::-1],)
                )  # 和下放调用.values()/.keys()配合去除重复的图例，顺便倒序让图例与图表保持一致
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc="center left",
                    ncol=1,
                    bbox_to_anchor=(1, 0.5),
                    labelspacing=1,
                    frameon=False,
                    prop={"family": "SimHei", "size": self.fontsize},
                )
            # 去除ticks
            # ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        return self.save()


if __name__ == "__main__":
    x = np.linspace(-3, 3, 201)
    y = np.tanh(x) + 0.1 * np.cos(5 * x)

    gs = GridSpec(1, 2)
    # gs = None
    f = plt.figure(
        FigureClass=PlotHeatGrid,
        data=[1, 2],
        title="draft",
        gs=gs,
        savepath="./Reporting/plots/",
    )
    f.plot()

# for i in range(df.shape[1]):
#     ax = plt.subplot(gs[i])
#     df_bar = df.iloc[:, i]
#     if df_pre is not None and df_pre.empty is False:
#         df_pre = df_pre.reindex(df_bar.index)
#         df_diff = df_bar - df_pre.iloc[:, i].fillna(0)

#     ax = df_bar.plot(
#         kind="barh", alpha=0.8, color=COLOR_LIST[i], edgecolor="black", zorder=3
#     )
