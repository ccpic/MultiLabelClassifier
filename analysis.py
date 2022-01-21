from itertools import groupby
import pandas as pd

import numpy as np
from data_class import Rx
from chart_class import (
    PlotStackedBar,
    PlotLine,
    PlotHorizontalBar,
    PlotPie,
    PlotStripDot,
    COLOR_LIST,
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

D_MAP_MAT = {
    "19Q4": "MAT20Q3",
    "20Q1": "MAT20Q3",
    "20Q2": "MAT20Q3",
    "20Q3": "MAT20Q3",
    "20Q4": "MAT21Q3",
    "21Q1": "MAT21Q3",
    "21Q2": "MAT21Q3",
    "21Q3": "MAT21Q3",
}


# def cal_ylim(df:pd.DataFrame, scale_to:float=0.1):
#     v_max = df.max()
#     v_min = df.min()

if __name__ == "__main__":
    df = pd.read_excel("./普瑞快思-2021Q3_test3.xlsx", engine="openpyxl", sheet_name="标准片数")
    df["MAT"] = df["日期"].map(D_MAP_MAT)
    mask = (df["原始诊断"] != "无诊断") & (df["统计项"] == "标准片数") & (df["来源"] == "门诊")
    mask = mask & (df["通用名"] != "沙库巴曲缬沙坦钠")
    # & (df["关注科室"].isin(["心内科"]))
    df2 = df.loc[mask, :]
    print(df2.columns, df2["MAT"].unique())
    r = Rx(df2, name="RAAS平片处方 - 门诊 - 标准片数")
    r_pre = Rx(df2[df2["MAT"] == "MAT20Q3"], name="RAAS平片处方 - 门诊 - 标准片数 - 20Q3")
    r_post = Rx(df2[df2["MAT"] == "MAT21Q3"], name="RAAS平片处方 - 门诊 - 标准片数 - 21Q3")

    # df3 = r.get_union(groupby="日期", len_set=1, sort_values=False).transpose()
    # df3.drop("", axis=1, inplace=True)
    # plot_data = []
    # gs_title = []
    # for col in df3.columns:
    #     gs_title.append(col)
    #     plot_data.append(df3.loc[:, col].to_frame())
    # print(plot_data)
    # gs = GridSpec(2, 4, wspace=0.1, hspace=0.5)
    # fmt = [",.0%"] * 8
    # title = "RAAS平片相关适应症处方贡献占比 - 季度趋势"
    # f = plt.figure(
    #     FigureClass=PlotLine,
    #     width=17,
    #     height=5,
    #     gs=gs,
    #     fmt=fmt,
    #     data=plot_data,
    #     fontsize=9,
    #     style={
    #         "title": title,
    #         "gs_title": gs_title,
    #         "xlabel_rotation": 90,
    #         "last_xticks_only": True,
    #         "remove_yticks": True,
    #         "ylim": (
    #             (0.9, 1),
    #             (0.4, 0.5),
    #             (0.1, 0.2),
    #             (0.45, 0.55),
    #             (0.03, 0.13),
    #             (0, 0.1),
    #             (0, 0.1),
    #             (0, 0.1),
    #         ),
    #     },
    # )

    # f.plot(show_legend=False)

    # r.plot_total_bar(groupby="MAT")
    # r.plot_group_barh(groupby="通用名", diffby="MAT")
    # r.plot_group_barh(groupby="关注科室", diffby="MAT")
    # r.plot_intersect(groupby="MAT")

    #     text_diff.append(
    #         df_post[col]
    #         .subtract(df_pre[col])
    #         .to_frame()
    #         .reindex_like(df_post[col].to_frame())
    #     )
    #     gs_title.append(col)


