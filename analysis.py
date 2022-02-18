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
import jieba
import re
from collections import Counter
from wordcloud import WordCloud

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

D_MAP_GATEGORY = {
    "缬沙坦": "ARB",
    "厄贝沙坦": "ARB",
    "氯沙坦": "ARB",
    "替米沙坦": "ARB",
    "坎地沙坦": "ARB",
    "奥美沙坦酯": "ARB",
    "阿利沙坦酯": "ARB",
    "培哚普利": "ACEI",
    "贝那普利": "ACEI",
    "沙库巴曲缬沙坦钠": "ARNI",
}


def daily_dose(strength: str, daily: str) -> str:
    strength_map = {
        "50 MG": 50,
        "100 MG": 100,
        "200 MG": 200,
    }
    daily_map = {"一天一次": 1, "一天两次": 2, "一天三次": 3, "其他": 4, "立即": 1, "无": 1}
    dose = strength_map[strength] * daily_map[daily]
    if dose > 400:
        dose = 400
    return f"{dose}MG"


if __name__ == "__main__":
    df = pd.read_excel("./普瑞快思-2021Q3_ARNI.xlsx", engine="openpyxl", sheet_name="标准片数")
    df["MAT"] = df["日期"].map(D_MAP_MAT)
    df["品类"] = df["通用名"].map(D_MAP_GATEGORY)
    mask = (df["原始诊断"] != "无诊断") & (df["统计项"] == "标准片数") & (df["来源"] == "门诊")
    mask = mask & (df["通用名"] == "沙库巴曲缬沙坦钠")
    # mask = mask & (df["关注科室"].isin(["肾内科"]))
    df2 = df.loc[mask, :]
    df2["日治疗剂量"] = df2.apply(
        lambda x: daily_dose(strength=x["规格"], daily=x["用法"]), axis=1
    )
    print(df2["日治疗剂量"])
    r = Rx(df2, name="ARNI - 门诊 - 标准片数")
    r_pre = Rx(df2[df2["MAT"] == "MAT20Q3"], name="ARNI - 门诊 - 标准片数 - 20Q3")
    r_post = Rx(df2[df2["MAT"] == "MAT21Q3"], name="ARNI - 门诊 - 标准片数 - 21Q3")

    r.plot_add(
        [
            "高血压 == 1 and 冠心病 == 0 and 心力衰竭 == 0",
            "高血压 == 0 and 冠心病 == 1 and 心力衰竭 == 0",
            "高血压 == 0 and 冠心病 == 0 and 心力衰竭 == 1",
        ]
    )
    # df3 = df2.query("高血压 == 1 and 冠心病 == 0 and 心力衰竭 == 0")
    # rx_numbers = "{:,.0f}".format(df3.shape[0])
    # print(rx_numbers)
    # df3 = df3.groupby("日治疗剂量").agg({"统计值": sum,})
    # plot_data = df3.copy()
    # plot_data.loc["其他"] = plot_data.reindex(
    #     index=["50MG", "150MG", "300MG", "400MG"]
    # ).sum()
    # plot_data = plot_data.reindex(index=["100MG", "200MG", "其他"])

    # df3 = df3.reset_index()
    # print(df3)
    # df3["日治疗剂量"] = df3["日治疗剂量"].str[:-2].astype(int)
    # add = (df3["日治疗剂量"] * df3["统计值"]).sum() / df3["统计值"].sum()
    # add = "{:,.0f}".format(add)
    # print(add)
    # title = "诺欣妥平均日治疗剂量"
    # donut_title = f"ADD: {add}MG\n\n处方张数: {rx_numbers}"
    # fmt = [",.0%"]
    # f = plt.figure(
    #     FigureClass=PlotPie,
    #     width=5,
    #     height=5,
    #     gs=None,
    #     fmt=fmt,
    #     data=plot_data,
    #     fontsize=16,
    #     style={"title": title,},
    # )
    # f.plot(donut_title=donut_title)

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

    # df3 = r_post.get_intersect("品类")
    # df3.drop([""], axis=0, inplace=True)

    # plot_data = []
    # gs_title = []
    # for col in df3:
    #     plot_data.append(df3.loc[:, col].sort_values(ascending=False).head(20))
    #     gs_title.append(col)

    # fmt = [".1%"] * df3.shape[1]
    # title = f"{r_post.name} - 具体合并症组合贡献占比 - Top20"
    # gs = GridSpec(1, 3, wspace=1)

    # f = plt.figure(
    #     FigureClass=PlotStripDot,
    #     width=15,
    #     height=6,
    #     fmt=fmt,
    #     gs=gs,
    #     data=plot_data,
    #     fontsize=10,
    #     style={"title": title, "gs_title": gs_title, "remove_xticks": True},
    # )
    # f.plot()

    # r.plot_group_barh("品类", same_xlim=True)
