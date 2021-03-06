import pandas as pd
import numpy as np
from itertools import chain, combinations
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib_venn import venn3, venn3_circles, venn2, venn2_circles
import matplotlib.font_manager as fm
from chart_func import plot_grid_barh
from prince import CA


# 设置matplotlib正常显示中文和负号
mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
mpl.rcParams["axes.unicode_minus"] = False  # 正常显示负号

D_SORTER = {
    "关注科室": ["心内科", "老干科", "普内科", "肾内科", "神内科", "内分泌科", "其他",],
    "通用名": ["缬沙坦", "厄贝沙坦", "氯沙坦", "替米沙坦", "坎地沙坦", "奥美沙坦酯", "阿利沙坦酯", "培哚普利", "贝那普利",],
    "商品名": ["代文", "安博维", "科素亚", "美卡素", "必洛斯", "傲坦", "信立坦", "雅施达", "洛汀新",],
    "来源": ["门诊", "病房",],
}


def all_subsets(ss):  # 获取当前tuple包含元素所有可能的combinations
    return list(chain(*map(lambda x: combinations(ss, x), range(1, len(ss) + 1))))


def refine(df, len_set=None, labels_in=None):
    df.dropna(axis=1, how="all", inplace=True)  # 删除所有行都为nan的列

    if len_set is not None:  # 返回指定标签数量的结果
        df = df[df.index.map(lambda x: x.count("+")) == len_set - 1]

    if labels_in is not None:  # 返回含有指定标签的结果
        df = df[df.index.map(lambda x: "高血压" in x) == True]

    if "占比" in df.columns:  # 排序
        df.sort_values(by="占比", ascending=False, inplace=True)
    else:
        df.sort_values(df.columns[0], ascending=False, inplace=True)

    df.fillna(0, inplace=True)  # 剩余的nan更新为0

    return df


def get_intersect(df, labels):
    total_number = df["统计值"].sum()
    df_intersect = df.loc[:, labels].apply(
        lambda s: [s.name if v == 1 else np.nan for v in s]
    )
    df_intersect = df_intersect.T.apply(lambda x: "+".join(x.dropna().tolist()))
    df_intersect = pd.concat([df_intersect, df["统计值"]], axis=1)
    df_intersect.columns = ["高血压合并症", "统计值"]
    df_intersect = df_intersect.groupby(["高血压合并症"]).sum()
    df_intersect["占比"] = df_intersect["统计值"] / total_number

    return df_intersect


def get_union(df_intersect):
    df_intersect.reset_index(inplace=True)
    df_intersect["高血压合并症"] = df_intersect["高血压合并症"].apply(
        lambda x: all_subsets(tuple(x.split("+")))
    )

    df_union = df_intersect.explode("高血压合并症").groupby("高血压合并症").sum()
    df_union.index = df_union.index.map(
        lambda x: "+".join(x) if type(x) is tuple else x
    )

    return df_union


class Rx(pd.DataFrame):
    @property
    def _constructor(self):
        return Rx._internal_constructor(self.__class__)

    class _internal_constructor(object):
        def __init__(self, cls):
            self.cls = cls

        def __call__(self, *args, **kwargs):
            kwargs["name"] = None
            return self.cls(*args, **kwargs)

        def _from_axes(self, *args, **kwargs):
            return self.cls._from_axes(*args, **kwargs)

    def __init__(
        self,
        data,
        name,
        savepath="./plots/",
        labels=["高血压", "冠心病", "糖尿病", "血脂异常", "慢性肾病", "卒中", "高尿酸", "心力衰竭",],
        index=None,
        columns=None,
        dtype=None,
        copy=True,
    ):
        super(Rx, self).__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )
        self.name = name
        self.savepath = savepath
        self.labels = labels

    def get_intersect(
        self, groupby=None, groupby_metric="占比", len_set=None, labels_in=None
    ):  # 获得指定条件的所有标签组合（互不相交，加和=100%）
        if groupby is None:
            df_intersect = get_intersect(self, self.labels)
            df_intersect = refine(
                df=df_intersect, len_set=len_set, labels_in=labels_in,
            )

            return df_intersect
        else:
            cols = self[groupby].unique()

            for i, col in enumerate(cols):
                df_intersect = get_intersect(self[self[groupby] == col], self.labels)[
                    groupby_metric
                ]
                if i == 0:
                    df_intersect_groupby = df_intersect
                else:
                    df_intersect_groupby = pd.concat(
                        [df_intersect_groupby, df_intersect], axis=1
                    )
            df_intersect_groupby.columns = cols

            if groupby in D_SORTER:
                try:
                    df_intersect_groupby = df_intersect_groupby.reindex(
                        columns=D_SORTER[groupby]
                    )  # 对于部分变量有固定列排序
                except KeyError:
                    pass

            df_intersect_groupby = refine(
                df=df_intersect_groupby, len_set=len_set, labels_in=labels_in
            )

            return df_intersect_groupby

    def get_union(
        self, groupby=None, groupby_metric="占比", len_set=None, labels_in=None
    ):  # 获得指定条件的所有标签组合的并集
        if groupby is None:
            df_intersect = self.get_intersect()
            df_union = get_union(df_intersect)

            df_union = refine(df=df_union, len_set=len_set, labels_in=labels_in,)

            return df_union
        else:
            cols = self[groupby].unique()

            for i, col in enumerate(cols):
                df_union = get_union(
                    get_intersect(self[self[groupby] == col], self.labels)
                )[groupby_metric]
                if i == 0:
                    df_union_groupby = df_union
                else:
                    df_union_groupby = pd.concat([df_union_groupby, df_union], axis=1)
            df_union_groupby.columns = cols

            if groupby in D_SORTER:
                try:
                    df_union_groupby = df_union_groupby.reindex(
                        columns=D_SORTER[groupby]
                    )  # 对于部分变量有固定列排序
                except KeyError:
                    pass

            df_union_groupby = refine(
                df=df_union_groupby, len_set=len_set, labels_in=labels_in
            )

            return df_union_groupby

    def get_como_len(self, groupby=None):
        if groupby is None:
            df_intersect = self.get_intersect()
            df_intersect.reset_index(inplace=True)
            df_intersect["高血压合并症"] = df_intersect["高血压合并症"].apply(
                lambda x: len(tuple(x.split("+")))
            )

            df_como_len = df_intersect.groupby(["高血压合并症"]).sum()

            return df_como_len
        else:
            cols = self[groupby].unique()

            for i, col in enumerate(cols):
                df_intersect = get_intersect(self[self[groupby] == col], self.labels)[
                    "占比"
                ].to_frame()
                df_intersect.reset_index(inplace=True)
                df_intersect["高血压合并症"] = df_intersect["高血压合并症"].apply(
                    lambda x: len(tuple(x.split("+")))
                )
                df_como_len = df_intersect.groupby(["高血压合并症"]).sum()
                if i == 0:
                    df_como_len_groupby = df_como_len
                else:
                    df_como_len_groupby = pd.concat(
                        [df_como_len_groupby, df_como_len], axis=1
                    )
            df_como_len_groupby.columns = cols

            if groupby in D_SORTER:
                try:
                    df_como_len_groupby = df_como_len_groupby.reindex(
                        D_SORTER[groupby]
                    )  # 对于部分变量有固定列排序
                except KeyError:
                    pass

            return df_como_len_groupby

    def get_ss_venn(self, sets):
        all_ss = list(map(lambda x: "+".join(x), all_subsets(sets)))
        ss = self.get_union().loc[all_ss, "占比"]

        if len(all_ss) == 7:
            ss111 = ss[6]
            ss011 = ss[5] - ss111
            ss101 = ss[4] - ss111
            ss110 = ss[3] - ss111
            ss001 = ss[2] - (ss[4] + ss[5] - ss111)
            ss010 = ss[1] - (ss[3] + ss[5] - ss111)
            ss100 = ss[0] - (ss[3] + ss[4] - ss111)
            ss_venn = (ss100, ss010, ss110, ss001, ss101, ss011, ss111)
            return ss_venn, [ss[0], ss[1], ss[2]]
        elif len(all_ss) == 3:
            ss11 = ss[2]
            ss01 = ss[1] - ss11
            ss10 = ss[0] - ss11
            ss_venn = (ss10, ss01, ss11)
            return ss_venn, [ss[0], ss[1]]
        else:
            return None

    def plot_venn(self, sets):
        COLOR_DICT = {
            "高血压": "darkblue",
            "冠心病": "deepskyblue",
            "糖尿病": "pink",
            "血脂异常": "crimson",
            "卒中": "darkgreen",
            "慢性肾病": "darkorange",
            "高尿酸": "purple",
            "心力衰竭": "saddlebrown",
        }

        fig = plt.figure(figsize=(5, 5))
        ax = fig.subplots(1)

        set_labels = sets
        subsets, share = self.get_ss_venn(set_labels)
        print(subsets, share)
        set_colors = (COLOR_DICT[k] for k in set_labels)
        print(set_labels)
        if len(set_labels) == 3:
            v = venn3(
                subsets=subsets,
                subset_label_formatter=lambda x: "{:.1%}".format(x),
                set_labels=set_labels,
                set_colors=set_colors,
            )
            c = venn3_circles(subsets=subsets, linewidth=1)

            set_labelsAlpha = ["A", "B", "C"]
        elif len(set_labels) == 2:
            v = venn2(
                subsets=subsets,
                subset_label_formatter=lambda x: "{:.1%}".format(x),
                set_labels=set_labels,
                set_colors=set_colors,
            )
            c = venn2_circles(subsets=subsets, linewidth=1)

            set_labelsAlpha = ["A", "B"]
        for i, label in enumerate(set_labels):  # 给各适应症打上自身并集占比
            v.get_label_by_id(set_labelsAlpha[i]).set_text(
                "%s(%s)" % (label, "{:.1%}".format(share[i]))
            )

        # plt.annotate('无合并\n%s%s的\n%s患者' % (set_labels[1], set_labels[2], set_labels[0]), xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
        #             ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
        #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))

        plt.title(
            "%s\n\n总体%s" % ("/".join(set_labels), "{:.1%}".format(sum(subsets))),
            fontsize=18,
        )

        plt.savefig(
            "%s%s%s.png" % (self.savepath, self.name, "".join(set_labels)),
            format="png",
            bbox_inches="tight",
            transparent=True,
            dpi=600,
        )

    def plot_barh(self, groupby):
        df = self.get_union(groupby=groupby, len_set=1)
        df.rename(index={"": "无相关适应症"}, inplace=True)
        labels = self.labels.copy()
        labels.append("无相关适应症")
        df = df.reindex(labels).fillna(0)
        formats = ["{:.0%}"] * df.shape[1]
        plot_grid_barh(
            df=df,
            savefile="%s%s适应症贡献占比.png" % (self.savepath, groupby),
            formats=formats,
        )

        df = self.get_union(groupby=groupby, len_set=2, labels_in="高血压")
        df_undup_htn = self.get_intersect(groupby=groupby, len_set=1, labels_in="高血压")
        df_undup_htn.rename(index={"高血压": "单纯高血压"}, inplace=True)
        df = pd.concat([df_undup_htn, df], axis=0)
        labels = [
            "单纯高血压",
            "高血压+冠心病",
            "高血压+血脂异常",
            "高血压+糖尿病",
            "高血压+慢性肾病",
            "高血压+卒中",
            "高血压+高尿酸",
            "高血压+心力衰竭",
        ]
        df = df.reindex(labels).fillna(0)

        plot_grid_barh(
            df=df,
            savefile="%s%s%s高血压合并贡献占比.png" % (self.savepath, self.name, groupby),
            formats=formats,
        )

    def plot_ca(self, groupby, len_set=None, labels_in=None):
        ca = CA(n_components=2, n_iter=3, random_state=101)
        df = self.get_union(groupby=groupby, len_set=len_set, labels_in=labels_in)

        ca.fit(df)

        ax = ca.plot_coordinates(X=df, figsize=(20, 8))
        ax.get_legend().remove()

        plt.savefig(
            "%s%s%s对应分析图.png" % (self.savepath, self.name, groupby),
            format="png",
            bbox_inches="tight",
            transparent=True,
            dpi=600,
        )


if __name__ == "__main__":
    df = pd.read_excel("./data.xlsx")
    mask = (df["原始诊断"] != "无诊断") & (df["统计项"] == "标准片数") & (df["来源"] == "门诊")
    df = df.loc[mask, :]

    r = Rx(df, name="门诊标准片数")

    # print(r.get_ss_venn(("高血压", "冠心病")))
    # r.plot_barh("关注科室")
    # print(r.get_union(groupby="关注科室", len_set=2))
    r.plot_ca("关注科室")
