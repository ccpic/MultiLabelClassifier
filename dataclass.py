import pandas as pd
import numpy as np
from itertools import chain, combinations
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from matplotlib_venn import venn3, venn3_circles, venn2, venn2_circles
import matplotlib.font_manager as fm
from chart_func import plot_grid_barh


# 设置matplotlib正常显示中文和负号
mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
mpl.rcParams["axes.unicode_minus"] = False  # 正常显示负号

D_SORTER = {
    "关注科室": ["心内科", "老干科", "普内科", "肾内科", "神内科", "内分泌科", "其他"],
    "通用名": ["缬沙坦", "厄贝沙坦", "氯沙坦", "替米沙坦", "坎地沙坦", "奥美沙坦酯", "阿利沙坦酯", "培哚普利", "贝那普利",],
    "商品名": ["代文", "安博维", "科素亚", "美卡素", "必洛斯", "傲坦", "信立坦", "雅施达", "洛汀新"],
    "来源": ["门诊", "病房"],
}


def all_subsets(ss):
    return list(chain(*map(lambda x: combinations(ss, x), range(1, len(ss) + 1))))


def refine(df, len_set=None, labels_in=None, change_index=False):
    if len_set is not None:  # 返回指定标签数量的结果
        # if type(df.index) == tuple:
        #     df = df[df.index.map(len) == len_set]
        df = df[df.index.map(lambda x: x.count("+")) == len_set - 1]

    if labels_in is not None:  # 返回含有指定标签的结果
        df = df[df.index.map(lambda x: "高血压" in x) == True]

    if change_index is True:  # 行标签由tuple改为+号相连的字符串
        df.index = df.index.map(lambda x: "+".join(x) if type(x) is tuple else x)

    if "占比" in df.columns:
        df.sort_values(by="占比", ascending=False, inplace=True)
    else:
        df.sort_values(df.columns[0], ascending=False, inplace=True)

    df.fillna(0, inplace=True)

    return df


def get_undup_cbns(df, labels):
    total_number = df["统计值"].sum()
    df_undup_cbns = df.loc[:, labels].apply(
        lambda s: [s.name if v == 1 else np.nan for v in s]
    )
    df_undup_cbns = df_undup_cbns.T.apply(lambda x: "+".join(x.dropna().tolist()))
    df_undup_cbns = pd.concat([df_undup_cbns, df["统计值"]], axis=1)
    df_undup_cbns.columns = ["高血压合并症", "统计值"]
    df_undup_cbns = df_undup_cbns.groupby(["高血压合并症"]).sum()
    df_undup_cbns["占比"] = df_undup_cbns["统计值"] / total_number

    return df_undup_cbns


def get_dup_cbns(df):
    df.reset_index(inplace=True)
    df["高血压合并症"] = df["高血压合并症"].apply(lambda x: all_subsets(tuple(x.split("+"))))

    df_dup_cbns = df.explode("高血压合并症").groupby("高血压合并症").sum()
    df_dup_cbns.index = df_dup_cbns.index.map(
        lambda x: "+".join(x) if type(x) is tuple else x
    )

    return df_dup_cbns


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

    def get_undup_cbns(
        self, change_index=False, len_set=None, labels_in=None
    ):  # 获得指定条件的所有标签组合（互不相交，加和=100%）
        df_undup_cbns = get_undup_cbns(self, self.labels)
        df_undup_cbns = refine(
            df=df_undup_cbns,
            len_set=len_set,
            labels_in=labels_in,
            change_index=change_index,
        )

        return df_undup_cbns

    def get_undup_cbns_groupby(
        self, groupby, len_set=None, labels_in=None
    ):  # 获得指定条件的所有标签组合的分组矩阵
        cols = self[groupby].unique()

        for i, col in enumerate(cols):
            df_undup_cbns = get_undup_cbns(self[self[groupby] == col], self.labels)[
                "占比"
            ]
            if i == 0:
                df_undup_cbns_groupby = df_undup_cbns
            else:
                df_undup_cbns_groupby = pd.concat(
                    [df_undup_cbns_groupby, df_undup_cbns], axis=1
                )
        df_undup_cbns_groupby.columns = cols

        if groupby in D_SORTER:
            try:
                df_undup_cbns_groupby = df_undup_cbns_groupby[
                    D_SORTER[groupby]
                ]  # 对于部分变量有固定列排序
            except KeyError:
                pass

        df_undup_cbns_groupby = refine(
            df=df_undup_cbns_groupby, len_set=len_set, labels_in=labels_in
        )

        return df_undup_cbns_groupby

    def get_dup_cbns(
        self, change_index=False, len_set=None, labels_in=None
    ):  # 获得指定条件的所有标签组合的并集
        df_undup_cbns = self.get_undup_cbns()
        df_dup_cbns = get_dup_cbns(df_undup_cbns)

        df_dup_cbns = refine(
            df=df_dup_cbns,
            len_set=len_set,
            labels_in=labels_in,
            change_index=change_index,
        )

        return df_dup_cbns

    def get_dup_cbns_groupby(
        self, groupby, len_set=None, labels_in=None
    ):  # 获得指定条件的所有标签组合的分组矩阵
        cols = self[groupby].unique()

        for i, col in enumerate(cols):
            df_dup_cbns = get_dup_cbns(
                get_undup_cbns(self[self[groupby] == col], self.labels)
            )["占比"]
            if i == 0:
                df_dup_cbns_groupby = df_dup_cbns
            else:
                df_dup_cbns_groupby = pd.concat(
                    [df_dup_cbns_groupby, df_dup_cbns], axis=1
                )
        df_dup_cbns_groupby.columns = cols

        if groupby in D_SORTER:
            try:
                df_dup_cbns_groupby = df_dup_cbns_groupby[
                    D_SORTER[groupby]
                ]  # 对于部分变量有固定列排序
            except KeyError:
                pass

        df_dup_cbns_groupby = refine(
            df=df_dup_cbns_groupby, len_set=len_set, labels_in=labels_in
        )

        return df_dup_cbns_groupby

    def get_cbns_len(self):
        df_undup_cbns = self.get_undup_cbns()
        df_undup_cbns.reset_index(inplace=True)
        df_undup_cbns["高血压合并症"] = df_undup_cbns["高血压合并症"].apply(
            lambda x: len(tuple(x.split("+")))
        )

        df_cbns_len = df_undup_cbns.groupby(["高血压合并症"]).sum()

        return df_cbns_len

    def get_cbns_len_groupby(self, groupby):

        cols = self[groupby].unique()

        for i, col in enumerate(cols):
            df_undup_cbns = get_undup_cbns(self[self[groupby] == col], self.labels)[
                "占比"
            ].to_frame()
            df_undup_cbns.reset_index(inplace=True)
            df_undup_cbns["高血压合并症"] = df_undup_cbns["高血压合并症"].apply(
                lambda x: len(tuple(x.split("+")))
            )
            df_cbns_len = df_undup_cbns.groupby(["高血压合并症"]).sum()
            if i == 0:
                df_cbns_len_groupby = df_cbns_len
            else:
                df_cbns_len_groupby = pd.concat(
                    [df_cbns_len_groupby, df_cbns_len], axis=1
                )
        df_cbns_len_groupby.columns = cols

        if groupby in D_SORTER:
            try:
                df_cbns_len_groupby = df_cbns_len_groupby[
                    D_SORTER[groupby]
                ]  # 对于部分变量有固定列排序
            except KeyError:
                pass

        return df_cbns_len_groupby

    def get_ss_venn(self, sets):
        all_ss = list(map(lambda x: "+".join(x), all_subsets(sets)))
        ss = self.get_dup_cbns().loc[all_ss, "占比"]

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
            "%s%s.png" % (self.savepath, "".join(set_labels)),
            format="png",
            bbox_inches="tight",
            transparent=True,
            dpi=600,
        )

    def plot_barh(self, groupby):
        df = self.get_dup_cbns_groupby(groupby=groupby, len_set=1)
        df.rename(index={'':'无相关适应症'},inplace=True)
        labels = self.labels.copy()
        labels.append("无相关适应症")
        df = df.reindex(labels)
        formats =  ["{:.0%}"] * df.shape[1]
        plot_grid_barh(df=df, savefile="%s%s适应症贡献占比.png" % (self.savepath, groupby), formats=formats)

        df = self.get_dup_cbns_groupby(groupby=groupby, len_set=2, labels_in="高血压")
        plot_grid_barh(df=df, savefile="%s%s高血压合并贡献占比.png" % (self.savepath, groupby), formats=formats)

        
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    df = pd.read_excel("./data.xlsx")
    mask = (df["原始诊断"] != "无诊断") & (df["统计项"] == "标准片数") & (df["来源"] == "门诊")
    df = df.loc[mask, :]

    r = Rx(df, name="门诊标准片数")
    filter = {"关注科室": ["心内科"]}
    # print(r.get_ss_venn(("高血压", "冠心病")))
    r.plot_barh_dup_cbns("关注科室")
