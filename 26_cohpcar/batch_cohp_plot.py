# @Time    : 2025/8/21 22:44
# @Author  : JunFei Cai
# @File    : batch_cohp_plot.py

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os

from warnings import simplefilter
# 忽略FutureiWarning
simplefilter(action='ignore', category=FutureWarning)

plot_params = {
    "font.family": "Arial",
    "figure.dpi": 300,
    "savefig.format": 'svg',
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.labelweight": "bold"
}

plt.rcParams.update(plot_params)

def cohp_extract(path):
    """
    读取cohp.lobster或者coop.lobster中的数据，分割数据后，把cohp中的作用对和每一列数据对应上
    本次使用只考虑了有考虑自旋的数据
    :param path: 数据文件的路径
    :return: 处理好的数据，dataframe类型。
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    atom_pair = []
    cohp_data = []
    for i in range(3, len(lines)):
        line_str = lines[i].replace("\n", "")
        if lines[i].startswith("No"):
            atom_pair.append(line_str)

        else:
            energy_data = [x for x in line_str.split(" ") if x != ""]
            cohp_data.append(energy_data)

    pair_num = len(atom_pair)
    # data_num = 4 * pair_num + 5
    column_name = []
    column_name.append("Energy")
    column_name.append("average_pCOHP_up")
    column_name.append("average_IpCOHP_up")
    for i in range(pair_num):
        num = i + 1
        column_name.append("pCOHP_up_" + str(num))
        column_name.append("IpCOHP_up_" + str(num))

    column_name.append("average_pCOHP_down")
    column_name.append("average_IpCOHP_down")
    for i in range(pair_num):
        num = i + 1
        column_name.append("pCOHP_down_" + str(num))
        column_name.append("IpCOHP_down_" + str(num))

    data_array = np.array(cohp_data, dtype=float)
    processed_data = pd.DataFrame(data_array, columns=column_name)
    return processed_data


def plot_cohp(cohp_data, ax, parameter_dict,
              pair_index="average",
              ):
    """
    在axes画图。
    :param cohp_data: 通过cohp_extract进行处理过的cohp数据
    :param ax: 通过索引得到axes
    :param pair_index: 对应cohp中的原子对,默认取平均值，也可以通过数字指定，注意从1开始而不是0
    :param parameter_dict: 绘图的参数，标题啊啥的
    """
    if pair_index == "average":
        plot_data = cohp_data.loc[:, ["Energy", "average_pCOHP_up", "average_pCOHP_down"]]

    else:
        column_index_up = "pCOHP_up_" + str(pair_index)
        column_index_down = "pCOHP_down_" + str(pair_index)
        plot_data = cohp_data.loc[:, ["Energy", column_index_up, column_index_down]]

    plot_data["Y1"] = plot_data.iloc[:, 1]  # * -1  # cohp乘以一个负值
    plot_data["Y2"] = plot_data.iloc[:, 2]  # * -1  # cohp乘以一个负值
    # 获取绝对值最大的数字用于设置纵轴范围
    max_abs = plot_data.iloc[:, 1:3].abs().max().max() * 1.2

    # 上自旋曲线
    sns.lineplot(data=plot_data,
                 x="Energy",
                 y="Y1",
                 label=parameter_dict["curve_label"][0],
                 color=parameter_dict["curve_color"][0],
                 linewidth=2.5,
                 ax=ax)
    # 下自旋曲线
    sns.lineplot(data=plot_data,
                 x="Energy",
                 y="Y2",
                 label=parameter_dict["curve_label"][1],
                 color=parameter_dict["curve_color"][1],
                 linewidth=2.5,
                 ax=ax)

    ax.legend(loc="best")  # 自动调整选择最合适的子图图例

    ax.set_xticks(parameter_dict["xticks"])

    ax.axvline(x=0, color="black", linestyle="--", label="Fermi energy")
    ax.axhline(y=0, color="black", linestyle="--")
    # ax.set_title(parameter_dict["title"])
    ax.set_xlabel(parameter_dict["xlabel"])
    ax.set_ylabel(parameter_dict["ylabel"])
    ax.set_xlim(parameter_dict["xlim"])
    ax.set_ylim(-max_abs, max_abs)

current_dir = os.getcwd()
subdirs = [d for d in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, d)) and d.endswith(".dat")]
# for subdir in subdirs:


x_stick = [i for i in range(-10, 10, 2)]
parameter_dict = {"xlim": (-10, 10),
                          "xticks": x_stick,
                          "curve_label": ("-up",  "-down"),
                          "curve_color": ("red", "blue"),
                          # "title": title_str,
                          "xlabel": "Energy $E-E_f$",
                          "ylabel": "-COHP"}


for subdir in subdirs:
    path = os.path.join(current_dir, subdir)
    processed_data = cohp_extract(path)
    pair_index = subdir.split("_COHP.dat")[0]
    parameter_dict["title"] = pair_index

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_cohp(processed_data, ax=ax, parameter_dict=parameter_dict, pair_index=1)
    file_name = "Li1.2Ni0.2Mn0.6O2_" + pair_index + "_COHP.svg"
    plt.savefig(os.path.join(current_dir, file_name))
    plt.close(fig)