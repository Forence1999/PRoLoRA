# -*- coding: utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 20/12/2023
# @Description:


import os
import sys
import time
import random
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def calculate_LoRA_params(L, d, D, r):
    """
    Calculate the parameters of LoRA for all linear layers.
    """
    qkvo = 4 * (d + d) * r * L
    up_gate = 2 * (d + D) * r * L
    down = (D + d) * r * L

    return qkvo + up_gate + down


def calculate_Tied_LoRA_params(L, d, D, r):
    """
    Calculate the parameters of Tied LoRA for QKV only.
    """
    matrix = 4 * d * r
    vector = (r + 3 * d) * L

    return matrix + vector


def calculate_VeRA_params(L, d, D, r):
    """
    Calculate the parameters of LoRA for all linear layers.
    """
    qkvo = 4 * (r + d) * L
    up_gate = 2 * (r + D) * L
    down = (r + d) * L

    return qkvo + up_gate + down


def calculate_Partially_shared_LoRA_params(L, d, D, r, n, x_A, x_B):
    """
    Calculate the parameters of Partially-shared LoRA for all linear layers.
    """
    qkvo = 4 * ((d + d) * n + d * (r - n) / x_A + d * (r - n) / x_B) * L
    up_gate = 2 * ((d + D) * n + d * (r - n) / x_A + D * (r - n) / x_B) * L
    down = ((D + d) * n + D * (r - n) / x_A + d * (r - n) / x_B) * L

    return qkvo + up_gate + down


def plot_params(data):
    """
    Plot the parameters of LoRA, Tied LoRA, VeRA, and Partially-shared LoRA.
    """

    # 假设你有两个方法的rank和params数据
    method1_rank = [1, 2, 3, 4, 5]
    method1_params = [10, 5, 2, 1, 0.5]

    method2_rank = [1, 2, 3, 4, 5]
    method2_params = [20, 10, 5, 2, 1]

    # 绘制曲线
    for name, data_ in data.items():
        plt.plot(data_["rank"], data_["params"], label=name, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Params vs Rank")
    plt.xlabel("Rank")
    plt.ylabel("Params")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    d = 4096
    D = 11008
    # r = 8
    L = 32
    n = 1  # number of ranks not shared in Partially-shared LoRA
    x_A = 15  # sharing ratio of lora_A
    x_B = 15  # sharing ratio of lora_B
    # assert x_A <= r - n, "x_A is too large"
    # assert x_B <= r - n, "x_B is too large"

    data = {
        # "method": {
        #     "rank": list of ranks,
        #     "params": list of params,
        # }
        "LoRA": {
            "rank": [],
            "params": [],
        },
        "Tied_LoRA": {
            "rank": [],
            "params": [],
        },
        "VeRA": {
            "rank": [],
            "params": [],
        },
        "Partially-shared_LoRA": {
            "rank": [],
            "params": [],
        },
    }

    for i in range(15):
        r = 2**i
        data["LoRA"]["rank"].append(r)
        data["LoRA"]["params"].append(calculate_LoRA_params(L, d, D, r))
        data["Tied_LoRA"]["rank"].append(r)
        data["Tied_LoRA"]["params"].append(calculate_Tied_LoRA_params(L, d, D, r))
        data["VeRA"]["rank"].append(r)
        data["VeRA"]["params"].append(calculate_VeRA_params(L, d, D, r))
        # print(data["VeRA"]["params"][-1])
        data["Partially-shared_LoRA"]["rank"].append(r)
        data["Partially-shared_LoRA"]["params"].append(
            calculate_Partially_shared_LoRA_params(L, d, D, r, n, x_A, x_B)
        )

    plot_params(data)
