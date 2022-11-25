import sys
import os
from tkinter.messagebox import NO
from traceback import print_tb

import argparse
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="generate dataset")
    parser.add_argument("--store_path", type=str, default="/public/home/pw/workspace/LearnLawsInMatrix/dataset/eva/", help="dataset store path")
    parser.add_argument("--task", type=str, default="MIS", help="transpose,permutation,MIS")
    parser.add_argument("--matrix_num", type=int, default=80000, help="dataset number")
    parser.add_argument("--matrix_order", type=int, default=15, help="matrix_order")
    parser.add_argument("--matrix_lower", type=int, default=0, help="lower of number in random matrix")
    parser.add_argument("--matrix_upper", type=int, default=5, help="upper of number in random matrix")
    parser.add_argument("--split_rate", type=str, default="0.6,0.4", help="split_rate")
    parser.add_argument("--shuffle_path", type=str, default="dataset/permutation/permutation_5_shuffle", help="")

    return parser


def conver_matrix2graph(m, if_save=False):
    m_r, m_c = m.shape
    g = nx.Graph()

    for r in range(m_r):
        for c in range(m_c):
            if c > r and m[r, c] == 1:
                g.add_edge(c + 1, r + 1)

    if if_save:
        plt.subplot(111)
        nx.draw(g, with_labels=True, font_weight='bold')
        plt.savefig('graph' + str(random.randint(0, 9999)) + '.jpg')
        plt.close()

    return g


def gen_con_matrix(node_num):
    while True:
        rand_zo_matrix = np.random.randint(0, 2, [node_num, node_num])
        rand_graph_con_triu = np.triu(rand_zo_matrix)

        for i in range(rand_graph_con_triu.shape[0]):
            rand_graph_con_triu[i, i] = 0

        rand_graph_con = rand_graph_con_triu + rand_graph_con_triu.T
        connect_sta = np.sum(rand_graph_con, axis=0)
        if np.any(connect_sta == 0):

            continue
        else:
            break

    return rand_graph_con


def find_mis_rec(max_mis):
    for i in range(max_mis.shape[1]):
        mis_col = max_mis[:, i]
        if np.all(mis_col == mis_col[0]):
            continue
        else:
            fst_mis_col = min(mis_col)
            fst_mis = []
            for i, v in enumerate(mis_col):
                if v == fst_mis_col:
                    fst_mis.append(max_mis[i])

            if len(fst_mis) == 1:
                return fst_mis[0].tolist()
            else:
                return find_mis_rec(np.array(fst_mis))


def find_mis(mis):
    mis_list = [list(mi) for mi in mis]
    len_mis_list = len(mis_list)
    if len_mis_list == 0:
        return [], 0
    mis_list_len = [len(i) for i in mis_list]
    max_mis_list_len = max(mis_list_len)
    max_mis = []
    for i, v in enumerate(mis_list_len):
        if v == max_mis_list_len:
            mis_sort = mis_list[i]
            mis_sort.sort()
            max_mis.append(mis_sort)

    if len(max_mis) == 1:
        return max_mis[0], 1
    max_mis = np.array(max_mis)
    return find_mis_rec(max_mis), 1


def get_dataset_list(file_line, split_rate):
    split_rate = params.split_rate.split(',')
    split_size = [int(file_line * float(x)) for x in split_rate]
    file_idx = [i for i in range(int(file_line))]
    random.shuffle(file_idx)

    dataset_idx = {"train": file_idx[0:split_size[0]], "test": file_idx[split_size[0]:]}

    return dataset_idx


def get_transpose_matrix(origin_matrix, task, matrix_order, shuffle_idx=None):
    if task == "transpose":
        transpose_matrix = origin_matrix.T

    if task == "permutation":
        assert not shuffle_idx is None
        matrix_a = origin_matrix.reshape(1, -1)[0]
        matrix_b = np.ones_like(matrix_a) * -1
        assert len(shuffle_idx) == len(matrix_a)
        i = 0
        for ea in matrix_a:
            matrix_b[shuffle_idx[i]] = ea
            i += 1
        assert not matrix_b.__contains__(-1)
        transpose_matrix = matrix_b.reshape(origin_matrix.shape)

    if task == "MIS":
        indp_label = [1 for _ in range(matrix_order)]
        indep_node_list = []
        ver_rand_matrix = np.where((origin_matrix == 0) | (origin_matrix == 1), origin_matrix ^ 1, origin_matrix)
        for i in range(ver_rand_matrix.shape[0]):
            ver_rand_matrix[i, i] = 0
        rand_graph_con = conver_matrix2graph(ver_rand_matrix, if_save=False)
        mis = nx.find_cliques(rand_graph_con)
        indep_node_list, len_mis_list = find_mis(mis)
        if len_mis_list == 0:
            return 0

        for node_idx in indep_node_list:
            indp_label[node_idx - 1] = 0

        transpose_matrix = np.array(indp_label)

    return transpose_matrix


def main(params):

    # params
    path = params.store_path
    task = params.task
    matrix_num = params.matrix_num
    matrix_order = params.matrix_order
    matrix_lower = params.matrix_lower
    matrix_upper = params.matrix_upper + 1
    shuffle_path = params.shuffle_path

    filename = task + "_" + str(matrix_order) + "_" + str(matrix_num)
    file_path = path + task + '/' + filename

    shuffle_idx = []

    if task == "permutation":

        if os.path.exists(shuffle_path):
            with open(shuffle_path, 'r', encoding='utf-8') as f:
                data = f.read()
            shuffle_idx = [int(i) for i in data[1:-1].split(',')]
        else:
            shuffle_idx = [_ for _ in range(matrix_order**2)]
            random.shuffle(shuffle_idx)

            shuffle_idx_filepath = file_path + "_shuffle"
            file_shuffle = open(shuffle_idx_filepath, 'w')
            file_shuffle.writelines(str(shuffle_idx))
            file_shuffle.close()      


    file_obj = open(file_path, 'w')
    num_i = 0
    with tqdm(total=int(matrix_num)) as pbar:
        while num_i < matrix_num:

            if task == "transpose" or task == "permutation":
                origin_matrix = np.random.randint(matrix_lower, matrix_upper, (matrix_order, matrix_order))

            if task == "MIS":
                origin_matrix = gen_con_matrix(matrix_order)

            transpose_matrix = get_transpose_matrix(origin_matrix, task, matrix_order, shuffle_idx)
            if isinstance(transpose_matrix, int):
                continue

            num_i += 1
            s = ""
            for e in origin_matrix.reshape(1, -1)[0]:
                s += str(e) + " "
            s += "\t"
            for e in transpose_matrix.reshape(1, -1)[0]:
                s += str(e) + " "

            if not num_i == matrix_num:
                s += "\n"
            file_obj.write(s)

            pbar.update(1)

    file_obj.close()

    file_obj = open(file_path, 'r')
    file_line = len(file_obj.readlines())
    print(file_line)
    file_obj.close()


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)