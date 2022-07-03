from read_dag import read_dag, read_dag_adj
import operator
import collections
import numpy as np
from random import randint, gauss
from typing import List
import random

from gurobipy import *
from typing import List
from Task import Task
from Processor import Processor


softlimit = 5
hardlimit = 100


def softtime(model, where):
    if where == GRB.Callback.MIP:
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        gap = abs((objbst - objbnd) / objbst)

        if runtime > softlimit and gap < 0.5:
            model.terminate()


def solveNLP(processSpeed: List[List[int]], taskWorkLoad: List[int]):
    # @description: 解决NLP问题
    # @param processSpeed: 二维数组，存储处理器运行任务时的速度。处理器 i 运行任务 j 的速度是 processSpeed[i][j] = s[i][j]
    # @param taskWorkLoad: 一维数组，存储任务载荷。任务 j 在处理器 i 上的运行时间是 p[i][j] =  taskWorkLoad[j] / processSpeed[i][j]

    M, N = len(processSpeed), len(taskWorkLoad)
    p = [[taskWorkLoad[j] / processSpeed[i][j] for j in range(N)]
         for i in range(M)]

    model = Model('nonlinear scheduling model')

    target = model.addVar(lb=-GRB.INFINITY,
                          ub=GRB.INFINITY,
                          vtype=GRB.CONTINUOUS,
                          name='TARGET')
    T = [model.addVar(lb=0,
                      ub=GRB.INFINITY,
                      vtype=GRB.CONTINUOUS,
                      name=f'T{j}')
         for j in range(N)]
    x = [[model.addVar(lb=0,
                       ub=1,
                       vtype=GRB.BINARY,
                       name=f'x{i}{j}') for j in range(N)]
         for i in range(M)]
    c = [[model.addVar(lb=0,
                       ub=1,
                       vtype=GRB.BINARY,
                       name=f'x{j}{k}') for k in range(N)]
         for j in range(N)]
    o = [[model.addVar(lb=0,
                       ub=1,
                       vtype=GRB.BINARY,
                       name=f'o{j}{k}') for k in range(N)]
         for j in range(N)]
    before = [[model.addVar(lb=0,
                            ub=1,
                            vtype=GRB.BINARY,
                            name=f'order{j}{k}') for k in range(N)]
              for j in range(N)]
    after = [[model.addVar(lb=0,
                           ub=1,
                           vtype=GRB.BINARY,
                           name=f'order{j}{k}') for k in range(N)]
             for j in range(N)]

    model.setObjective(target, GRB.MAXIMIZE)
    k, offset = 1, 0
    model.addQConstr((target == - quicksum(k * t for t in T) + offset),
                     "target")

    model.addConstrs(((quicksum(x[i][j] for i in range(M)) == 1) for j in range(N)),
                     "cpu_allocate")

    model.addConstrs((T[j] >= quicksum(x[i][j] * p[i][j] for i in range(M)) for j in range(N)),
                     "process_time_limit")

    for j in range(N):
        for k in range(N):
            model.addQConstr(c[j][k] == quicksum(
                x[i][j] * x[i][k] for i in range(M)))

    for j in range(N):
        for k in range(N):
            model.addQConstr(before[j][k] == o[j][k] * c[j][k])
            model.addQConstr(after[j][k] == (1 - o[j][k]) * c[j][k])

    for k in range(N):
        for j in range(k):
            model.addQConstr(
                T[k] - quicksum(before[j][k] * x[i][k] * p[i][k] for i in range(M)) - before[j][k] * T[j] >= 0, "order_limit")
            model.addQConstr(
                T[j] - quicksum(after[j][k] * x[i][j] * p[i][j] for i in range(M)) - after[j][k] * T[k] >= 0, "order_limit")

    model.setParam('TimeLimit', hardlimit)
    model.optimize(softtime)

    # print('-----------------------------------------------------------------')
    # print('Optimal Obj: {}'.format(model.ObjVal))
    # print('-----------------------------------------------------------------')
    min_makespan = float("inf")
    job_no = 0

    for j in range(N):
        print('T{} = {}'.format(j, T[j].x))
        if T[j].x < min_makespan:
            min_makespan = min(T[j].x, min_makespan)
            job_no = j

    return [min_makespan, job_no]


class Solution:
    def __init__(self, input_list=None, file=None, verbose=False, processors=3, b=0.5, ccr=0.5):
        if input_list is None and file is not None:
            self.num_tasks, self.num_processors, self.sizes, self.edges = read_dag_adj(
                file, processors, b, ccr)

        if verbose:
            print("No. of Tasks: ", self.num_tasks)
            print("No. of processors: ", self.num_processors)
            # print("Adjacent Graph: ", self.edges)
            # print("Computational Cost Matrix:")
            # for i in range(self.num_tasks):
            #     print(comp_cost[i])
            # print("Graph Matrix:")
            # for line in self.graph:
            #     print(line)

        indeg = [0] * (self.num_tasks + 2)

        for s in self.edges:
            for t in self.edges[s]:
                indeg[t] += 1
            # source, dest = int(e.get_source()), int(e.get_destination())
            # edges[source].append(dest)
        # self.tasks = [Task(i) for i in range(self.num_tasks)]
        # self.processors = [Processor(i) for i in range(self.num_processors)]

        queue = collections.deque(
            [u for u in range(self.num_tasks) if indeg[u] == 0])

        # result = list()
        makespan = 0.0

        while queue:
            N = len(queue)
            tasks = []
            orders = []
            # taskWorkLoad = [0] * N
            processSpeed = [[1] * N for _ in range(processors)]
            free_processors = processors - N

            for u in queue:
                tasks.append(self.sizes[u])
                orders.append(u)
                # for v in self.edges[u]:
                #     indeg[v] -= 1
                #     if indeg[v] == 0:
                #         queue.insert(0, v)
            min_makespan, job_no = solveNLP(processSpeed, tasks)
            # TODO:
            # 该层最小完成时间，任务编号
            print(min_makespan, job_no)

            for _ in range(N):
                u = queue.popleft()
                for v in self.edges[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        queue.append(v)
        # print(ans)


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
                    help="DAG description as a .dot file")
    args = ap.parse_args()

    new_sch = Solution(file=args.input, verbose=True,
                       processors=3, b=0.1, ccr=0.1)
