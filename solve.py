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
hardlimit = 300


def softtime(model, where):
    if where == GRB.Callback.MIP:
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        gap = abs((objbst - objbnd) / objbst)

        if runtime > softlimit and gap < 0.5:
            model.terminate()


def solveNLP(orders: List[int], start: List[float], processSpeed: List[List[int]], taskWorkLoad: List[int]):
    # @description: 解决NLP问题
    # @param processSpeed: 二维数组，存储处理器运行任务时的速度。处理器 i 运行任务 j 的速度是 processSpeed[i][j] = s[i][j]
    # @param taskWorkLoad: 一维数组，存储任务载荷。任务 j 在处理器 i 上的运行时间是 p[i][j] =  taskWorkLoad[j] / processSpeed[i][j]

    M, N = len(processSpeed), len(taskWorkLoad)
    p = [[taskWorkLoad[j] / processSpeed[i][j] for j in range(N)]
         for i in range(M)]
    env = Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = Model('nonlinear scheduling model', env=env)

    target = model.addVar(lb=-GRB.INFINITY,
                          ub=GRB.INFINITY,
                          vtype=GRB.CONTINUOUS,
                          name='TARGET')
    T = [model.addVar(lb=start[j],
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

    min_makespan = float("inf")
    max_makespan = 0
    min_job_idx = 0
    jobs = [Task(orders[j]) for j in range(N)]
    cpus = collections.defaultdict(list)

    for j in range(N):
        jobs[j].duration['start'] = start[j]
        jobs[j].duration['end'] = T[j].x
        for k in range(M):
            if x[k][j].x == 1:
                cpus[k].append(jobs[j])

        max_makespan = max(T[j].x, max_makespan)
        if T[j].x < min_makespan:
            min_makespan = min(T[j].x, min_makespan)
            min_job_idx = j

    return [max_makespan, min_makespan, min_job_idx, jobs, cpus]


class Solution:
    def __init__(self, input_list=None, file=None, verbose=False, processors=3, b=0.5, ccr=0.5):
        if input_list is None and file is not None:
            self.num_tasks, self.num_processors, self.sizes, self.edges = read_dag_adj(
                file, processors, b, ccr)

        self.tasks = [Task(i) for i in range(self.num_tasks + 2)]

        if verbose:
            print("No. of Tasks: ", self.num_tasks)
            print("No. of processors: ", self.num_processors)

        indeg = [0] * (self.num_tasks + 2)

        for s in self.edges:
            for t in self.edges[s]:
                indeg[t] += 1

        queue = [u for u in range(self.num_tasks) if indeg[u] == 0]

        u = queue.pop(0)
        for v in self.edges[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

        N = len(queue)
        processSpeed = [[1] * N for _ in range(processors)]
        makespan = 0.0
        cur_time = float('inf')
        starts = [0] * len(queue)
        tasks = []
        orders = []

        for u in queue:
            tasks.append(self.sizes[u])
            orders.append(u)

        max_makespan, min_makespan, min_job_idx, jobs, cpus = solveNLP(
            orders, starts,  processSpeed, tasks)

        # makespan += max_makespan

        # cpus: CPU 分配情况
        # jobs: 每个任务的完成时间
        # min_makespan: 第一个完成的任务时间
        # min_job_idx: 第一个完成的任务
        print(cpus.items())
        # print(jobs)

        for job in jobs:
            print('T{}, start: {}, end: {}'.format(
                job.id, job.duration['start'], job.duration['end']))

        print(min_makespan, min_job_idx)
        for _, cpu_task in cpus.items():
            for t in cpu_task:
                print(t.id)
        # cpus_time = collections.defaultdict(list)
        # for task in orders:
        #     print('T{} = {}'.format(task, jobs[task]))
        # 抢占：queue 移走第一个完成的
        # 非抢占：
        # for _ in range(N):
        #     u = queue.pop()
        #     for v in self.edges[u]:
        #         indeg[v] -= 1
        #         if indeg[v] == 0:
        #             queue.append(v)
        # print('Makespan = {}'.format(makespan))

        # processSpeed = [[1] * self.num_tasks for _ in range(processors)]
        # tasks = [self.sizes[i + 1] for i in range(self.num_tasks)]
        # max_makespan, min_makespan, min_job_idx, jobs, cpus = solveNLP(
        #     [i + 1 for i in range(self.num_tasks)], processSpeed, tasks)
        # print('Makespan should be', max(jobs.items(), key=lambda a: a[1]))


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
                    help="DAG description as a .dot file")
    args = ap.parse_args()

    new_sch = Solution(file=args.input, verbose=True,
                       processors=2, b=0.1, ccr=0.1)
