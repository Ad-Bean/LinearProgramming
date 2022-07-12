import collections
from gurobipy import *
from typing import List
from Task import Task
from read_dag import read_dag_adjacency
import matplotlib.pyplot as plt

softlimit = 5
hardlimit = 120


def softtime(model, where):
    if where == GRB.Callback.MIP:
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        gap = abs((objbst - objbnd) / objbst)

        if runtime > softlimit and gap < 0.5:
            model.terminate()


def get_sub_set(nums: List[int]):
    sub_sets = [[]]
    for x in nums:
        arr = [item + [x] for item in sub_sets]
        sub_sets.extend(arr)
    sub_sets.remove([])
    return sub_sets


def solveNLP(processSpeed: List[List[float]], taskWorkLoad: List[float], graph: List[List[int]], preset: List[List[int]], z: float):
    # @description: 解决NLP问题
    # @param processSpeed: 二维数组，存储处理器运行任务时的速度。处理器 i 运行任务 j 的速度是 processSpeed[i][j] = s[i][j]
    # @param taskWorkLoad: 一维数组，存储任务载荷。任务 j 在处理器 i 上的运行时间是 p[i][j] =  taskWorkLoad[j] / processSpeed[i][j]
    # @param graph:二维数组，graph[j][k] == 1 代表任务 j -> k 存在一条有向边（ j ， k 邻接，且 j 需要在 k 之前执行）

    # ---------- 0.初始化变量 -----------------------------------
    # M = 处理器数量；N = 任务数量
    M, N = len(processSpeed), len(taskWorkLoad)

    # 特判
    if N == 0 or len(graph) != N or len(graph[0]) != N:
        print('数组 taskWorkLoad 的长度和 graph 的两个维度的长度需要相同，均等于任务数量')
        return None

    # 初始化运行时间数组：p[i][j]代表任务 j 在处理器 i 上的运行时间
    p = [[taskWorkLoad[j] / processSpeed[i][j] for j in range(N)]
         for i in range(M)]

    # ---------- 1.创建模型和变量---------------------------------
    # 创建模型
    env = Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = Model('nonlinear scheduling model', env=env)
    # model = Model('nonlinear scheduling model')

    # 1.1.创建目标函数变量：target = sum(Uj(T[j])) for j in N
    target = model.addVar(lb=-GRB.INFINITY,
                          ub=GRB.INFINITY,
                          vtype=GRB.CONTINUOUS,
                          name='TARGET')
    # 1.2.创建优化变量数组T：存储完成时间(T[j])
    T = [model.addVar(lb=0,
                      ub=GRB.INFINITY,
                      vtype=GRB.CONTINUOUS,
                      name=f'T{j}')
         for j in range(N)]
    # 1.3.创建任务分配变量数组x：x[i][j]为01变量。x[i][j] == 1代表任务 j 被分配到处理器 i 中
    x = [[model.addVar(lb=preset[j][i],
                       ub=1,
                       vtype=GRB.BINARY,
                       name=f'x{i}{j}') for j in range(N)]
         for i in range(M)]

    # 1.4.辅助变量数组c：c[j][k] == 1 代表任务 j 和任务 k 使用着同一个CPU
    c = [[model.addVar(lb=0,
                       ub=1,
                       vtype=GRB.BINARY,
                       name=f'x{j}{k}') for k in range(N)]
         for j in range(N)]
    # 1.5.辅助变量数组o：o[j][k] == 1 代表任务 j 在任务 k 之前
    o = [[model.addVar(lb=0,
                       ub=1,
                       vtype=GRB.BINARY,
                       name=f'o{j}{k}') for k in range(N)]
         for j in range(N)]
    # 1.6.辅助变量数组before：before[j][k] == 1 代表任务 j 在任务 k 同CPU，且 j 在 k 之前
    before = [[model.addVar(lb=0,
                            ub=1,
                            vtype=GRB.BINARY,
                            name=f'order{j}{k}') for k in range(N)]
              for j in range(N)]
    # 1.7.辅助变量数组after：after[j][k] == 1 代表任务 j 在任务 k 同CPU，且 j 在 k 之后
    after = [[model.addVar(lb=0,
                           ub=1,
                           vtype=GRB.BINARY,
                           name=f'order{j}{k}') for k in range(N)]
             for j in range(N)]

    # 1.8.辅助变量数组 pt[j] 存储 pj
    pt = [model.addVar(lb=0,
                       ub=GRB.INFINITY,
                       vtype=GRB.CONTINUOUS,
                       name=f'T{j}')
          for j in range(N)]

    # ---------- 2.设置约束 --------------------------------

    # 2.0.设置目标函数约束：target = max - sum(k * T[j]) + offset for j in N
    model.setObjective(target, GRB.MAXIMIZE)
    # 添加目标函数约束
    k, offset = 1, 10000
    model.addQConstr((target == - quicksum(k * t for t in T) + offset),
                     "target")

    # # 设置目标函数约束：target = max sum(1 / T[j]) for j in N
    # model.setObjective(target, GRB.MAXIMIZE)
    # # 设置辅助变量 dt[i] = 1 / T[j]
    # dt = [None] * N
    # for j in range(N):
    #     dt[j] = model.addVar(lb=0,
    #                          ub=GRB.INFINITY,
    #                          vtype=GRB.CONTINUOUS,
    #                          name=f'dt{j}=1/T{j}')
    #     model.addGenConstrPow(T[j], dt[j], -1)
    # # 添加目标函数约束
    # model.addQConstr(target == quicksum(c for c in dt),
    #                  "target")

    # 2.1.约束条件 sum(x[i][j]) == 1 for j in N
    model.addConstrs(((quicksum(x[i][j] for i in range(M)) == 1) for j in range(N)),
                     "cpu_allocate")

    # 2.2.约束条件 T[j] >= p[j] for j in N
    model.addConstrs((T[j] >= quicksum(x[i][j] * p[i][j] for i in range(M)) for j in range(N)),
                     "process_time_limit")

    # 2.3.辅助变量c(同CPU标志位)约束 c[j][k] = sum(x[i][j] * x[i][k]) for i in M
    for j in range(N):
        for k in range(N):
            model.addQConstr(c[j][k] == quicksum(
                x[i][j] * x[i][k] for i in range(M)))

    # 2.4.约束条件 T[k] >= p[k] + T[j] for j before k
    for k in range(N):
        for j in range(N):
            model.addConstr((T[k] - graph[j][k] * quicksum(x[i][k] * p[i][k] for i in range(M)) - graph[j][k] * T[j] >= 0),
                            "order_limit")

    # 2.4.辅助变量(顺序变量)约束
    before[j][k] = o[j][k] * c[j][k]
    after[j][k] = (1 - o[j][k]) * c[j][k]
    for j in range(N):
        for k in range(N):
            model.addQConstr(before[j][k] == o[j][k] * c[j][k])
            model.addQConstr(after[j][k] == (1 - o[j][k]) * c[j][k])

    # 2.5.约束条件 (T[k] >= p[k] + T[j]) or (T[j] >= p[j] + T[k]) for independent (j,k) pairs
    for k in range(N):
        for j in range(k):
            model.addQConstr(
                T[k] - quicksum(before[j][k] * x[i][k] * p[i][k] for i in range(M)) - before[j][k] * T[j] >= 0, "order_limit")
            model.addQConstr(
                T[j] - quicksum(after[j][k] * x[i][j] * p[i][j] for i in range(M)) - after[j][k] * T[k] >= 0, "order_limit")

    # # # 2.5.松弛
    # # 2.5.1 辅助约束条件 pt: pt[j] 任务 j 的完成时间
    # model.addConstrs((pt[j] == quicksum(x[i][j] * p[i][j] for i in range(M))
    #                  for j in range(N)), "temp_process_time_limit")

    # # 2.5.2 z = max{s[i][j] / s[i'][j']}

    # # 2.5.3  forall S sub N, pS = sum pj for j in S
    # sub_tasks = get_sub_set([i for i in range(N)])
    # for S in sub_tasks:
    #     model.addConstr(
    #         (quicksum(pt[j] * T[j] for j in range(N)) >=
    #          (quicksum(pt[j] for j in S) ** 2 +
    #           quicksum(pt[j] * pt[j] for j in S))
    #          / (2 * len(S) * z)),
    #         "relaxed_constrs")

    # ---------- 3.求解 -----------------------------------
    model.setParam('TimeLimit', hardlimit)
    model.optimize(softtime)
    # model.optimize()

    # for j in range(N):
    # print('T{} = {}'.format(j + 1, T[j].x))

    cpus = collections.defaultdict(list)
    jobs = [Task(j + 1) for j in range(N)]

    for j in range(N):
        jobs[j].duration['end'] = T[j].x
        for k in range(M):
            if x[k][j].x == 1:
                cpus[k].append(jobs[j])

    return [model.ObjVal, cpus, jobs]


def solution():
    # ./daggen -n 25 --fat 0.4 --density 0.4 --regular 0.2 --jump 2 --minalpha 20 --maxalpha 200 --dot -o ../task25.dot
    files = ['task20.dot', 'task21.dot', 'task22.dot', 'task23.dot', 'task24.dot', 'task25.dot',
             'task26.dot', 'task27.dot', 'task28.dot', 'task29.dot', 'task30.dot', 'task40.dot']
    makespan_dots = []
    utility_dots = []
    for filename in files:
        num_tasks, workloads, adj_matrix = read_dag_adjacency(filename)
        M, N = 4, num_tasks  # M = 处理器数量；N = 任务数量
        processSpeed = [[1 + i] * N for i in range(M)]  # 异构处理器速度
        presets = [[0] * M for _ in range(N)]
        final_utility = float('-inf')
        final_makespan = float('inf')

        for i in range(num_tasks):
            cur_adj_matrix = adj_matrix[: i + 1, : i + 1]      # 当前 i 个任务的邻接矩阵
            cur_utility = float('-inf')                      # 当前 i 个任务的 U 函数
            cur_preset = []                                  # 任务 i 是否在处理器 j 上
            makespan = float('inf')                          # 当前 i 个任务的完成时间
            for j in range(M):
                presets[i][j] = 1                            # 任务 i 固定在处理器 j
                cur_sizes = workloads[: i + 1]               # 当前任务载荷
                z = M
                utility, cpus, jobs = solveNLP(
                    processSpeed, cur_sizes, cur_adj_matrix, presets, z)
                presets[i][j] = 0
                # print()
                if utility > cur_utility:
                    cur_utility = utility
                    cur_preset = [i, j]
                cur_max_makespan = float('-inf')
                # print('If Task {} runs on CPU {}:'.format(i + 1, j + 1))
                for _, cpu_task in cpus.items():
                    # print('CPU {}:'.format(_ + 1))
                    for t in cpu_task:
                        # print('T{}: {}'.format(t.id, t.duration['end']))
                        cur_max_makespan = max(
                            t.duration['end'], cur_max_makespan)
                    # print()
                makespan = min(makespan, cur_max_makespan)
            # print()
            # print('Task {} should be running on CPU {}'.format(
            #     cur_preset[0] + 1, cur_preset[1] + 1))
            presets[cur_preset[0]][cur_preset[1]] = 1
            # print()
            if i == num_tasks - 1:
                final_utility = cur_utility
                final_makespan = makespan
        print('Num of tasks = {}'.format(num_tasks))
        print('Utility = {}'.format(final_utility))
        print('Makespan = {}'.format(final_makespan))
        utility_dots.append(final_utility)
        makespan_dots.append(final_makespan)

    x = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 40]  # 点的横坐标

    plt.plot(x, makespan_dots, 's-', color='r', label="makespan")  # s-:方形
    plt.plot(x, utility_dots, 'o-', color='g', label="utility")   # o-:圆形
    plt.xlabel("num of tasks")  # 横坐标名字
    plt.ylabel("makespan & utility")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.show()


if __name__ == '__main__':
    # python solve_lp.py -i test.dot
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
                    help="DAG description as a .dot file")
    args = ap.parse_args()

    # solution()  # 跑 20、30、40 个任务作图

    # num_tasks 任务总数
    # workloads 任务载荷
    # adj_matrix 邻接矩阵
    num_tasks, workloads, adj_matrix = read_dag_adjacency(args.input)

    M, N = 4, num_tasks  # M = 处理器数量；N = 任务数量
    processSpeed = [[1 + i] * N for i in range(M)]  # 异构处理器速度
    # processSpeed = [[1] * N for i in range(M)]  # 同构处理器速度
    z = M
    presets = [[0] * M for _ in range(N)]
    final_utility = float('-inf')
    final_makespan = float('inf')

    for i in range(num_tasks):
        cur_adj_matrix = adj_matrix[:i + 1, :i + 1]      # 当前 i 个任务的邻接矩阵
        cur_utility = float('-inf')                      # 当前 i 个任务的 U 函数
        cur_preset = []                                  # 任务 i 是否在处理器 j 上
        makespan = float('inf')                          # 当前 i 个任务的完成时间
        for j in range(M):
            presets[i][j] = 1                            # 任务 i 固定在处理器 j
            cur_sizes = workloads[: i + 1]               # 当前任务载荷
            utility, cpus, jobs = solveNLP(
                processSpeed, cur_sizes, cur_adj_matrix, presets, z)
            presets[i][j] = 0
            # print()
            if utility > cur_utility:
                cur_utility = utility
                cur_preset = [i, j]
            cur_max_makespan = float('-inf')
            # print('If Task {} runs on CPU {}:'.format(i + 1, j + 1))
            for _, cpu_task in cpus.items():
                # print('CPU {}:'.format(_ + 1))
                for t in cpu_task:
                    # print('T{}: {}'.format(t.id, t.duration['end']))
                    cur_max_makespan = max(t.duration['end'], cur_max_makespan)
                # print()
            makespan = min(makespan, cur_max_makespan)
            # print('Makespan = {}'.format(cur_max_makespan))
            # print('Utility= {}'.format(utility))
        # print()
        print('Task {} should be running on CPU {}'.format(
            cur_preset[0] + 1, cur_preset[1] + 1))
        presets[cur_preset[0]][cur_preset[1]] = 1
        # print()
        if i == num_tasks - 1:
            final_utility = cur_utility
            final_makespan = makespan
    print()
    print('Utility = {}'.format(final_utility))
    print('Makespan = {}'.format(final_makespan))
