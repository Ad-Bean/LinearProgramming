# python heft.py -i test.dot

import operator
from Processor import Processor
from read_dag import read_dag
from Task import Task


class HEFT:
    def __init__(self, input_list=None, file=None, verbose=False, p=3, b=0.5, ccr=0.5):
        """ 
        @param file: 输入文件, 由 DAGGEN 生成
        @param verbose: boolean, 输出调试信息
        @param p: processor, 处理器个数
        @param b: 
        @param ccr: 
        """
        if input_list is None and file is not None:
            self.num_tasks, self.num_processors, comp_cost, self.graph = read_dag(
                file, p, b, ccr)
        elif len(input_list) == 4 and file is None:
            self.num_tasks, self.num_processors, comp_cost, self.graph = input_list
        else:
            print('Enter filename or input params')
            raise Exception()

        if verbose:
            print("No. of Tasks: ", self.num_tasks)
            print("No. of processors: ", self.num_processors)
            # print("Computational Cost Matrix:")
            # for i in range(self.num_tasks):
            #     print(comp_cost[i])
            # print("Graph Matrix:")
            # for line in self.graph:
            #     print(line)

        self.tasks = [Task(i) for i in range(self.num_tasks)]
        self.processors = [Processor(i) for i in range(self.num_processors)]

        # HEFT: compute cost and rank
        for i in range(self.num_tasks):
            self.tasks[i].comp_cost = comp_cost[i]
            self.tasks[i].avg_comp = sum(comp_cost[i]) / self.num_processors

        self.__computeRanks(self.tasks[0])

        if verbose:
            for task in self.tasks:
                print("Task ", task.id, "-> Rank: ", task.rank)
        self.tasks.sort(key=lambda x: x.rank, reverse=True)

        self.__allotProcessor()
        self.makespan = max([t.duration['end'] for t in self.tasks])

    def __computeRanks(self, task):
        # Assume that task[0] is the initial task, as generated by TGFF
        # Assume communicate rate is equal between processors
        curr_rank = 0
        for succ in self.tasks:
            if self.graph[task.id][succ.id] != -1:
                if succ.rank is None:
                    self.__computeRanks(succ)
                curr_rank = max(
                    curr_rank, self.graph[task.id][succ.id] + succ.rank)
        task.rank = task.avg_comp + curr_rank

    def __get_est(self, t, p):
        est = 0
        for pre in self.tasks:
            # if pre also done on p, no communication cost
            if self.graph[pre.id][t.id] != -1:
                c = self.graph[pre.id][t.id] if pre.processor_id != p.id else 0
                est = max(est, pre.duration['end'] + c)
        free_times = []
        if len(p.task_list) == 0:       # no task has yet been assigned to processor
            free_times.append([0, float('inf')])
        else:
            for i in range(len(p.task_list)):
                if i == 0:
                    # if p is not busy from time 0
                    if p.task_list[i].duration['start'] != 0:
                        free_times.append(
                            [0, p.task_list[i].duration['start']])
                else:
                    free_times.append(
                        [p.task_list[i-1].duration['end'], p.task_list[i].duration['start']])
            free_times.append([p.task_list[-1].duration['end'], float('inf')])
        for slot in free_times:     # free_times is already sorted based on avaialbe start times
            if est < slot[0] and slot[0] + t.comp_cost[p.id] <= slot[1]:
                return slot[0]
            if est >= slot[0] and est + t.comp_cost[p.id] <= slot[1]:
                return est

    def __allotProcessor(self):
        for t in self.tasks:
            if t == self.tasks[0]:   # the one with highest rank
                p, w = min(enumerate(t.comp_cost), key=operator.itemgetter(1))
                t.processor_id = p
                t.duration['start'] = 0
                t.duration['end'] = w
                self.processors[p].task_list.append(t)
            else:
                aft = float("inf")
                for p in self.processors:
                    est = self.__get_est(t, p)
                    # print("Task: ", t.id, ", Proc: ", p.id, " -> EST: ", est)
                    eft = est + t.comp_cost[p.id]
                    if eft < aft:   # found better case of processor
                        aft = eft
                        best_p = p.id

                t.processor_id = best_p
                t.duration['start'] = aft - t.comp_cost[best_p]
                t.duration['end'] = aft
                self.processors[best_p].task_list.append(t)
                self.processors[best_p].task_list.sort(
                    key=lambda x: x.duration['start'])

    def __str__(self):
        print_str = ""
        for p in self.processors:
            print_str += 'Processor {}:\n '.format(p.id)
            for t in p.task_list:
                print_str += 'Task {}: start = {}, end = {}\n'.format(
                    t.id, t.duration['start'], t.duration['end'])
        print_str += "Makespan = {}\n".format(self.makespan)
        return print_str


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()

    ap.add_argument('-i', '--input', required=True,
                    help="DAG description as a .dot file")
    args = ap.parse_args()
    new_sch = HEFT(file=args.input, verbose=True, p=4, b=0.1, ccr=0.1)
    print(new_sch)
