# Microservice Scheduler

DAG generator (./generator):

Constructing new example DAGs requires the [DAGGEN](https://github.com/frs69wq/daggen) github repository.

```bash
./daggen -n 20 --fat 0.4 --density 0.2 --regular 0.2 --jump 2 --minalpha 20 --maxalpha 200 --dot -o ../test.dot
```

> fat 越低 Algo 2 的 makespan 越小，和 HEFT 差距不大 （7 - 10%）；反之差距很大
>
> density 影响不大
>
> --minalpha 20 --maxalpha 200 任务载荷范围

## Solver

Algorithm2 (gurobi linear programming):

```bash
python algorithm2.py -i test.dot
```

HEFT:

```bash
python heft.py -i test.dot
```
