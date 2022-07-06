from Task import Task

jobs = [Task(j) for j in range(10)]

for i in range(10):
    print(jobs[i].id)
