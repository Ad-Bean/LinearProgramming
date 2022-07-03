class Task:
    def __init__(self, id):
        self.id = id
        self.processor_id = None
        self.rank = None
        self.comp_cost = []
        self.weight = None
        self.duration = {'start': None, 'end': None}
        self.avg_comp = None
        self.CNP = False
