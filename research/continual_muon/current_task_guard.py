"""Enforce the active task at the data-loader boundary of the diagnostic."""


class CurrentTaskGuard:
    def __init__(self, loader, raw, task_indices):
        self.loader, self.raw = loader, raw
        self.allowed = {task: frozenset(indices) for task, indices in task_indices.items()}
        if set(self.allowed) != {1, 2} or self.allowed[1] & self.allowed[2]:
            raise ValueError('expected disjoint T2 and T3 train partitions')
        self.task = 1
        self.visits = {1: 0, 2: 0}

    def __call__(self, raw, indices, batch, shuffle=False):
        if raw is not self.raw or not set(indices) <= self.allowed[self.task]:
            raise ValueError('only the current task training partition is permitted')
        source = self.loader(raw, indices, batch, shuffle)
        owner = self
        task = self.task

        class Batches:
            def __len__(self):
                return len(source)

            def __iter__(self):
                if owner.task != task:
                    raise ValueError('loader used after task switch')
                for x, y in source:
                    owner.visits[task] += len(y)
                    yield x, y
        return Batches()

    def enter_t3(self):
        if self.task != 1:
            raise ValueError('invalid or repeated task switch')
        self.task = 2
