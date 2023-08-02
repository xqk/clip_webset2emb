"""distributors provide way to compute using several gpus and several machines"""

from .worker import worker


class SequentialDistributor:
    def __init__(self, tasks, worker_args):
        self.tasks = tasks
        self.worker_args = worker_args

    def __call__(self):
        """
        call a single `worker(...)` and pass it everything.
        """
        worker(
            tasks=self.tasks,
            **self.worker_args,
        )
