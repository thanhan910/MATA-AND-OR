import itertools
import math
import numpy as np


def gen_tasks(task_ids, max_capNum, cap_ids):
    """
    Generate tasks, each task is represented by a list of capabilities it requires
    """
    # n is the number of task, max_capNum is the maximum number of cap a task could require
    return {
        j: sorted(
            np.random.choice(
                a=cap_ids, size=np.random.randint(3, max_capNum + 1), replace=False
            )
        )
        for j in task_ids
    }
