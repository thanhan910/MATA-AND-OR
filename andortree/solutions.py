from .rewards import sys_rewards_tree_agents
from .tree_utils import Node

import numpy as np

def random_solution_and_or_tree(agents, tasks, constraints, tree_info: list[Node], gamma=1):
    '''
    Randomly allocate tasks to agents
    '''
    task_num = len(tasks)
    agent_num = len(agents)
    a_taskInds = constraints[0]
    allocation_structure = [np.random.choice(a_taskInds[i] + [task_num]) for i in range(0, agent_num)]
    return allocation_structure, sys_rewards_tree_agents(tree_info, tasks, agents, allocation_structure, gamma=gamma)