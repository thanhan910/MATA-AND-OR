from .rewards import sys_rewards_tree_agents
from .node_type import NodeType

import numpy as np

def random_solution_and_or_tree(
        node_type_info : dict[int, NodeType],
        children_info : dict[int, list[int]],
        leaf2task : dict[int, int], 
        tasks : list[list[int]], 
        agents : list[list[int]],
        constraints, 
        gamma=1
    ):
    '''
    Randomly allocate tasks to agents
    '''
    dummy_task_id = len(tasks)
    agent_num = len(agents)
    a_taskInds = constraints[0]
    allocation_structure = [np.random.choice(a_taskInds[i] + [dummy_task_id]) for i in range(0, agent_num)]
    return allocation_structure, sys_rewards_tree_agents(node_type_info, children_info, leaf2task, tasks, agents, allocation_structure, gamma=gamma)