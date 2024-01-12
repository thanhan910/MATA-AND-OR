from .node_type import NodeType

def task_reward(task, agents, gamma=1):
    # task is represented by a list of capabilities it requires, agents is a list agents, where each represented by a list cap contribution values
    """
    Calculate the reward of a single task
    :param: `task`: the list of capabilities the task requires
    :param: `agents`: the list of agents
    :param: `gamma`: the discount factor
    :return: the reward of the task
    """
    if agents == []:
        return 0
    else:
        return sum([max([agent[c] for agent in agents]) for c in task]) * (
            gamma ** len(agents)
        )

def sys_rewards_tree_agents(
        node_type_info : dict[int, NodeType],
        children_info : dict[int, list[int]],
        leaf2task : dict[int, int], 
        tasks : list[list[int]], 
        agents : list[list[int]], 
        allocation_structure : list[int], 
        root_node_id=0, 
        gamma=1
    ):
    """
    Calculate the reward of the system, given the allocation structure: agent -> task
    """
    def sys_rewards_node(node_id : int):
        
        node_type = node_type_info[node_id]
        
        if node_type == NodeType.LEAF or node_type == NodeType.DUMMY:
            return task_reward(tasks[leaf2task[node_id]], [agent for i, agent in enumerate(agents) if allocation_structure[i] == node_id], gamma)
        
        child_rewards = [sys_rewards_node(child_id) for child_id in children_info[node_id]]
        
        if node_type == NodeType.AND:
            return sum(child_rewards)
        elif node_type == NodeType.OR:
            return max(child_rewards)
        
    return sys_rewards_node(root_node_id)


def sys_rewards_tree_tasks(
        node_type_info : dict[int, NodeType],
        children_info : dict[int, list[int]],
        leaf2task : dict[int, int], 
        tasks : list[list[int]], 
        agents : list[list[int]], 
        coalition_structure : list[list[int]], 
        root_node_id=0, 
        gamma=1
    ):
    """
    Calculate the reward of the system, given the coalition structure: task -> agents (coalition)
    """
    def sys_rewards_node(node_id : int):
        
        node_type = node_type_info[node_id]
        
        if node_type == NodeType.LEAF or node_type == NodeType.DUMMY:
            task_id = leaf2task[node_id]
            return task_reward(tasks[task_id], [agents[i] for i in coalition_structure[task_id]], gamma)
        
        child_rewards = [sys_rewards_node(child_id) for child_id in children_info[node_id]]
        
        if node_type == NodeType.AND:
            return sum(child_rewards)
        elif node_type == NodeType.OR:
            return max(child_rewards)
        
    return sys_rewards_node(root_node_id)
