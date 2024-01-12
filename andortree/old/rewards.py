from .tree_utils_old import Node, NodeType

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

def sys_rewards_tree_agents(tree_info : list[Node], tasks, agents, allocation_structure, root_node_index=-1, gamma=1):
    """
    Calculate the reward of the system, given the allocation structure: agent -> task
    """
    def sys_rewards_node(node : Node):
        if node.node_type == NodeType.LEAF or node.node_type == NodeType.DUMMY:
            return task_reward(tasks[node.node_id], [agent for i, agent in enumerate(agents) if allocation_structure[i] == node.node_id], gamma)
        rewards = [sys_rewards_node(tree_info[i]) for i in node.children_ids]
        if node.node_type == NodeType.AND:
            return sum(rewards)
        elif node.node_type == NodeType.OR:
            return max(rewards)
    return sys_rewards_node(tree_info[root_node_index])


def sys_rewards_tree_tasks(tree_info, tasks, agents, coalition_structure, root_node_index=-1, gamma=1):
    """
    Calculate the reward of the system, given the coalition structure: task -> agents (coalition)
    """
    def sys_rewards_node(node : Node):
        if node.node_type == NodeType.LEAF or node.node_type == NodeType.DUMMY:
            return task_reward(tasks[node.node_id], [agents[i] for i in coalition_structure[node.node_id]], gamma)
        rewards = [sys_rewards_node(tree_info[i]) for i in node.children_ids]
        if node.node_type == NodeType.AND:
            return sum(rewards)
        elif node.node_type == NodeType.OR:
            return max(rewards)
    return sys_rewards_node(tree_info[root_node_index])
