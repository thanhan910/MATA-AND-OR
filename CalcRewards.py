from GenAndOrTree import Node, NodeType


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


def sys_reward_agents(agents, tasks, allocation_structure, gamma=1):
    """
    Calculate the reward of the system, given the allocation structure: agent -> task
    """
    # allocation_structure is a vector of size M, each element indicate which task the agent is allocated to
    return sum(
        task_reward(task, [agent for i, agent in enumerate(agents) if allocation_structure[i] == j], gamma)
        for j, task in enumerate(tasks)
    )


def sys_rewards_tasks(tasks, agents, coalition_structure, gamma=1):
    """
    Calculate the reward of the system, given the coalition structure: task -> agents (coalition)
    """
    return sum(
        task_reward(task, [agents[i] for i in coalition_structure[j]], gamma)
        for j, task in enumerate(tasks)
    )


def sys_rewards_tree_agents(tree_info : list[Node], tasks, agents, allocation_structure, node_values : list = None, root_node_index=-1, gamma=1):
    """
    Calculate the reward of the system, given the allocation structure: agent -> task
    """
    def sys_rewards_node(node : Node):
        if node.node_type == NodeType.LEAF or node.node_type == NodeType.DUMMY:
            if node_values is not None and node_values != []:
                return node_values[node.node_id]
            return task_reward(tasks[node.node_id], [agent for i, agent in enumerate(agents) if allocation_structure[i] == node.node_id], gamma)
        rewards = [sys_rewards_node(tree_info[i]) for i in node.children_ids]
        if node.node_type == NodeType.AND:
            return sum(rewards)
        elif node.node_type == NodeType.OR:
            return max(rewards)
    return sys_rewards_node(tree_info[root_node_index])


def sys_rewards_tree_tasks(tree_info, root_node_type, tasks, agents, coalition_structure, node_values : list = None, root_node_index=-1, gamma=1):
    """
    Calculate the reward of the system, given the coalition structure: task -> agents (coalition)
    """
    def sys_rewards_node(node : Node):
        if node.node_type == NodeType.LEAF or node.node_type == NodeType.DUMMY:
            if node_values is not None and node_values != []:
                return node_values[node.node_id]
            return task_reward(tasks[node.node_id], [agents[i] for i in coalition_structure[node.node_id]], gamma)
        rewards = [sys_rewards_node(tree_info[i]) for i in node.children_ids]
        if node.node_type == NodeType.AND:
            return sum(rewards)
        elif node.node_type == NodeType.OR:
            return max(rewards)
    return sys_rewards_node(tree_info[root_node_index])

