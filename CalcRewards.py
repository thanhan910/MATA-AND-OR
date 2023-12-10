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


def sys_rewards_tree_agents(tree, root_node_type, tasks, agents, allocation_structure, gamma=1):
    """
    Calculate the reward of the system, given the allocation structure: agent -> task
    """
    if isinstance(tree, int):
        return task_reward(tasks[tree], [agent for i, agent in enumerate(agents) if allocation_structure[i] == tree], gamma)
    rewards = [
        sys_rewards_tree_agents(subtree, root_node_type, tasks, agents, allocation_structure, gamma)
        for subtree in tree
    ]
    if root_node_type == "AND":
        return sum(rewards)
    elif root_node_type == "OR":
        return max(rewards)


def sys_rewards_tree_tasks(tree, root_node_type, tasks, agents, coalition_structure, gamma=1):
    """
    Calculate the reward of the system, given the coalition structure: task -> agents (coalition)
    """
    if isinstance(tree, int):
        return task_reward(tasks[tree], [agents[i] for i in coalition_structure[tree]], gamma)
    rewards = [
        sys_rewards_tree_tasks(subtree, root_node_type, tasks, agents, coalition_structure, gamma)
        for subtree in tree
    ]
    if root_node_type == "AND":
        return sum(rewards)
    elif root_node_type == "OR":
        return max(rewards)

