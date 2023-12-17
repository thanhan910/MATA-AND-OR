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



