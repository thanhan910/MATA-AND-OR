import itertools
import math
import numpy as np

def gen_tasks(task_num, max_capNum, capabilities):
    """
    Generate tasks, each task is represented by a list of capabilities it requires
    :param: `task_num`: the number of tasks
    :param: `max_capNum`: the maximum number of capabilities a task could require
    :param: `capabilities`: the list of capabilities
    :return: the list of tasks. Each task is represented by a list of capabilities it requires.
    """
    # n is the number of task, max_capNum is the maximum number of cap a task could require
    return [
        sorted(
            np.random.choice(
                a=capabilities, size=np.random.randint(3, max_capNum + 1), replace=False
            )
        )
        for j in range(0, task_num)
    ]


def gen_constraints(agent_num, task_num, power=1, a_min_edge=2, t_max_edge=5):
    """
    Generate agent's constraints, each agent is represented by a list of tasks it has capability to work on.
    :param: `agent_num`: the number of agents
    :param: `task_num`: the number of tasks
    :param: `power`: the power used to magnify the probability
    :param: `a_min_edge`: the minimum number of tasks an agent has the capabilities work on.
    :param: `t_max_edge`: the maximum number of agents that could work on a task.
    :return: For each agent, the list of tasks it has the capabilities to work on. For each task, the list of agents that could work on it.
    """

    # power is the inforce you put in the probabilities
    # the maximum tasks an agent could work on depends on the number of tasks available (e.g, if |T| = 1/2|A|, then roughly each agent can work on two tasks)

    # calculate the max and min edges for agents
    available_seats = math.floor(t_max_edge * task_num)
    a_taskInds = [[] for i in range(0, agent_num)]
    a_taskNums = []
    for i in range(0, agent_num):
        a_max_edge = min((available_seats - a_min_edge * (agent_num - 1 - i)), t_max_edge, task_num)
        a_min_edge = min(a_min_edge, a_max_edge)
        
        # radomly indicate the number of task the agent could work on, based on the maximum and minimum number of tasks the agent could work on
        a_taskNum = np.random.randint(a_min_edge, a_max_edge + 1)
        
        a_taskNums.append(a_taskNum)
        
        available_seats -= a_taskNum

    t_agents_counts = [0 for j in range(0, task_num)]  # each indicate the current number of agents on the task

    # make sure no further draw for those reached the maximum limit.
    t_indexes = [j for j in range(0, task_num) if t_agents_counts[j] < t_max_edge]

    for i, a_taskNum in enumerate(a_taskNums):
        if any(tc == 0 for tc in t_agents_counts):  # if there are tasks that have not been allocated to any agent
            t_prob = [
                (math.e ** (t_max_edge - t_agents_counts[j])) ** power
                for j in t_indexes
            ]  # power is used to manify the probability
            sum_prob = sum(t_prob)
            t_prop_2 = [prop / sum_prob for prop in t_prob]

            # draw tasks accounting to their current allocations
            a_taskInds[i] = list(
                np.random.choice(
                    a=t_indexes,
                    size=min(a_taskNum, len(t_indexes)),
                    replace=False,
                    p=[prop / sum_prob for prop in t_prob],
                )
            )
            # increase the chosen task counters
        else:
            a_taskInds[i] = list(
                np.random.choice(
                    a=t_indexes, size=min(a_taskNum, len(t_indexes)), replace=False
                )
            )

        for j in a_taskInds[i]:
            t_agents_counts[j] += 1

        # make sure no further draw for those reached the maximum limit.
        t_indexes = [
            j for j in range(0, task_num) if t_agents_counts[j] < t_max_edge
        ]

    # get also the list of agents for each task
    t_agents = [
        [i for i in range(0, agent_num) if j in a_taskInds[i]]
        for j in range(0, task_num)
    ]

    return a_taskInds, t_agents


def gen_agents(a_taskInds, tasks, max_capNum, capabilities, max_capVal):  
    # m is the number of task, max_capNum is the maximum number of cap a task could require, max_capVal is the maximum capability value
    """
    Generate agents, each agent is represented by a list of capabilities it has and a list of contribution values for each capability
    :param: `a_taskInds`: the list of list of tasks each agent could work on
    :param: `tasks`: the list of tasks, represented by a list of capabilities it requires
    :param: `max_capNum`: the maximum number of capabilities an agent could have
    :param: `capabilities`: the list of capabilities
    :param: `max_capVal`: the maximum value of a capability
    """
    caps_lists = []
    contri_lists = []
    for a_taskInd in a_taskInds:
        t_caps = [tasks[j] for j in a_taskInd]  # lists of caps that each task agent could perform

        caps_union = set(itertools.chain(*t_caps))  # union of unique caps of tasks that agent could perform.

        a_cap_num = np.random.randint(
            min(3, max_capNum, len(caps_union)), 
            min(len(caps_union), max_capNum) + 1
        )  # the num of caps the agent will have

        a_caps = set([np.random.choice(t_c) for t_c in t_caps])  # initial draw to guarantee the agent has some contribution to each of the task that the agent has the capability to perform.

        # Randomly draw the remaining capabilities, possibly none
        remaining_choices = list(caps_union.difference(a_caps))
        if remaining_choices != []:
            a_caps.update(
                np.random.choice(
                    remaining_choices,
                    min(max(0, a_cap_num - len(a_taskInd)), len(remaining_choices)),
                    replace=False,
                )
            )
        
        # a_caps.update(np.random.choice(remaining_choices, min(0,len(remaining_choices),a_cap_num-len(a_taskInd)),replace = False))

        caps_list = sorted(list(a_caps))
        contri_list = [
            (np.random.randint(1, max_capVal + 1) if c in caps_list else 0)
            for c in range(0, len(capabilities))
        ]

        caps_lists.append(caps_list)
        contri_lists.append(contri_list)

    return caps_lists, contri_lists


def gen_agents_random(capabilities, agent_num, max_capNum, max_capVal):
    """
    Generate agents, each agent is represented by a list of capabilities it has and a list of contribution values for each capability
    :param: `capabilities`: the list of capabilities
    :param: `agent_num`: the number of agents
    :param: `max_capNum`: the maximum number of capabilities an agent could have
    :param: `max_capVal`: the maximum value of a capability
    """
    caps_lists = []
    contri_lists = []
    for i in range(0, agent_num):
        a_cap_num = np.random.randint(1, max_capNum + 1)  # the num of caps the agent will have
        a_caps = set(np.random.choice(capabilities, a_cap_num, replace=False))
        caps_list = sorted(list(a_caps))
        contri_list = [
            (np.random.randint(1, max_capVal + 1) if c in caps_list else 0)
            for c in range(0, len(capabilities))
        ]

        caps_lists.append(caps_list)
        contri_lists.append(contri_list)

    return caps_lists, contri_lists



def get_constraints(agents, tasks):
    """
    Calculate the constraints of the system, where the system consists of tasks and agents with constraints.

    :param: `agents`: the list of agents
    :param: `tasks`: the list of tasks
    :return: For each agent, the list of tasks it has the capabilities to work on. For each task, the list of agents that could work on it.
    """
    a_taskInds = []
    t_agentInds = []
    for agent in agents:
        a_taskInds.append([j for j, task in enumerate(tasks) if set(task) <= set(agent)])
    for task in tasks:
        t_agentInds.append([i for i, agent in enumerate(agents) if set(task) <= set(agent)])
    return a_taskInds, t_agentInds
