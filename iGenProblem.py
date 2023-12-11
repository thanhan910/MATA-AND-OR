import itertools
import math
import numpy as np


def gen_tasks(taskIds, max_capNum, capIds):
    """
    Generate tasks, each task is represented by a list of capabilities it requires
    """
    # n is the number of task, max_capNum is the maximum number of cap a task could require
    return {
        j: sorted(
            np.random.choice(
                a=capIds, size=np.random.randint(3, max_capNum + 1), replace=False
            )
        )
        for j in taskIds
    }

def gen_constraints(agentIds, taskIds, power=1, a_min_edge=2, t_max_edge=5):
    """
    Generate agent's constraints, each agent is represented by a list of tasks it has full capability to work on (so for each task for each agent, the agent has full capabilities that task requires).
    """
    # power is the inforce you put in the probabilities
    # the maximum tasks an agent could work on depends on the number of tasks available (e.g, if |T| = 1/2|A|, then roughly each agent can work on two tasks)

    # calculate the max and min edges for agents
    agent_num = len(agentIds)
    task_num = len(taskIds)
    available_seats = math.floor(t_max_edge * task_num)
    agents_taskIds = { i: [] for i in agentIds }
    agents_taskNums = { i: 0 for i in agentIds }
    for i in agentIds:
        a_max_edge = min((available_seats - a_min_edge * (agent_num - 1 - i)), t_max_edge, task_num)
        a_min_edge = min(a_min_edge, a_max_edge)
        
        # radomly indicate the number of task the agent could work on, based on the maximum and minimum number of tasks the agent could work on
        agents_taskNums[i] = np.random.randint(a_min_edge, a_max_edge + 1)
        
        available_seats -= agents_taskNums[i]

    tasks_agentsCounts = { j: 0 for j in taskIds }  # each indicate the current number of agents on the task

    # make sure no further draw for those reached the maximum limit.
    taskIndexes = [j for j in taskIds if tasks_agentsCounts[j] < t_max_edge]

    for i, a_taskNum in agents_taskNums.items():
        if any(tc == 0 for tc in tasks_agentsCounts.values()):  # if there are tasks that have not been allocated to any agent
            t_prob = [
                (math.e ** (t_max_edge - tasks_agentsCounts[j])) ** power
                for j in taskIndexes
            ]  # power is used to manify the probability
            sum_prob = sum(t_prob)
            t_prop_2 = [prop / sum_prob for prop in t_prob]

            # draw tasks accounting to their current allocations
            agents_taskIds[i] = list(
                np.random.choice(
                    a=taskIndexes,
                    size=min(a_taskNum, len(taskIndexes)),
                    replace=False,
                    p=[prop / sum_prob for prop in t_prob],
                )
            )
            # increase the chosen task counters
        else:
            agents_taskIds[i] = list(
                np.random.choice(
                    a=taskIndexes, size=min(a_taskNum, len(taskIndexes)), replace=False
                )
            )

        for j in agents_taskIds[i]:
            tasks_agentsCounts[j] += 1

        # make sure no further draw for those reached the maximum limit.
        taskIndexes = [j for j in taskIds if tasks_agentsCounts[j] < t_max_edge]

    # get also the list of agents for each task
    tasks_agentIds = [
        [i for i in agentIds if j in agents_taskIds[i]]
        for j in taskIds
    ]

    return agents_taskIds, tasks_agentIds


def gen_agents(agents_taskIds, tasks_capIds, max_capNum, capIds, max_capVal):  
    # m is the number of task, max_capNum is the maximum number of cap a task could require, max_capVal is the maximum capability value
    """
    Generate agents, each agent is represented by a list of capabilities it has and a list of contribution values for each capability.
    
    Generate based on the tasks that the agent could fully perform.
    """
    agents_capIds = {}
    agents_capContributions = {}
    for i, a_taskId in agents_taskIds.items():
        a_t_caps_list = [tasks_capIds[j] for j in a_taskId]  # lists of caps that each task agent could perform

        a_caps_union = set(itertools.chain(*a_t_caps_list))  # union of unique caps of tasks that agent could perform.

        a_cap_num = np.random.randint(min(3, max_capNum, len(a_caps_union)), min(len(a_caps_union), max_capNum) + 1)  # the num of caps the agent will have

        a_caps = set([np.random.choice(t_c) for t_c in a_t_caps_list])  # initial draw to guarantee the agent has some contribution to each of the task that the agent has the capability to perform.

        # Randomly draw the remaining capabilities, possibly none
        remaining_choices = list(a_caps_union.difference(a_caps))
        if remaining_choices != []:
            a_caps.update(
                np.random.choice(
                    remaining_choices,
                    min(max(0, a_cap_num - len(a_taskId)), len(remaining_choices)),
                    replace=False,
                )
            )
        
        # a_caps.update(np.random.choice(remaining_choices, min(0,len(remaining_choices),a_cap_num-len(a_taskInd)),replace = False))

        a_caps_list = sorted(list(a_caps))
        a_contri = {
            c : (np.random.randint(1, max_capVal + 1) if c in a_caps_list else 0)
            for c in capIds
        }

        agents_capIds[i] = a_caps_list
        agents_capContributions[i] = a_contri

    return agents_capIds, agents_capContributions



def gen_agents_random(agentIds, capIds, max_capNum, max_capVal):
    """
    Generate agents, each agent is represented by a list of capabilities it has and its contribution values for each capability.
    """
    agents_capIds = {}
    agents_capContributions = {}
    for i in agentIds:
        a_cap_num = np.random.randint(1, max_capNum + 1)  # the num of caps the agent will have
        a_caps = set(np.random.choice(capIds, a_cap_num, replace=False))

        agents_capIds[i] = sorted(list(a_caps))
        agents_capContributions[i] = {
            c: (np.random.randint(1, max_capVal + 1) if c in agents_capIds[i] else 0)
            for c in capIds
        }

    return agents_capIds, agents_capContributions


def calc_constraints(agents_capIds, task_caps):
    """
    Calculate the constraints of the system, where the system consists of tasks and agents with constraints.
    """
    agents_taskIds = { i: [] for i in agents_capIds }
    tasks_agentIds = { j: [] for j in task_caps }
    for i, a_caps in agents_capIds.items():
        for j, t_caps in task_caps.items():
            if set(t_caps).issubset(a_caps):
                agents_taskIds[i].append(j)
                tasks_agentIds[j].append(i)
    return agents_taskIds, tasks_agentIds
