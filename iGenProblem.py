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

def gen_constraints(agent_ids, task_ids, power=1, a_min_edge=2, t_max_edge=5):
    """
    Generate agent's constraints, each agent is represented by a list of tasks it has full capability to work on (so for each task for each agent, the agent has full capabilities that task requires).
    """
    # power is the inforce you put in the probabilities
    # the maximum tasks an agent could work on depends on the number of tasks available (e.g, if |T| = 1/2|A|, then roughly each agent can work on two tasks)

    # calculate the max and min edges for agents
    agent_num = len(agent_ids)
    task_num = len(task_ids)
    available_seats = math.floor(t_max_edge * task_num)
    a_taskIds = { i: [] for i in agent_ids }
    a_taskNums = { i: 0 for i in agent_ids }
    for i in agent_ids:
        a_max_edge = min((available_seats - a_min_edge * (agent_num - 1 - i)), t_max_edge, task_num)
        a_min_edge = min(a_min_edge, a_max_edge)
        
        # radomly indicate the number of task the agent could work on, based on the maximum and minimum number of tasks the agent could work on
        a_taskNums[i] = np.random.randint(a_min_edge, a_max_edge + 1)
        
        available_seats -= a_taskNums[i]

    t_agents_counts = { j: 0 for j in task_ids }  # each indicate the current number of agents on the task

    # make sure no further draw for those reached the maximum limit.
    t_indexes = [j for j in task_ids if t_agents_counts[j] < t_max_edge]

    for i, a_taskNum in a_taskNums.items():
        if any(tc == 0 for tc in t_agents_counts.values()):  # if there are tasks that have not been allocated to any agent
            t_prob = [
                (math.e ** (t_max_edge - t_agents_counts[j])) ** power
                for j in t_indexes
            ]  # power is used to manify the probability
            sum_prob = sum(t_prob)
            t_prop_2 = [prop / sum_prob for prop in t_prob]

            # draw tasks accounting to their current allocations
            a_taskIds[i] = list(
                np.random.choice(
                    a=t_indexes,
                    size=min(a_taskNum, len(t_indexes)),
                    replace=False,
                    p=[prop / sum_prob for prop in t_prob],
                )
            )
            # increase the chosen task counters
        else:
            a_taskIds[i] = list(
                np.random.choice(
                    a=t_indexes, size=min(a_taskNum, len(t_indexes)), replace=False
                )
            )

        for j in a_taskIds[i]:
            t_agents_counts[j] += 1

        # make sure no further draw for those reached the maximum limit.
        t_indexes = [
            j for j in task_ids if t_agents_counts[j] < t_max_edge
        ]

    # get also the list of agents for each task
    t_agentIds = [
        [i for i in agent_ids if j in a_taskIds[i]]
        for j in task_ids
    ]

    return a_taskIds, t_agentIds