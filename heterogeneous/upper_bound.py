import itertools
import numpy as np

def upperBound(capabilities, tasks, agents):
    """
    Calculate the upper bound of the system reward, where the system consists of tasks and agents with constraints.

    This mathematical upper bound is calculated by sorting the agents based on their contribution values for each capability, in descending order, then count `m`, the number of tasks that require each capability, and sum up the contribution values of the top `m` agents for each capability.
    
    :param: `capabilities`: the list of capabilities
    :param: `tasks`: the list of tasks
    :param: `agents`: the list of agents
    :return: the upper bound of the system reward
    """
    cap_ranked = [sorted([a[c] for a in agents], reverse=True) for c in capabilities] # Time complexity: O(len(capabilities) * log(len(capabilities)) * len(agents))
    cap_req_all = list(itertools.chain(*tasks)) # Time complexity: O(size of tasks capabilities combined), around O(len(tasks) * len(capabilities))
    cap_req_num = [cap_req_all.count(c) for c in capabilities] # Time complexity: O(len(cap_req_all) * len(capabilities)). However, can be optimized to O(len(cap_req_all)).
    return sum([sum(cap_ranked[c][:cap_req_num[c]]) for c in capabilities]) # Time complexity: O(len(cap_req_all))
    # Evaluated time complexity: max(O(len(capabilities) * log(len(capabilities)) * len(agents)), O(len(tasks) * len(capabilities)))


def upperBound_ver2(capabilities, tasks, agents, constraints):
    """
    Calculate the upper bound of the system reward, where the system consists of tasks and agents with constraints.

    This upper bound is calculated by sorting the agents based on their contribution values for each capability, in descending order, then iteratively allocate the top agents to the tasks that require that capability.

    This allows for a more precise upper bound than upperBound, since it takes into account the `constraints`: the top agents might only be able to work on the same limited tasks.

    :param: `capabilities`: the list of capabilities
    :param: `tasks`: the list of tasks
    :param: `agents`: the list of agents
    :param: `constraints`: the list of constraints
    :return: the upper bound of the system reward
    """
    agent_num = len(agents)
    task_num = len(tasks)
    a_taskInds = constraints[0]
    cap_req_all = list(itertools.chain(*tasks))
    cap_req_num = [cap_req_all.count(c) for c in capabilities]

    sys_rewards = 0
    for c in capabilities:
        
        a_cap_vals = [agent[c] for agent in agents]

        # the list of tasks that each agent has the capability to perform and that require the capability c
        a_cap_tasks = [[j for j in a_taskInd if j != task_num and c in tasks[j]] for a_taskInd in a_taskInds] 

        # sort the agents based on their contribution values for the capability c, in descending order
        cap_rank_pos = np.argsort(a_cap_vals)[::-1]

        a_cap_vals_ordered = [0 for _ in range(0, agent_num)]
        a_cap_tasks_ordered = [[] for _ in range(0, agent_num)]
        for p, pos in enumerate(cap_rank_pos):
            a_cap_vals_ordered[p] = a_cap_vals[pos]
            a_cap_tasks_ordered[p] = a_cap_tasks[pos]

        cap_rewards = a_cap_vals_ordered[0]
        cap_tasks = set(a_cap_tasks_ordered[0])
        a_cap_num = 1
        for a_iter in range(1, agent_num):
            cap_tasks = cap_tasks.union(set(a_cap_tasks_ordered[a_iter]))
            if len(cap_tasks) > a_cap_num:
                cap_rewards += a_cap_vals_ordered[a_iter]
                a_cap_num += 1
            # break if they got enough agents to contribute the number of required cap c
            if (a_cap_num >= cap_req_num[c]):  
                break
        sys_rewards += cap_rewards
    return sys_rewards

