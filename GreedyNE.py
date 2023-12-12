import numpy as np

from CalcRewards import *


def agent_contribution(agents, tasks, query_agentIndex, query_taskIndex, coalition, constraints, gamma=1):
    """
    Return contribution of agent i to task j in coalition C_j
    
    = U_i(C_j, j) - U_i(C_j \ {i}, j) if i in C_j

    = U_i(C_j U {i}, j) - U_i(S, j) if i not in C_j
    """
    a_taskInds = constraints[0]
    if query_taskIndex == len(tasks):
        return 0
    if query_taskIndex not in a_taskInds[query_agentIndex]:
        return 0
    cur_reward = task_reward(tasks[query_taskIndex], [agents[i] for i in coalition], gamma)
    if query_agentIndex in coalition:
        return cur_reward - task_reward(tasks[query_taskIndex], [agents[i] for i in coalition if i != query_agentIndex], gamma)
    else:
        return task_reward(tasks[query_taskIndex], [agents[i] for i in coalition] + [agents[query_agentIndex]], gamma) - cur_reward


def eGreedy2(agents, tasks, constraints, eps=0, gamma=1, coalition_structure=[]):
    re_assign = 0
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)
    allocation_structure = [task_num] * agent_num  # each indicate the current task that agent i is allocated to, if = N, means not allocated
    if coalition_structure == []:
        coalition_structure = [[]] * (task_num + 1)  # current coalition structure, the last one is dummy coalition
        cur_con = [0] * agent_num
    else:
        coalition_structure.append([])
        for j in range(0, task_num):
            for i in coalition_structure[j]:
                allocation_structure[i] = j
        cur_con = [
            agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            for i, j in enumerate(allocation_structure)
        ]

    task_cons = [
        [
            agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            if j in a_taskInds[i]
            else -1000
            for j in range(0, task_num)
        ] + [0]
        for i in range(0, agent_num)
    ]
    # the last 0 indicate not allocated

    move_vals = [
        [
            task_cons[i][j] - cur_con[i] if j in a_taskInds[i] + [task_num] else -1000
            for j in range(0, task_num + 1)
        ]
        for i in range(0, agent_num)
    ]

    max_moveIndexs = [
        np.argmax([move_vals[i][j] for j in a_taskInds[i]] + [0])
        for i in range(0, agent_num)
    ]

    max_moveVals = [
        move_vals[i][a_taskInds[i][max_moveIndexs[i]]]
        if max_moveIndexs[i] < len(a_taskInds[i])
        else move_vals[i][task_num]
        for i in range(0, agent_num)
    ]

    iteration = 0
    while True:
        iteration += 1
        feasible_choices = [i for i in range(0, agent_num) if max_moveVals[i] > 0]
        if feasible_choices == []:
            break  # reach NE solution
        # when eps = 1, it's Random, when eps = 0, it's Greedy
        if np.random.uniform() <= eps:
            # exploration: random allocation
            a_index = np.random.choice(feasible_choices)
        else:
            # exploitation: allocationelse based on reputation or efficiency
            a_index = np.argmax(max_moveVals)
            
        t_index = a_taskInds[a_index][max_moveIndexs[a_index]] if max_moveIndexs[a_index] < len(a_taskInds[a_index]) else task_num

        # perfom move
        old_t_index = allocation_structure[a_index]
        allocation_structure[a_index] = t_index
        coalition_structure[t_index].append(a_index)

        # update agents in the new coalition
        affected_a_indexes = []
        affected_t_indexes = []
        if t_index != task_num:
            affected_a_indexes.extend(coalition_structure[t_index])
            affected_t_indexes.append(t_index)

            # task_cons[i][t_index]
            for i in coalition_structure[t_index]:
                task_cons[i][t_index] = agent_contribution(agents, tasks, i, t_index, coalition_structure[t_index], constraints, gamma)
                cur_con[i] = task_cons[i][t_index]
        else:
            affected_a_indexes.append(a_index)
            task_cons[a_index][t_index] = 0
            cur_con[a_index] = 0

        # update agent in the old coalition (if applicable)
        if (old_t_index != task_num):  
            # if agents indeed moved from another task, we have to change every agent from the old as well
            re_assign += 1
            coalition_structure[old_t_index].remove(a_index)
            affected_a_indexes.extend(coalition_structure[old_t_index])
            affected_t_indexes.append(old_t_index)
            for i in coalition_structure[old_t_index]:
                task_cons[i][old_t_index] = agent_contribution(agents, tasks, i, old_t_index, coalition_structure[old_t_index], constraints, gamma)
                cur_con[i] = task_cons[i][old_t_index]

        for i in affected_a_indexes:
            move_vals[i] = [
                task_cons[i][j] - cur_con[i]
                if j in a_taskInds[i] + [task_num]
                else -1000
                for j in range(0, task_num + 1)
            ]

        ## update other agents w.r.t the affected tasks
        for t_ind in affected_t_indexes:
            for i in range(0, agent_num):
                if (i not in coalition_structure[t_ind]) and (t_ind in a_taskInds[i]):
                    task_cons[i][t_ind] = agent_contribution(agents, tasks, i, t_ind, coalition_structure[t_ind], constraints, gamma)
                    move_vals[i][t_ind] = task_cons[i][t_ind] - cur_con[i]

        max_moveIndexs = [
            np.argmax([move_vals[i][j] for j in a_taskInds[i] + [task_num]])
            for i in range(0, agent_num)
        ]
        max_moveVals = [
            move_vals[i][a_taskInds[i][max_moveIndexs[i]]]
            if max_moveIndexs[i] < len(a_taskInds[i])
            else move_vals[i][task_num]
            for i in range(0, agent_num)
        ]

    return (
        coalition_structure,
        sys_rewards_tasks(tasks, agents, coalition_structure, gamma),
        iteration,
        re_assign,
    )


def eGreedyNE2(agents, tasks, constraints, agentIds, taskIds, coalition_structure=[], eps=0, gamma=1):
    """
    Solve the problem using GreedyNE, but only consider the agents and tasks in the given agentIds and taskIds
    """
    new_agents = [agents[i] for i in agentIds]
    new_tasks = [tasks[j] for j in taskIds]
    reverse_taskIds_map = {j: j1 for j1, j in enumerate(taskIds)}
    reverse_agentIds_map = {i: i1 for i1, i in enumerate(agentIds)}
    new_constraints = [[], []]
    for i in agentIds:
        new_constraints[0].append([reverse_taskIds_map[j] for j in constraints[0][i]])
    for j in taskIds:
        new_constraints[1].append([reverse_agentIds_map[i] for i in constraints[1][j]])
    coalition_structure, sys_reward, iteration_count, re_assign_count = eGreedy2(new_agents, new_tasks, new_constraints, eps, gamma, coalition_structure)
    return (
        [agentIds[i] for i in coalition_structure[j]]
        for j in range(0, len(taskIds))
    ), sys_reward, iteration_count, re_assign_count
    


def eGreedyDNF(agents, tasks, constraints, dnf_tree, eps=0, gamma=1, coalition_structure=[]):
    """
    Solve the problem using GreedyNE when the tasks are organized in a single OR-AND tree (Disjunctive Normal Form)
    """
    max_reward = 0
    for x, subtree in enumerate(dnf_tree):
        subtasksIds = [subtree] if isinstance(subtree, int) else subtree
        subagentIds = range(0, len(agents))
        coalition_structure, sys_reward, iteration_count, re_assign_count = eGreedyNE2(agents, tasks, constraints, subagentIds, subtasksIds, coalition_structure, eps, gamma)
        if sys_reward > max_reward:
            max_reward = sys_reward
            max_coalition_structure = coalition_structure
    return max_coalition_structure, max_reward, iteration_count, re_assign_count