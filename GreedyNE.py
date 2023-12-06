from functools import reduce  # python3 compatibility
import itertools
import json
import math
import numpy as np
from operator import mul
import os
import time


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
        for _ in range(task_num)
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
    a_taskInds = [[]] * agent_num
    a_taskNums = []
    for a_id in range(0, agent_num):
        a_max_edge = min((available_seats - a_min_edge * (agent_num - 1 - a_id)), t_max_edge, task_num)
        a_min_edge = min(a_min_edge, a_max_edge)
        
        # radomly indicate the number of task the agent could work on, based on the maximum and minimum number of tasks the agent could work on
        a_taskNum = np.random.randint(a_min_edge, a_max_edge + 1)
        
        a_taskNums.append(a_taskNum)
        
        available_seats -= a_taskNum

    t_agents_counts = [0] * task_num  # each indicate the current number of agents on the task

    # make sure no further draw for those reached the maximum limit.
    t_indexes = [
        t_id for t_id in range(0, task_num) if t_agents_counts[t_id] < t_max_edge
    ]

    for a_id, a_taskNum in enumerate(a_taskNums):
        if any(tc == 0 for tc in t_agents_counts):  # if there are tasks that have not been allocated to any agent
            t_prob = [
                (math.e ** (t_max_edge - t_agents_counts[t_id])) ** power
                for t_id in t_indexes
            ]  # power is used to manify the probability
            sum_prob = sum(t_prob)
            t_prop_2 = [prop / sum_prob for prop in t_prob]

            # draw tasks accounting to their current allocations
            a_taskInds[a_id] = list(
                np.random.choice(
                    a=t_indexes,
                    size=min(a_taskNum, len(t_indexes)),
                    replace=False,
                    p=[prop / sum_prob for prop in t_prob],
                )
            )
            # increase the chosen task counters
        else:
            a_taskInds[a_id] = list(
                np.random.choice(
                    a=t_indexes, size=min(a_taskNum, len(t_indexes)), replace=False
                )
            )

        for t_id in a_taskInds[a_id]:
            t_agents_counts[t_id] += 1

        # make sure no further draw for those reached the maximum limit.
        t_indexes = [
            t_id for t_id in range(0, task_num) if t_agents_counts[t_id] < t_max_edge
        ]

    # get also the list of agents for each task
    t_agents = [
        [a_id for a_id in range(0, agent_num) if t_id in a_taskInds[a_id]]
        for t_id in range(0, task_num)
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
        t_caps = [tasks[t_id] for t_id in a_taskInd]  # lists of caps that each task agent could perform

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
        a_cap_tasks = [[t_id for t_id in a_taskInd if t_id != task_num and c in tasks[t_id]] for a_taskInd in a_taskInds] 

        # sort the agents based on their contribution values for the capability c, in descending order
        cap_rank_pos = np.argsort(a_cap_vals)[::-1]

        a_cap_vals_ordered = [0] * agent_num
        a_cap_tasks_ordered = [[]] * agent_num
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


def task_reward(task, agents, gamma=1):
    # task is represented by a list of capabilities it requires, agents is a list agents, where each represented by a list cap contribution values
    """
    Calculate the reward of a task
    :param: `task`: the list of capabilities the task requires
    :param: `agents`: the list of agents
    :param: `gamma`: the discount factor
    :return: the reward of the task
    """
    if agents == []:
        return 0
    else:
        return sum([max([agent[c] for agent in agents]) for c in task]) * (gamma ** len(agents))


def sys_reward_agents(agents, tasks, alloc, gamma=1):
    # alloc is a vector of size M each element indicate which task the agent is allocated to
    return sum(
        task_reward(task, [agent for a_id, agent in enumerate(agents) if alloc[a_id] == t_id], gamma)
        for t_id, task in enumerate(tasks)
    )


def sys_rewards_tasks(tasks, agents, coalition_structure, gamma=1):
    return sum(
        task_reward(task, [agents[a_id] for a_id in coalition_structure[t_id]], gamma)
        for t_id, task in enumerate(tasks) 
    )


def agent_con(agents, tasks, query_agentIndex, query_taskIndex, cur_task_alloc, constraints, gamma=1):
    a_taskInds = constraints[0]
    if query_taskIndex == len(tasks):
        return 0
    if query_taskIndex not in a_taskInds[query_agentIndex]:
        return 0
    cur_reward = task_reward(tasks[query_taskIndex], [agents[a_id] for a_id in cur_task_alloc], gamma)
    if query_agentIndex in cur_task_alloc:
        agents_list = [agents[i] for i in cur_task_alloc if i != query_agentIndex]
        return cur_reward - task_reward(tasks[query_taskIndex], agents_list, gamma)
    else:
        agents_list = [agents[i] for i in cur_task_alloc]
        agents_list.append(agents[query_agentIndex])
        return task_reward(tasks[query_taskIndex], agents_list, gamma) - cur_reward


def eGreedy2(agents, tasks, constraints, eps=0, gamma=1, coalition_structure=[]):
    re_assign = 0
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)
    alloc = [task_num] * agent_num  # each indicate the current task that agent i is allocated to, if = N, means not allocated
    if coalition_structure == []:
        coalition_structure = [[]] * (task_num + 1)  # current coalition structure, the last one is dummy coalition
        cur_con = [0] * agent_num
    else:
        coalition_structure.append([])
        for t_id in range(0, task_num):
            for a_id in coalition_structure[t_id]:
                alloc[a_id] = t_id
        cur_con = [
            agent_con(agents, tasks, a_id, t_id, coalition_structure[t_id], constraints, gamma)
            for a_id, t_id in enumerate(alloc)
        ]

    task_cons = [
        [
            agent_con(agents, tasks, a_id, t_id, coalition_structure[t_id], constraints, gamma)
            if t_id in a_taskInds[a_id]
            else -1000
            for t_id in range(0, task_num)
        ]
        + [0]
        for a_id in range(0, agent_num)
    ]
    # the last 0 indicate not allocated

    move_vals = [
        [
            task_cons[a_id][t_id] - cur_con[a_id] if t_id in a_taskInds[a_id] + [task_num] else -1000
            for t_id in range(0, task_num + 1)
        ]
        for a_id in range(0, agent_num)
    ]

    max_moveIndexs = [
        np.argmax([move_vals[a_id][t_id] for t_id in a_taskInds[a_id]] + [0])
        for a_id in range(0, agent_num)
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
        # when eps = 1, it's Random, when eps = 0, it's Greedy
        feasible_choices = [i for i in range(0, agent_num) if max_moveVals[i] > 0]
        if feasible_choices == []:
            break  # reach NE solution
        if np.random.uniform() <= eps:
            # exploration: random allocation
            a_index = np.random.choice(feasible_choices)
            t_index = (
                a_taskInds[a_index][max_moveIndexs[a_index]]
                if max_moveIndexs[a_index] < len(a_taskInds[a_index])
                else task_num
            )

        else:
            # exploitation: allocationelse based on reputation or efficiency
            a_index = np.argmax(max_moveVals)
            t_index = (
                a_taskInds[a_index][max_moveIndexs[a_index]]
                if max_moveIndexs[a_index] < len(a_taskInds[a_index])
                else task_num
            )

        # perfom move
        old_t_index = alloc[a_index]
        alloc[a_index] = t_index
        coalition_structure[t_index].append(a_index)

        affected_a_indexes = []
        affected_t_indexes = []
        # update agents in the new coalition
        if t_index != task_num:
            affected_a_indexes.extend(coalition_structure[t_index])
            affected_t_indexes.append(t_index)

            # task_cons[i][t_index]
            for a_id in coalition_structure[t_index]:
                task_cons[a_id][t_index] = agent_con(
                    agents,
                    tasks,
                    a_id,
                    t_index,
                    coalition_structure[t_index],
                    constraints,
                    gamma,
                )
                cur_con[a_id] = task_cons[a_id][t_index]
        else:
            affected_a_indexes.append(a_index)
            task_cons[a_index][t_index] = 0
            cur_con[a_index] = 0

        # update agent in the old coalition (if applicable)
        if (
            old_t_index != task_num
        ):  # if agents indeed moved from another task, we have to change every agent from the old as well
            re_assign += 1
            coalition_structure[old_t_index].remove(a_index)
            affected_a_indexes.extend(coalition_structure[old_t_index])
            affected_t_indexes.append(old_t_index)
            for a_id in coalition_structure[old_t_index]:
                task_cons[a_id][old_t_index] = agent_con(
                    agents,
                    tasks,
                    a_id,
                    old_t_index,
                    coalition_structure[old_t_index],
                    constraints,
                    gamma,
                )

                cur_con[a_id] = task_cons[a_id][old_t_index]

        for a_id in affected_a_indexes:
            move_vals[a_id] = [
                task_cons[a_id][j] - cur_con[a_id]
                if j in a_taskInds[a_id] + [task_num]
                else -1000
                for j in range(0, task_num + 1)
            ]

        ## updating other agents r.w.t the affected tasks
        for t_ind in affected_t_indexes:
            for a_id in range(0, agent_num):
                if (a_id not in coalition_structure[t_ind]) and (t_ind in a_taskInds[a_id]):
                    task_cons[a_id][t_ind] = agent_con(
                        agents,
                        tasks,
                        a_id,
                        t_ind,
                        coalition_structure[t_ind],
                        constraints,
                        gamma,
                    )
                    move_vals[a_id][t_ind] = task_cons[a_id][t_ind] - cur_con[a_id]

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


def resultCal(agents, tasks, constraints, r_msgs, q_msgs, iteration, iter_over, converge, gamma=1):
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)

    a_msg_sum = [
        {
            d_key: sum([r_msgs[j][i][0] for j in a_taskInds[i] if j != d_key])
            + r_msgs[d_key][i][1]
            for d_key in a_taskInds[i]
        }
        for i in range(0, agent_num)
    ]

    alloc = [max(ams, key=ams.get) if ams != {} else task_num for ams in a_msg_sum]

    return (
        alloc,
        sys_reward_agents(agents, tasks, alloc, gamma),
        iteration,
        iter_over,
        converge,
    )


def FMS(agents, tasks, constraints, gamma, time_bound=500):
    converge = False
    iter_over = False
    start_time = time.perf_counter()
    a_taskInds = constraints[0]
    t_agentInds = constraints[1]
    task_num = len(tasks)
    agent_num = len(agents)

    q_msgs = [{t_key: {} for t_key in a_taskInds[i]} for i in range(0, agent_num)]
    r_msgs = [
        {
            t_agentInds[j][i]: (
                {1: -100}
                if len(a_taskInds[t_agentInds[j][i]]) == 1
                else {key: -100 for key in [0, 1]}
            )
            for i in range(0, len(t_agentInds[j]))
        }
        for j in range(0, task_num)
    ]

    q_flags = [False for i in range(0, agent_num)]
    r_flags = [False for j in range(0, task_num)]

    iteration = 0
    while True:
        if time.perf_counter() - start_time >= time_bound:
            return resultCal(
                agents,
                tasks,
                constraints,
                r_msgs,
                q_msgs,
                iteration,
                iter_over,
                converge,
                gamma,
            )
        iteration += 1

        if iteration > agent_num + task_num:
            iter_over = True
            return resultCal(
                agents,
                tasks,
                constraints,
                r_msgs,
                q_msgs,
                iteration,
                iter_over,
                converge,
                gamma,
            )

        if all(q_flags) and all(r_flags):  # converge, msgs are all the same.
            converge = True
        #             break
        for i in range(0, agent_num):
            linked_taskInds = a_taskInds[i]

            flag = True
            for t_key in linked_taskInds:
                ####### check time bound
                if time.perf_counter() - start_time >= time_bound:
                    return resultCal(
                        agents,
                        tasks,
                        constraints,
                        r_msgs,
                        q_msgs,
                        iteration,
                        iter_over,
                        converge,
                        gamma,
                    )
                ####### check time bound
                msgs = {}

                if len(linked_taskInds) > 1:
                    msgs[1] = sum(
                        [
                            m[0]
                            for m in [
                                r_msgs[j][i] for j in linked_taskInds if j != t_key
                            ]
                        ]
                    )
                    msg_0 = []
                    ts = list(linked_taskInds)
                    ts.remove(t_key)

                    for k in ts:
                        msg_0.append(
                            sum([m[0] for m in [r_msgs[j][i] for j in ts if j != k]])
                            + r_msgs[k][i][1]
                        )

                    msgs[0] = 0 if msg_0 == [] else max(msg_0)
                else:
                    msgs[1] = 0

                alphas = -sum(msgs.values()) / len(msgs.keys())

                msgs_regularised = {
                    d_key: msgs[d_key] + alphas for d_key in msgs.keys()
                }

                old_msg = q_msgs[i][t_key]
                if old_msg != {} and any(
                    [
                        abs(msgs_regularised[d_key] - old_msg[d_key]) > 10 ** (-5)
                        for d_key in old_msg.keys()
                    ]
                ):
                    flag = False

                q_msgs[i][t_key] = msgs_regularised

            if flag:  # agent i sending the same info
                q_flags[i] = True

        if time.perf_counter() - start_time >= time_bound:
            break
        ###################### SAME thing, using comprehension
        #             msgs = {t_key:{d_key:sum([m[d_key] for m in [r_msgs[j][i] for j in linked_taskInds if j != t_key]])
        #                            for d_key in linked_taskInds}
        #                                 for t_key in linked_taskInds}
        #             alphas = {t_key:-sum(msgs[t_key].values())/len(msgs.keys())
        #                       for t_key in linked_taskInds}
        #             msgs_regularised = {t_key:{d_key:msgs[t_key][d_key] + alphas[t_key]
        #                            for d_key in linked_taskInds}
        #                                 for t_key in linked_taskInds}
        for j in range(0, task_num):
            linked_agentInds = t_agentInds[j]
            msg_con = [q_msgs[a][j] for a in linked_agentInds]

            com_dict = []
            com_rewards = []
            dom_com = [
                [0, 1] if len(a_taskInds[i]) > 1 else [1] for i in linked_agentInds
            ]

            for c in itertools.product(*dom_com):
                ####### check time bound
                if time.perf_counter() - start_time >= time_bound:
                    return resultCal(
                        agents,
                        tasks,
                        constraints,
                        r_msgs,
                        q_msgs,
                        iteration,
                        iter_over,
                        converge,
                        gamma,
                    )
                ####### check time bound

                com_dict.append({linked_agentInds[i]: c[i] for i in range(0, len(c))})
                com_rewards.append(
                    task_reward(
                        tasks[j],
                        [
                            agents[a_key]
                            for a_key in com_dict[-1].keys()
                            if com_dict[-1][a_key] == 1
                        ],
                        gamma,
                    )
                )

            flag = True
            for a_key in linked_agentInds:
                ####### check time bound
                if time.perf_counter() - start_time >= time_bound:
                    return resultCal(
                        agents,
                        tasks,
                        constraints,
                        r_msgs,
                        q_msgs,
                        iteration,
                        iter_over,
                        converge,
                        gamma,
                    )
                ####### check time bound

                old_msg = r_msgs[j][a_key]
                q_table = []
                for c in range(0, len(com_dict)):
                    q_table.append(
                        sum(
                            [
                                q_msgs[a][j][com_dict[c][a]]
                                for a in linked_agentInds
                                if a != a_key
                            ]
                        )
                        + com_rewards[c]
                    )

                r_msgs[j][a_key] = {
                    d_key: max(
                        [
                            q_table[c]
                            for c in range(0, len(com_dict))
                            if com_dict[c][a_key] == d_key
                        ]
                    )
                    for d_key in ([0, 1] if len(a_taskInds[a_key]) > 1 else [1])
                }

                if any(
                    [
                        abs(r_msgs[j][a_key][d_key] - old_msg[d_key]) > 10 ** (-5)
                        for d_key in old_msg.keys()
                    ]
                ):
                    flag = False

            if flag:  # task j sending the same info
                r_flags[j] = True

    return resultCal(
        agents,
        tasks,
        constraints,
        r_msgs,
        q_msgs,
        iteration,
        iter_over,
        converge,
        gamma,
    )


def OPD(agents, tasks, constraints, gamma):
    edge_pruned = 0
    task_num = len(tasks)
    agent_num = len(agents)
    a_taskInds = [list(con) for con in constraints[0]]
    t_agentInds = [list(con) for con in constraints[1]]

    a_ubs = [[0 for j in a_taskInds[i]] for i in range(0, agent_num)]
    a_lbs = [[0 for j in a_taskInds[i]] for i in range(0, agent_num)]

    for j in range(0, task_num):
        linked_agentInds = t_agentInds[j]
        com_dict = []
        com_rewards = []
        for c in itertools.product(*[[0, 1] for i in linked_agentInds]):
            com_dict.append({linked_agentInds[i]: c[i] for i in range(0, len(c))})
            com_rewards.append(
                task_reward(
                    tasks[j],
                    [
                        agents[a_key]
                        for a_key in com_dict[-1].keys()
                        if com_dict[-1][a_key] == 1
                    ],
                    gamma,
                )
            )

        for i in t_agentInds[j]:
            t_ind = a_taskInds[i].index(j)
            a_ind = t_agentInds[j].index(i)
            reward_t = [
                com_rewards[c] for c in range(0, len(com_dict)) if com_dict[c][i] == 1
            ]
            reward_f = [
                com_rewards[c] for c in range(0, len(com_dict)) if com_dict[c][i] == 0
            ]
            cons_j = [reward_t[k] - reward_f[k] for k in range(0, len(reward_t))]
            a_lbs[i][t_ind] = min(cons_j)
            a_ubs[i][t_ind] = max(cons_j)

    for i in range(0, agent_num):
        t_flag = [True for j in a_taskInds[i]]
        for t_ind in range(0, len(a_taskInds[i])):
            for t2_ind in range(0, len(a_taskInds[i])):
                if t_ind != t2_ind and a_ubs[i][t_ind] < a_lbs[i][t2_ind]:
                    t_flag[t_ind] = False
                    edge_pruned += 1
                    break

        for t_ind in range(0, len(t_flag)):
            if not t_flag[t_ind]:
                t_agentInds[a_taskInds[i][t_ind]].remove(i)

        new_a_taskInds = [
            a_taskInds[i][t_ind]
            for t_ind in range(0, len(a_taskInds[i]))
            if t_flag[t_ind]
        ]
        a_taskInds[i] = new_a_taskInds

    return a_taskInds, t_agentInds, edge_pruned


def random(agents, tasks, constraints, gamma=1):
    task_num = len(tasks)
    agent_num = len(agents)
    a_taskInds = constraints[0]
    alloc = [np.random.choice(a_taskInds[i] + [task_num]) for i in range(0, agent_num)]
    return alloc, sys_reward_agents(agents, tasks, alloc, gamma)


def alloc_to_CS(tasks, alloc):
    task_num = len(tasks)
    CS = [[] for j in range(0, len(tasks))]
    for i in range(0, len(alloc)):
        if alloc[i] < task_num:  # means allocated (!=task_num)
            CS[alloc[i]].append(i)
    return CS


def append_record(record, filename, typ):
    with open(filename, "a") as f:
        if typ != "":
            json.dump(record, f, default=typ)
        else:
            json.dump(record, f)
        f.write(os.linesep)
        f.close()
