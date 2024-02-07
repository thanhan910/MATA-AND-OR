import numpy as np

from .rewards import task_reward, sys_rewards_tasks


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


def eGreedyNE(
        agents : list[list[float]], 
        tasks : list[list[int]], 
        constraints : tuple[list[list[int]], list[list[int]]],
        coalition_structure : list[list[int]] = [],
        eps=0, 
        gamma=1
    ):
    """
    The original GreedyNE algorithm implementation
    """
    re_assignment_count = 0
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)
    allocation_structure = [task_num for i in range(0, agent_num)]  # each indicate the current task that agent i is allocated to, if = N, means not allocated
    if coalition_structure is None or coalition_structure == []:
        coalition_structure = [[] for j in range(0, task_num)] + [list(range(0, agent_num))]  # default coalition structure, the last one is dummy coalition
        cur_con = [0 for j in range(0, agent_num)]
    else:
        if len(coalition_structure) == task_num:
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
            else float("-inf")
            for j in range(0, task_num)
        ] + [0]
        for i in range(0, agent_num)
    ]
    # the last 0 indicate not allocated

    move_vals = [
        [
            task_cons[i][j] - cur_con[i] if j in a_taskInds[i] + [task_num] else float("-inf")
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

    iteration_count = 0
    while True:
        iteration_count += 1
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
            re_assignment_count += 1
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
        iteration_count,
        re_assignment_count,
    )


def eGreedyNE_subset(
        agents : list[list[float]], 
        tasks : list[list[int]], 
        constraints : tuple[list[list[int]], list[list[int]]],
        current_coalition_structure : list[list[int]] = [],
        selected_tasks : list[int] = None,
        eps=0, 
        gamma=1
    ):
    """
    The original GreedyNE algorithm implementation, but performed on a subset of tasks.
    """
    if current_coalition_structure is None or current_coalition_structure == []:
        current_coalition_structure = [[] for j in range(0, len(tasks))] + [list(range(0, len(agents)))]

    if selected_tasks is None:
        selected_tasks = list(range(len(tasks)))

    tasks_subset = []
    temp_coalition_structure = []
    
    task_selected = [False for i in range(len(tasks))]

    for j in selected_tasks:
        tasks_subset.append(tasks[j])
        temp_coalition_structure.append(current_coalition_structure[j].copy())
        task_selected[j] = True
        
    dummy_coalition = []
    for j in range(len(tasks)):
        if not task_selected[j]:
            dummy_coalition += current_coalition_structure[j]

    temp_coalition_structure.append(dummy_coalition)

    temp_coalition_structure, system_reward, iteration_count, re_assignment_count = eGreedyNE(agents, tasks_subset, constraints, temp_coalition_structure, eps, gamma)

    new_coalition_structure = [[] if j in selected_tasks else current_coalition_structure[j] for j in range(len(tasks))] + [temp_coalition_structure[-1]]

    return new_coalition_structure, system_reward, iteration_count, re_assignment_count


def aGreedyNE(
        agents : list[list[float]], 
        tasks : list[list[int]],
        constraints : tuple[list[list[int]], list[list[int]]],
        coalition_structure : list[list[int]] = [],
        selected_tasks : list[int] = None,
        selected_agents : list[int] = None,
        eps=0, 
        gamma=1
    ):
    """
    GreedyNE on a subset of tasks.

    The target is to focus all agents on working on only the selected tasks.
    """
    re_assignment_count = 0
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)

    if selected_tasks is None:
        selected_tasks = list(range(len(tasks)))
        task_selected = [True for j in range(len(tasks))]

    else:    
        task_selected = [False for j in range(len(tasks))]
        for j in selected_tasks:
            task_selected[j] = True

    task_selected.append(True) # dummy task


    if selected_agents is None:
        selected_agents = list(range(0, agent_num))
        agent_selected = [True for i in range(len(agents))]

    else:
        agent_selected = [False for i in range(len(agents))]
        for i in selected_agents:
            agent_selected[i] = True


    allocation_structure = { i : task_num for i in selected_agents }
    # each indicate the current task that agent i is allocated to, if = N, means not allocated
    
    if coalition_structure is None or coalition_structure == []:
        coalition_structure = [[] for j in range(0, task_num)] + [list(range(0, agent_num))]  # default coalition structure, the last one is dummy coalition
        cur_con = { i : 0 for i in selected_agents }
    else:

        if len(coalition_structure) < task_num:
            coalition_structure.append([])

        for j in range(0, task_num):
            if not task_selected[j]:
                coalition_structure[task_num] += coalition_structure[j]
                coalition_structure[j] = []

        for j in range(0, task_num):
            for i in coalition_structure[j]:
                allocation_structure[i] = j

        cur_con = {
            i : agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            if j != task_num and task_selected[j]
            else 0
            for i, j in allocation_structure.items()
        }
        

    task_cons = {
        i : {
            j : agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            if j != task_num and task_selected[j]
            else 0 if j == task_num
            else float("-inf")
            for j in a_taskInds[i] + [task_num]
        }
        for i in selected_agents
    }
    # the last 0 indicate not allocated

    move_vals = {
        i : {
            j : task_cons[i][j] - cur_con[i] 
            if j == task_num or task_selected[j]
            else float("-inf")
            for j in a_taskInds[i] + [task_num]
        }
        for i in selected_agents
    }

    max_moves = {
        i : max(move_vals[i].items(), key=lambda x: x[1])
        for i in selected_agents
    }

    iteration_count = 0
    while True:
        iteration_count += 1
        feasible_choices = [i for i in selected_agents if max_moves[i][1] > 0]
        if feasible_choices == []:
            break  # reach NE solution
        # when eps = 1, it's Random, when eps = 0, it's Greedy
        if np.random.uniform() <= eps:
            # exploration: random allocation
            a_index = np.random.choice(feasible_choices)
            t_index = max_moves[a_index][0]
        else:
            # exploitation: allocationelse based on reputation or efficiency
            best_move = max(max_moves.items(), key=lambda x: x[1][1])
            a_index = best_move[0]
            t_index = best_move[1][0]

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
                if agent_selected[i]:
                    task_cons[i][t_index] = agent_contribution(agents, tasks, i, t_index, coalition_structure[t_index], constraints, gamma)
                    cur_con[i] = task_cons[i][t_index]
        else:
            affected_a_indexes.append(a_index)
            task_cons[a_index][t_index] = 0
            cur_con[a_index] = 0

        # update agent in the old coalition (if applicable)
        if (old_t_index != task_num):  
            # if agents indeed moved from another task, we have to change every agent from the old as well
            re_assignment_count += 1
            coalition_structure[old_t_index].remove(a_index)
            affected_a_indexes.extend(coalition_structure[old_t_index])
            affected_t_indexes.append(old_t_index)
            for i in coalition_structure[old_t_index]:
                task_cons[i][old_t_index] = agent_contribution(agents, tasks, i, old_t_index, coalition_structure[old_t_index], constraints, gamma)
                cur_con[i] = task_cons[i][old_t_index]

        for i in affected_a_indexes:
            move_vals[i] = {
                j : task_cons[i][j] - cur_con[i]
                if j == task_num or task_selected[j]
                else float("-inf")
                for j in a_taskInds[i] + [task_num]
            }


        ## update other agents w.r.t the affected tasks
        for t_ind in affected_t_indexes:
            for i in range(0, agent_num):
                if (i not in coalition_structure[t_ind]) and (t_ind in a_taskInds[i]):
                    task_cons[i][t_ind] = agent_contribution(agents, tasks, i, t_ind, coalition_structure[t_ind], constraints, gamma)
                    move_vals[i][t_ind] = task_cons[i][t_ind] - cur_con[i]

        max_moves = {
            i : max(move_vals[i].items(), key=lambda x: x[1])
            for i in selected_agents
        }


    return (
        coalition_structure,
        sys_rewards_tasks(tasks, agents, coalition_structure, gamma),
        iteration_count,
        re_assignment_count,
    )


def adGreedyNE(
        agents : list[list[float]], 
        tasks : list[list[int]],
        constraints : tuple[list[list[int]], list[list[int]]],
        original_coalition_structure : dict[int, list[int]] = None,
        selected_tasks : list[int] = None,
        selected_agents : list[int] = None,
        eps=0, 
        gamma=1,
        dummy_task_id = None
    ):
    """
    GreedyNE on a subset of tasks.

    The target is to focus all agents on working on only the selected tasks and skip all other tasks.
    """
    re_assignment_count = 0
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)

    if dummy_task_id is None:
        dummy_task_id = len(tasks)

    task_selected = {}
    if selected_tasks is None:
        selected_tasks = list(range(len(tasks)))
        task_selected = { j : True for j in range(len(tasks)) }
    else:    
        task_selected = { j : False for j in range(len(tasks)) }
        for j in selected_tasks:
            task_selected[j] = True
    task_selected[dummy_task_id] = True


    if selected_agents is None:
        selected_agents = list(range(0, agent_num))
        agent_selected = { i : True for i in range(len(agents)) }

    else:
        agent_selected = { i : True for i in range(len(agents)) }
        for i in selected_agents:
            agent_selected[i] = True

    new_coalition_structure = { j : [] for j in selected_tasks }

    new_coalition_structure[dummy_task_id] = []

    if original_coalition_structure is None or original_coalition_structure == {}:
        original_coalition_structure = {}
        new_coalition_structure[dummy_task_id] = list(range(0, agent_num))

    
    for j in range(0, task_num):
        if task_selected[j]:
            new_coalition_structure[j] = [i for i in original_coalition_structure.get(j, []) if agent_selected[i]]
        else:
            new_coalition_structure[dummy_task_id] += [i for i in original_coalition_structure.get(j, []) if agent_selected[i]]


    allocation_structure = {}
    cur_con = {}

    # for j in new_coalition_structure:
    #     for i in new_coalition_structure[j]:
    #         allocation_structure[i] = j
    #         if j != task_num:
    #             cur_con[i] = agent_contribution(agents, tasks, i, j, new_coalition_structure[j], constraints, gamma)
    #         else:
    #             cur_con[i] = 0

    allocation_structure = { i : j for j in new_coalition_structure for i in new_coalition_structure[j] }

    cur_con = { i : agent_contribution(agents, tasks, i, j, new_coalition_structure[j], constraints, gamma) if j != dummy_task_id else 0 for i, j in allocation_structure.items() }
        

    task_cons = {
        i : {
            j : agent_contribution(agents, tasks, i, j, new_coalition_structure[j], constraints, gamma)
            if j != dummy_task_id and task_selected[j]
            else 0 if j == dummy_task_id
            else float("-inf")
            for j in a_taskInds[i] + [dummy_task_id]
        }
        for i in selected_agents
    }
    # the last 0 indicate not allocated

    move_vals = {
        i : {
            j : task_cons[i][j] - cur_con[i] 
            if j == dummy_task_id or task_selected[j]
            else float("-inf")
            for j in a_taskInds[i] + [dummy_task_id]
        }
        for i in selected_agents
    }

    max_moves = {
        i : max(move_vals[i].items(), key=lambda x: x[1])
        for i in selected_agents
    }

    iteration_count = 0
    while True:
        iteration_count += 1
        feasible_choices = [i for i in selected_agents if max_moves[i][1] > 0]
        if feasible_choices == []:
            break  # reach NE solution
        # when eps = 1, it's Random, when eps = 0, it's Greedy
        if np.random.uniform() <= eps:
            # exploration: random allocation
            a_index = np.random.choice(feasible_choices)
            t_index = max_moves[a_index][0]
        else:
            # exploitation: allocationelse based on reputation or efficiency
            best_move = max(max_moves.items(), key=lambda x: x[1][1])
            a_index = best_move[0]
            t_index = best_move[1][0]

        # perfom move
        old_t_index = allocation_structure[a_index]
        allocation_structure[a_index] = t_index
        new_coalition_structure[t_index].append(a_index)

        # update agents in the new coalition
        affected_a_indexes = []
        affected_t_indexes = []
        if t_index != dummy_task_id:
            affected_a_indexes.extend(new_coalition_structure[t_index])
            affected_t_indexes.append(t_index)

            # task_cons[i][t_index]
            for i in new_coalition_structure[t_index]:
                if agent_selected[i]:
                    task_cons[i][t_index] = agent_contribution(agents, tasks, i, t_index, new_coalition_structure[t_index], constraints, gamma)
                    cur_con[i] = task_cons[i][t_index]
        else:
            affected_a_indexes.append(a_index)
            task_cons[a_index][t_index] = 0
            cur_con[a_index] = 0

        # update agent in the old coalition (if applicable)
        if (old_t_index != dummy_task_id):  
            # if agents indeed moved from another task, we have to change every agent from the old as well
            re_assignment_count += 1
            new_coalition_structure[old_t_index].remove(a_index)
            affected_a_indexes.extend(new_coalition_structure[old_t_index])
            affected_t_indexes.append(old_t_index)
            for i in new_coalition_structure[old_t_index]:
                task_cons[i][old_t_index] = agent_contribution(agents, tasks, i, old_t_index, new_coalition_structure[old_t_index], constraints, gamma)
                cur_con[i] = task_cons[i][old_t_index]

        for i in affected_a_indexes:
            move_vals[i] = {
                j : task_cons[i][j] - cur_con[i]
                if j == dummy_task_id or task_selected[j]
                else float("-inf")
                for j in a_taskInds[i] + [dummy_task_id]
            }


        ## update other agents w.r.t the affected tasks
        for t_ind in affected_t_indexes:
            for i in range(0, agent_num):
                if (i not in new_coalition_structure[t_ind]) and (t_ind in a_taskInds[i]):
                    task_cons[i][t_ind] = agent_contribution(agents, tasks, i, t_ind, new_coalition_structure[t_ind], constraints, gamma)
                    move_vals[i][t_ind] = task_cons[i][t_ind] - cur_con[i]

        max_moves = {
            i : max(move_vals[i].items(), key=lambda x: x[1])
            for i in selected_agents
        }


    return (
        new_coalition_structure,
        sum(
            task_reward(tasks[j], [agents[i] for i in new_coalition_structure[j]], gamma)
            for j in selected_tasks
        ),
        iteration_count,
        re_assignment_count,
    )


def alGreedyNE(
        agents : list[list[float]], 
        tasks : list[list[int]],
        constraints : tuple[list[list[int]], list[list[int]]],
        original_allocation_structure : dict[int, int] = None,
        selected_tasks : list[int] = None,
        selected_agents : list[int] = None,
        eps=0, 
        gamma=1,
        dummy_task_id = None
    ):
    """
    GreedyNE on a subset of tasks.

    The target is to focus all agents on working on only the selected tasks and skip all other tasks.
    """
    re_assignment_count = 0
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)

    if dummy_task_id is None:
        dummy_task_id = len(tasks)

    task_selected = {}
    if selected_tasks is None:
        selected_tasks = list(range(len(tasks)))
        task_selected = { j : True for j in range(len(tasks)) }
    else:    
        task_selected = { j : False for j in range(len(tasks)) }
        for j in selected_tasks:
            task_selected[j] = True
    task_selected[dummy_task_id] = True


    if selected_agents is None:
        selected_agents = list(range(0, agent_num))
        agent_selected = { i : True for i in range(len(agents)) }

    else:
        agent_selected = { i : True for i in range(len(agents)) }
        for i in selected_agents:
            agent_selected[i] = True


    if original_allocation_structure is None or original_allocation_structure == {}:
        original_allocation_structure = { i : dummy_task_id for i in selected_agents }


    allocation_structure = { i : dummy_task_id for i in selected_agents }
    for i, j in original_allocation_structure.items():
        if agent_selected[i]:
            allocation_structure[i] = j

    coalition_structure = { j : [] for j in selected_tasks + [dummy_task_id] }

    cur_con = { i : 0 for i in selected_agents }

    for i, j in allocation_structure.items():
        if task_selected[j]:
            coalition_structure[j].append(i)
            if j != dummy_task_id:
                cur_con[i] = agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            else:
                cur_con[i] = 0
        else:
            coalition_structure[dummy_task_id].append(i)
            cur_con[i] = 0

        

    task_cons = {
        i : {
            j : agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            if j != dummy_task_id and task_selected[j]
            else 0 if j == dummy_task_id
            else float("-inf")
            for j in a_taskInds[i] + [dummy_task_id]
        }
        for i in selected_agents
    }
    # the last 0 indicate not allocated

    move_vals = {
        i : {
            j : task_cons[i][j] - cur_con[i] 
            if j == dummy_task_id or task_selected[j]
            else float("-inf")
            for j in a_taskInds[i] + [dummy_task_id]
        }
        for i in selected_agents
    }


    max_moves = {
        i : max(move_vals[i].items(), key=lambda x: x[1])
        for i in selected_agents
    }

    iteration_count = 0
    while True:
        iteration_count += 1
        feasible_choices = [i for i in selected_agents if max_moves[i][1] > 0]
        if feasible_choices == []:
            break  # reach NE solution
        # when eps = 1, it's Random, when eps = 0, it's Greedy
        if np.random.uniform() <= eps:
            # exploration: random allocation
            a_index = np.random.choice(feasible_choices)
            t_index = max_moves[a_index][0]
        else:
            # exploitation: allocationelse based on reputation or efficiency
            best_move = max(max_moves.items(), key=lambda x: x[1][1])
            a_index = best_move[0]
            t_index = best_move[1][0]

        # perfom move
        old_t_index = allocation_structure[a_index]
        allocation_structure[a_index] = t_index
        coalition_structure[t_index].append(a_index)

        # update agents in the new coalition
        affected_a_indexes = []
        affected_t_indexes = []
        if t_index != dummy_task_id:
            affected_a_indexes.extend(coalition_structure[t_index])
            affected_t_indexes.append(t_index)

            # task_cons[i][t_index]
            for i in coalition_structure[t_index]:
                if agent_selected[i]:
                    task_cons[i][t_index] = agent_contribution(agents, tasks, i, t_index, coalition_structure[t_index], constraints, gamma)
                    cur_con[i] = task_cons[i][t_index]
        else:
            affected_a_indexes.append(a_index)
            task_cons[a_index][t_index] = 0
            cur_con[a_index] = 0

        # update agent in the old coalition (if applicable)
        if (old_t_index != dummy_task_id):  
            # if agents indeed moved from another task, we have to change every agent from the old as well
            re_assignment_count += 1
            coalition_structure[old_t_index].remove(a_index)
            affected_a_indexes.extend(coalition_structure[old_t_index])
            affected_t_indexes.append(old_t_index)
            for i in coalition_structure[old_t_index]:
                task_cons[i][old_t_index] = agent_contribution(agents, tasks, i, old_t_index, coalition_structure[old_t_index], constraints, gamma)
                cur_con[i] = task_cons[i][old_t_index]

        for i in affected_a_indexes:
            move_vals[i] = {
                j : task_cons[i][j] - cur_con[i]
                if j == dummy_task_id or task_selected[j]
                else float("-inf")
                for j in a_taskInds[i] + [dummy_task_id]
            }


        ## update other agents w.r.t the affected tasks
        for t_ind in affected_t_indexes:
            for i in range(0, agent_num):
                if (i not in coalition_structure[t_ind]) and (t_ind in a_taskInds[i]):
                    task_cons[i][t_ind] = agent_contribution(agents, tasks, i, t_ind, coalition_structure[t_ind], constraints, gamma)
                    move_vals[i][t_ind] = task_cons[i][t_ind] - cur_con[i]

        max_moves = {
            i : max(move_vals[i].items(), key=lambda x: x[1])
            for i in selected_agents
        }


    return (
        coalition_structure,
        allocation_structure,
        sum(
            task_reward(tasks[j], [agents[i] for i in coalition_structure[j]], gamma)
            for j in selected_tasks
        ),
        iteration_count,
        re_assignment_count,
    )
