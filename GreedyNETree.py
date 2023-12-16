import numpy as np

from CalcRewards import *
from TreeUtils import *


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


def calc_temp_node_values(query_a_index, tasks, tree_info, allocation_structure, cur_con, realtime_node_values, root_node_index):
    """
    Calculate temp_node_values[i], for when agent i is removed from the system
    """
    temp_node_values_i = realtime_node_values.copy()
    a_current_t_id = allocation_structure[query_a_index]

    sys_lost_value = cur_con[query_a_index]
    temp_node_values_i[a_current_t_id] -= sys_lost_value
    
    parent_id = tree_info[a_current_t_id].parent_id
    while parent_id is not None and parent_id != len(tasks) and sys_lost_value > 0:
        
        if(tree_info[parent_id].node_type == NodeType.AND):
            temp_node_values_i[parent_id] -= sys_lost_value
        
        elif(tree_info[parent_id].node_type == NodeType.OR):
            new_parent_value = max([temp_node_values_i[j] for j in tree_info[parent_id].children_ids])
            sys_lost_value = temp_node_values_i[parent_id] - new_parent_value
            temp_node_values_i[parent_id] = new_parent_value

        if parent_id == root_node_index:
            break
        parent_id = tree_info[parent_id].parent_id

    return temp_node_values_i


def calc_max_move_value(query_a_index, tasks, tree_info, a_taskInds, allocation_structure, cur_con, task_cons, realtime_node_values, temp_node_values, task_num, root_node_index=-1):
    """
    Calculate the best move values for agent query_a_index
    """
    # Movement value for moving agent from current coalition to dummy coalition (removing the agent from the system):
    sys_exit_value = temp_node_values[query_a_index][root_node_index] - realtime_node_values[root_node_index]

    # Initialize the max move value
    max_move = (sys_exit_value, 0, task_num)
    
    # Calculate the best move values for each agent
    for j in a_taskInds[query_a_index]:

        if j == allocation_structure[query_a_index]:
            continue
    
        sys_added_value = task_cons[query_a_index][j]
        node_val = temp_node_values[query_a_index][j] + sys_added_value
        parent_id = tree_info[j].parent_id

        while parent_id is not None and parent_id != len(tasks) and (sys_added_value + sys_exit_value) >= max_move[0] and sys_added_value > 0:

            # Break conditions: 
            # parent_id is invalid (None) (i.e, node_id is root node) or parent_id is the dummy node
            # sys_added_value <= 0
            # (sys_added_value + sys_exit_value) < best_move_value

            # max_move_value only increases, and thus (max_move_value - move_vals_exit) always >= 0
            # meanwhile, sys_added_value only decreases

            parent_val = temp_node_values[query_a_index][parent_id]
            
            if(tree_info[parent_id].node_type == NodeType.AND):
                parent_val += sys_added_value
            
            elif(tree_info[parent_id].node_type == NodeType.OR):
                if (parent_val < node_val):
                    sys_added_value = node_val - parent_val
                    parent_val = node_val
                else:
                    sys_added_value = 0
                    break

            node_val = parent_val
            if parent_id == root_node_index:
                break
            parent_id = tree_info[parent_id].parent_id
        
        move_val_j = sys_exit_value + sys_added_value
        
        if move_val_j > max_move[0]:
            max_move = (move_val_j, task_cons[query_a_index][j] - cur_con[query_a_index], j)
        elif(move_val_j == max_move[0]):
            # Tie breaking: choose the one with higher contribution
            if task_cons[query_a_index][j] > task_cons[query_a_index][max_move[2]]:
                max_move = (move_val_j, task_cons[query_a_index][j] - cur_con[query_a_index], j)
            elif task_cons[query_a_index][j] == task_cons[query_a_index][max_move[2]]:
                # Tie breaking: choose the one that moves out of the dummy coalition
                if max_move[2] == task_num or j < max_move[2]:
                    max_move = (move_val_j, task_cons[query_a_index][j] - cur_con[query_a_index], j)
    
    return max_move
    

def calc_best_move(agent_num, task_num, tasks, tree_info, a_taskInds, allocation_structure, cur_con, task_cons, realtime_node_values, temp_node_values, root_node_index=-1):
    """
    Calculate the best move values for agent query_a_index
    """
    best_move = (float("-inf"), float("-inf"), allocation_structure[0], 0)
    for i in range(0, agent_num):
        # Movement value for moving agent from current coalition to dummy coalition (removing the agent from the system):
        sys_exit_value = temp_node_values[i][root_node_index] - realtime_node_values[root_node_index]
        
        # Calculate the best move values for each agent
        for j in a_taskInds[i]:

            if j == allocation_structure[i]:
                continue
        
            sys_added_value = task_cons[i][j]
            node_val = temp_node_values[i][j] + sys_added_value
            parent_id = tree_info[j].parent_id

            # Backtrack to the root node
            while parent_id is not None and parent_id != len(tasks) and (sys_added_value + sys_exit_value) >= best_move[0] and sys_added_value > 0:

                # Break conditions: 
                # parent_id is invalid (None) (i.e, node_id is root node) or parent_id is the dummy node
                # sys_added_value <= 0
                # (sys_added_value + sys_exit_value) < best_move_value

                # max_move_value only increases, and thus (max_move_value - move_vals_exit) always >= 0
                # meanwhile, sys_added_value only decreases

                parent_val = temp_node_values[i][parent_id]
                
                if(tree_info[parent_id].node_type == NodeType.AND):
                    parent_val += sys_added_value
                
                elif(tree_info[parent_id].node_type == NodeType.OR):
                    if (parent_val < node_val):
                        sys_added_value = node_val - parent_val
                        parent_val = node_val
                    else:
                        sys_added_value = 0
                        break

                node_val = parent_val
                if parent_id == root_node_index:
                    break
                parent_id = tree_info[parent_id].parent_id
            
            # Compare the calculated system move value with the best move value

            sys_move_val_i_j = sys_exit_value + sys_added_value
            task_move_val_i_j = task_cons[i][j] - cur_con[i]

            if(sys_move_val_i_j < best_move[0]):
                continue

            if sys_move_val_i_j > best_move[0]:
                best_move = (sys_move_val_i_j, task_move_val_i_j, j, i)
                continue       
            
            # else: sys_move_val_i_j == best_move[0]:
            
            # Tie breaking: choose the one with higher contribution to a single task
            if task_move_val_i_j > best_move[1]:
                best_move = (sys_move_val_i_j, task_move_val_i_j, j, i)
                continue
            
            # Tie breaking: choose the one that moves out of the dummy coalition
            if task_move_val_i_j == best_move[1] and allocation_structure[i] == task_num:
                best_move = (sys_move_val_i_j, task_move_val_i_j, j, i)
                continue
        
    return best_move


def greedyNETree(agents, tasks, constraints, tree_info : list[Node], root_node_index=-1, agentIds=None, eps=0, gamma=1, coalition_structure=[], strict_greedy=True, cur_con=None, task_cons=None, realtime_node_values=None, temp_node_values=None):
    """
    GreedyNE algorithm for solving the problem when the tasks are organized in strictly alternating AND-OR tree (i.e. each OR node has only AND children, and each AND node has only OR children)
    """    
    task_num = len(tasks)
    agent_num = len(agents)
    if agentIds is None:
        agentIds = list(range(0, agent_num))

    if root_node_index < 0:
        root_node_index = len(tree_info) + root_node_index
    
    nodes = list(traverse_tree_advanced(tree_info, order='dfs', root_node_index=root_node_index)) + [tree_info[task_num]]
    leaf_nodes = [node.node_id for node in nodes if node.node_type == NodeType.LEAF]
    a_taskInds = [[j for j in constraints[0][i] if j in leaf_nodes] for i in range(0, len(agents))]
    
    # each indicate the current task that agent i is allocated to, if = N, means not allocated
    allocation_structure = [task_num for i in range(0, agent_num)]

    # Initialize the coalition structure and contribution values of each agent to its current task
    if coalition_structure == None or coalition_structure == []:
        coalition_structure = [[] for j in range(0, task_num)] + [list(range(0, agent_num))]  # current coalition structure, the last one is dummy coalition
        if cur_con is None or cur_con == []:
            cur_con = [0 for i in range(0, agent_num)]
    else:
        for j in range(0, task_num + 1):
            for i in coalition_structure[j]:
                allocation_structure[i] = j
        # Contribution values of each agent to its current task
        if cur_con is None or cur_con == []:
            cur_con = [
                agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
                for i, j in enumerate(allocation_structure)
            ]

    # Contribution values of each agent to each task
    if task_cons is None or task_cons == []:
        task_cons = [
            [
                agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
                if j in a_taskInds[i]
                else float("-inf")
                for j in range(0, task_num)
            ] + [0]
            for i in range(0, agent_num)
        ]

    if not strict_greedy:
        # Max move values for each agent. Move value of an agent to new coalition j is the difference in the system value when the agent is moved from its current coalition to j.
        max_moves = [(float("-inf"), float("-inf"), task_num) for i in range(0, agent_num)]

    # Node values for the current coalition structure
    if realtime_node_values is None or realtime_node_values == []:
        realtime_node_values = [0 for node in range(0, len(tree_info))]
    
    # temp_node_values is used to store the alternative node values when the agents are removed from the system (i.e., moved to the dummy coalition)
    if temp_node_values is None or temp_node_values == []:
        temp_node_values = [[0 for node in range(0, len(tree_info))] for i in range(0, agent_num)]

        for i in agentIds:
            temp_node_values[i] = calc_temp_node_values(i, tasks, tree_info, allocation_structure, cur_con, realtime_node_values, root_node_index=root_node_index)

    if strict_greedy:
        best_move_NE = calc_best_move(agent_num, task_num, tasks, tree_info, a_taskInds, allocation_structure, cur_con, task_cons, realtime_node_values, temp_node_values, root_node_index=root_node_index)
    else:
        for i in agentIds:
            max_moves[i] = calc_max_move_value(i, tasks, tree_info, a_taskInds, allocation_structure, cur_con, task_cons, realtime_node_values, temp_node_values, task_num, root_node_index=root_node_index)

    re_assignment_count = 0
    iteration_count = 0
    while True:
        iteration_count += 1

        if strict_greedy:
            if not (best_move_NE[0] > 0 or (best_move_NE[0] == 0 and (best_move_NE[1] > 0 or (best_move_NE[1] == 0 and best_move_NE[2] != task_num and allocation_structure[best_move_NE[3]] == task_num)))): 
                break  # reach NE solution

            selected_a_index = best_move_NE[3]
            
            new_t_index = best_move_NE[2]

        else:
            feasible_choices = [i for i in agentIds if max_moves[i][0] > 0 or (max_moves[i][0] == 0 and (max_moves[i][1] > 0 or (max_moves[i][1] == 0 and max_moves[i][2] != task_num and allocation_structure[i] == task_num)))]
            if len(feasible_choices) == 0:
                break  # reach NE solution
            # when eps = 1, it's Random, when eps = 0, it's Greedy
            if np.random.uniform() <= eps:
                # exploration: random allocation
                selected_a_index = np.random.choice(feasible_choices)
            else:
                # exploitation: allocation else based on reputation or efficiency
                # selected_a_index = np.argmax(max_moves)
                selected_a_index = max(enumerate(max_moves), key=lambda x: (x[1][0], x[1][1], x[1][2] != task_num and allocation_structure[x[0]] == task_num))[0]
                
            new_t_index = max_moves[selected_a_index][2]

        # perform move
        old_t_index = allocation_structure[selected_a_index]
        allocation_structure[selected_a_index] = new_t_index
        coalition_structure[new_t_index].append(selected_a_index)
        coalition_structure[old_t_index].remove(selected_a_index)

        # print("iteration:", iteration_count, "  selected_a_index:", selected_a_index, "  old_t_index:", old_t_index, "  new_t_index:", new_t_index, "  max_moves:", max_moves[selected_a_index], "  task_cons_old:", task_cons[selected_a_index][old_t_index], "  task_cons_new:", task_cons[selected_a_index][new_t_index], "  cur_con:", cur_con[selected_a_index])

        # update agents in the new coalition
        affected_t_indexes = []
        if new_t_index != task_num:
            affected_t_indexes.append(new_t_index)
            for i in coalition_structure[new_t_index]:
                task_cons[i][new_t_index] = agent_contribution(agents, tasks, i, new_t_index, coalition_structure[new_t_index], constraints, gamma)
                cur_con[i] = task_cons[i][new_t_index]
        else:
            task_cons[selected_a_index][new_t_index] = 0
            cur_con[selected_a_index] = 0

        # update agent in the old coalition (if applicable)
        if (old_t_index != task_num):  
            # if agents indeed moved from another task, we have to change every agent from the old as well
            re_assignment_count += 1
            affected_t_indexes.append(old_t_index)
            for i in coalition_structure[old_t_index]:
                task_cons[i][old_t_index] = agent_contribution(agents, tasks, i, old_t_index, coalition_structure[old_t_index], constraints, gamma)
                cur_con[i] = task_cons[i][old_t_index]

        # update other agents with respect to the affected tasks
        for t_ind in affected_t_indexes:
            for i in agentIds:
                if (t_ind in a_taskInds[i]) and (i not in coalition_structure[t_ind]):
                    task_cons[i][t_ind] = agent_contribution(agents, tasks, i, t_ind, coalition_structure[t_ind], constraints, gamma)
                

        # Update real-time node values

        # First, update the node values when the chosen agent is removed from the system
        realtime_node_values = temp_node_values[selected_a_index].copy()

        # Second, update the node values when the chosen agent is added to the new coalition
        realtime_node_values[new_t_index] += cur_con[selected_a_index]
        sys_added_value = cur_con[selected_a_index]
        node_val = realtime_node_values[new_t_index]
        parent_id = tree_info[new_t_index].parent_id

        while parent_id is not None and parent_id != len(tasks) and sys_added_value > 0:
            parent_val = realtime_node_values[parent_id]

            if(tree_info[parent_id].node_type == NodeType.AND):
                realtime_node_values[parent_id] += sys_added_value
            
            elif(tree_info[parent_id].node_type == NodeType.OR):
                if (parent_val < node_val):
                    sys_added_value = node_val - parent_val
                    realtime_node_values[parent_id] = node_val
                else:
                    sys_added_value = 0
                    break

            node_val = realtime_node_values[parent_id]
            if parent_id == root_node_index:
                break
            parent_id = tree_info[parent_id].parent_id

        # For each agent, recalculate temp_node_values and the max move value
        for i in agentIds:
            if (i != selected_a_index):
                # We can skip calculating the temp_node_values of the selected agent, since it's just been updated
                temp_node_values[i] = calc_temp_node_values(i, tasks, tree_info, allocation_structure, cur_con, realtime_node_values, root_node_index=root_node_index)
                
        
        if strict_greedy:
            best_move_NE = calc_best_move(agent_num, task_num, tasks, tree_info, a_taskInds, allocation_structure, cur_con, task_cons, realtime_node_values, temp_node_values, root_node_index=root_node_index)
        else:
            for i in agentIds:
                max_moves[i] = calc_max_move_value(i, tasks, tree_info, a_taskInds, allocation_structure, cur_con, task_cons, realtime_node_values, temp_node_values, task_num, root_node_index=root_node_index)
                

    return (
        coalition_structure,
        realtime_node_values[root_node_index],
        iteration_count,
        re_assignment_count,
    )

