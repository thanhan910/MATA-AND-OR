from .GreedyNE import agent_contribution, aGreedyNE 
from .node_type import NodeType
from .rewards import task_reward, sys_rewards_tasks

import numpy as np


def get_node_value_info(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        leaf2task: list[int],
        coalition_structure: list[list[int]],
        root_node_id=0,
    ):
    """
    Calculate the value of each node in the tree, given the allocation structure.
    """
    node_value_info = {}
    def get_node_value(node_id: int):
        
        if node_id in node_value_info:
            return node_value_info[node_id]
        
        node_type = node_type_info[node_id]
        
        if node_type == NodeType.LEAF:
            task_id = leaf2task[node_id]
            node_value_info[node_id] = task_reward(task_id, coalition_structure[task_id])
        
        elif node_type == NodeType.AND:
            node_value_info[node_id] = sum(get_node_value(child_id) for child_id in children_info[node_id])
        
        else: # OR node
            node_value_info[node_id] = max(get_node_value(child_id) for child_id in children_info[node_id])

        return node_value_info[node_id]
    
    get_node_value(root_node_id)

    return node_value_info


def get_cur_con_tree(
        query_aId: int,
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        task2leaf: list[int],
        allocation_structure: list[int],
        node_value_info: dict[int, float],
        cur_con_info: list[float],
        root_node_id=0,
    ):
    """
    Calculate the change in nodes values when an agent defects from its current task.
    """
    nodes_alt_value = {}

    task_id = allocation_structure[query_aId]

    if task_id == len(task2leaf):
        return nodes_alt_value, 0
    
    current_node = task2leaf[allocation_structure[query_aId]]
    
    value_lost = cur_con_info[query_aId]
    
    nodes_alt_value[current_node] = node_value_info[current_node] - value_lost
    
    while current_node != root_node_id and value_lost > 0:
        parent_node = parent_info[current_node]
        if node_type_info[parent_node] == NodeType.AND:
            nodes_alt_value[parent_node] = node_value_info[parent_node] - value_lost
        else: # OR node
            new_parent_value = max(nodes_alt_value.get(child_id, node_value_info[child_id]) for child_id in children_info[parent_node])
            value_lost = node_value_info[parent_node] - new_parent_value
            if value_lost > 0:
                nodes_alt_value[parent_node] = new_parent_value
            else:
                break
        current_node = parent_node
    
    return nodes_alt_value, value_lost


def get_move_val_tree(
        query_aId: int,
        query_tId: int,
        deflect_nodes_alt_value: dict[int, float],
        node_type_info: dict[int, NodeType],
        parent_info: dict[int, int],
        task2leaf: list[int],
        node_value_info: dict[int, float],
        task_cons_info: list[list[float]],
        root_node_id=0,
        value_added_benchmark=0,
    ):

    nodes_alt_value = {}
    
    value_added = task_cons_info[query_aId][query_tId]

    current_node = task2leaf[query_tId]
    
    nodes_alt_value[current_node] = deflect_nodes_alt_value.get(current_node, node_value_info[current_node]) + value_added
    
    while current_node != root_node_id and value_added > 0 and value_added >= value_added_benchmark:
        parent_node = parent_info[current_node]
        current_parent_value = deflect_nodes_alt_value.get(parent_node, node_value_info[parent_node])
        if node_type_info[parent_node] == NodeType.AND:
            nodes_alt_value[parent_node] = current_parent_value + value_added
        else: # OR node
            if nodes_alt_value[current_node] > current_parent_value:
                value_added = nodes_alt_value[current_node] - current_parent_value
                nodes_alt_value[parent_node] = nodes_alt_value[current_node]
            else:
                value_added = 0
                break
        current_node = parent_node
    
    return nodes_alt_value, value_added


def treeGNE(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        task2leaf: list[int],
        leaf2task: dict[int, int],
        tasks: list[list[int]],
        agents: list[dict[int, float]],
        constraints,
        coalition_structure : list[list[int]] = [],
        selected_tasks : list[int] = None,
        eps=0, 
        gamma=1,
        root_node_id=0,
    ):
    
    """
    GreedyNE on an AND-OR tree.

    At each iteration, calculate the change in nodes values when an agent defects from its current task, and the change in nodes values when an agent moves to a new task.
    """
    re_assignment_count = 0
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)

    if selected_tasks is None:
        selected_tasks = list(range(len(tasks)))
        task_selected = [True for i in range(len(tasks))]

    else:    
        task_selected = [False for i in range(len(tasks))]
        for j in selected_tasks:
            task_selected[j] = True

    allocation_structure = [task_num for i in range(0, agent_num)]  # each indicate the current task that agent i is allocated to, if = N, means not allocated
    if coalition_structure is None or coalition_structure == []:
        coalition_structure = [[] for j in range(0, task_num)] + [list(range(0, agent_num))]  # default coalition structure, the last one is dummy coalition
        cur_con = [0 for j in range(0, agent_num)]
    else:

        if len(coalition_structure) < task_num:
            coalition_structure.append([])

        for j in range(0, task_num):
            if not task_selected[j]:
                coalition_structure[len(task_num)] += coalition_structure[j]
                coalition_structure[j] = []

        for j in range(0, task_num):
            for n_id in coalition_structure[j]:
                allocation_structure[n_id] = j

        cur_con = [
            agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            if j != task_num and task_selected[j]
            else 0
            for i, j in enumerate(allocation_structure)
        ]

    task_cons = [
        [
            agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            if j in a_taskInds[i] and task_selected[j]
            else float("-inf")
            for j in range(0, task_num)
        ] + [0]
        for i in range(0, agent_num)
    ]
    # the last 0 indicate not allocated


    node_value_info = get_node_value_info(node_type_info, children_info, leaf2task, coalition_structure, root_node_id)

    info_get_cur_con = {
        i: get_cur_con_tree(i, node_type_info, children_info, parent_info, task2leaf, allocation_structure, node_value_info, cur_con, root_node_id)
        for i in range(agent_num)
    }

    deflect_nodes_alt_value_info = {
        i: info_get_cur_con[i][0]
        for i in range(agent_num)
    }

    value_lost_info = {
        i: info_get_cur_con[i][1]
        for i in range(agent_num)
    }

    info_get_move_val_tree = {
        i: {
            j: get_move_val_tree(i, j, deflect_nodes_alt_value_info[i], node_type_info, parent_info, task2leaf, node_value_info, task_cons, root_node_id)
            for j in a_taskInds[i] if task_selected[j]
        }
        for i in range(agent_num)
    }

    added_nodes_alt_value_info = {
        i: {
            j: info_get_move_val_tree[i][j][0]
            for j in a_taskInds[i] if task_selected[j]
        }
        for i in range(agent_num)
    }

    value_added_info = {
        i: {
            j: info_get_move_val_tree[i][j][1]
            for j in a_taskInds[i] if task_selected[j]
        }
        for i in range(agent_num)
    }

    sys_improvement_values_info = {
        i: {
            j: value_added_info[i][j] - value_lost_info[i]
            for j in a_taskInds[i] if task_selected[j]
        }
        for i in range(agent_num)
    }

    max_sys_improvement_values = {
        i: max(sys_improvement_values_info[i].items(), key=lambda x: x[1])
        for i in range(agent_num)
    }


    iteration_count = 0
    while True:
        iteration_count += 1
        feasible_choices = [i for i in range(0, agent_num) if max_sys_improvement_values[i][1] > 0]
        if feasible_choices == []:
            break  # reach NE solution
        # when eps = 1, it's Random, when eps = 0, it's Greedy
        if np.random.uniform() <= eps:
            # exploration: random allocation
            a_index = np.random.choice(feasible_choices)
            t_index = max_sys_improvement_values[a_index][0]
        else:
            # exploitation: allocationelse based on reputation or efficiency
            best_sys_improvement_value = max(max_sys_improvement_values.items(), key=lambda x: x[1][1])
            a_index, t_index = best_sys_improvement_value[0], best_sys_improvement_value[1][0]

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
            for n_id in coalition_structure[t_index]:
                task_cons[n_id][t_index] = agent_contribution(agents, tasks, n_id, t_index, coalition_structure[t_index], constraints, gamma)
                cur_con[n_id] = task_cons[n_id][t_index]
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
            for n_id in coalition_structure[old_t_index]:
                task_cons[n_id][old_t_index] = agent_contribution(agents, tasks, n_id, old_t_index, coalition_structure[old_t_index], constraints, gamma)
                cur_con[n_id] = task_cons[n_id][old_t_index]

        ## update other agents w.r.t the affected tasks
        for t_ind in affected_t_indexes:
            for n_id in range(0, agent_num):
                if (n_id not in coalition_structure[t_ind]) and (t_ind in a_taskInds[n_id]):
                    task_cons[n_id][t_ind] = agent_contribution(agents, tasks, n_id, t_ind, coalition_structure[t_ind], constraints, gamma)

        # update node values
        for n_id, n_value in deflect_nodes_alt_value_info[a_index].items():
            node_value_info[n_id] = n_value

        for n_id, n_value in added_nodes_alt_value_info[a_index][t_index].items():
            node_value_info[n_id] = n_value


        info_get_cur_con = {
            i: get_cur_con_tree(i, node_type_info, children_info, parent_info, task2leaf, allocation_structure, node_value_info, cur_con)
            for i in range(agent_num)
        }

        deflect_nodes_alt_value_info = {
            i: info_get_cur_con[i][0]
            for i in range(agent_num)
        }

        value_lost_info = {
            i: info_get_cur_con[i][1]
            for i in range(agent_num)
        }

        info_get_move_val_tree = {
            i: {
                j: get_move_val_tree(i, j, deflect_nodes_alt_value_info[i], node_type_info, parent_info, task2leaf, node_value_info, task_cons, root_node_id)
                for j in a_taskInds[i] if task_selected[j]
            }
            for i in range(agent_num)
        }

        added_nodes_alt_value_info = {
            i: {
                j: info_get_move_val_tree[i][j][0]
                for j in a_taskInds[i] if task_selected[j]
            }
            for i in range(agent_num)
        }

        value_added_info = {
            i: {
                j: info_get_move_val_tree[i][j][1]
                for j in a_taskInds[i] if task_selected[j]
            }
            for i in range(agent_num)
        }

        sys_improvement_values_info = {
            i: {
                j: value_added_info[i][j] - value_lost_info[i]
                for j in a_taskInds[i] if task_selected[j]
            }
            for i in range(agent_num)
        }

        max_sys_improvement_values = {
            i: max(sys_improvement_values_info[i].items(), key=lambda x: x[1])
            for i in range(agent_num)
        }


    return (
        coalition_structure,
        node_value_info,
        iteration_count,
        re_assignment_count,
    )


def treeGNE2(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        task2leaf: list[int],
        leaf2task: dict[int, int],
        tasks: list[list[int]],
        agents: list[dict[int, float]],
        constraints,
        coalition_structure : list[list[int]] = [],
        selected_tasks : list[int] = None,
        eps=0, 
        gamma=1,
        root_node_id=0,
    ):
    
    """
    GreedyNE on an AND-OR tree.

    At each iteration, calculate the change in nodes values when an agent defects from its current task, and the change in nodes values when an agent moves to a new task.
    """
    re_assignment_count = 0
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)

    if selected_tasks is None:
        selected_tasks = list(range(len(tasks)))
        task_selected = [True for i in range(len(tasks))]

    else:    
        task_selected = [False for i in range(len(tasks))]
        for j in selected_tasks:
            task_selected[j] = True

    allocation_structure = [task_num for i in range(0, agent_num)]  # each indicate the current task that agent i is allocated to, if = N, means not allocated
    if coalition_structure is None or coalition_structure == []:
        coalition_structure = [[] for j in range(0, task_num)] + [list(range(0, agent_num))]  # default coalition structure, the last one is dummy coalition
        cur_con = [0 for j in range(0, agent_num)]
    else:

        if len(coalition_structure) < task_num:
            coalition_structure.append([])

        for j in range(0, task_num):
            if not task_selected[j]:
                coalition_structure[len(task_num)] += coalition_structure[j]
                coalition_structure[j] = []

        for j in range(0, task_num):
            for n_id in coalition_structure[j]:
                allocation_structure[n_id] = j

        cur_con = [
            agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            if j != task_num and task_selected[j]
            else 0
            for i, j in enumerate(allocation_structure)
        ]

    task_cons = [
        [
            agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            if j in a_taskInds[i] and task_selected[j]
            else float("-inf")
            for j in range(0, task_num)
        ] + [0]
        for i in range(0, agent_num)
    ]

    move_vals = {
        i : {
            j : task_cons[i][j] - cur_con[i]
            for j in a_taskInds[i] if task_selected[j]
        }
        for i in range(agent_num)
    }


    node_value_info = get_node_value_info(node_type_info, children_info, leaf2task, coalition_structure, root_node_id)

    info_get_cur_con = {
        i: get_cur_con_tree(i, node_type_info, children_info, parent_info, task2leaf, allocation_structure, node_value_info, cur_con, root_node_id)
        for i in range(agent_num)
    }

    deflect_nodes_alt_value_info = {
        i: info_get_cur_con[i][0]
        for i in range(agent_num)
    }

    value_lost_info = {
        i: info_get_cur_con[i][1]
        for i in range(agent_num)
    }

    info_get_move_val_tree = {
        i: {
            j: get_move_val_tree(i, j, deflect_nodes_alt_value_info[i], node_type_info, parent_info, task2leaf, node_value_info, task_cons, root_node_id)
            for j in a_taskInds[i] if task_selected[j]
        }
        for i in range(agent_num)
    }

    added_nodes_alt_value_info = {
        i: {
            j: info_get_move_val_tree[i][j][0]
            for j in a_taskInds[i] if task_selected[j]
        }
        for i in range(agent_num)
    }

    value_added_info = {
        i: {
            j: info_get_move_val_tree[i][j][1]
            for j in a_taskInds[i] if task_selected[j]
        }
        for i in range(agent_num)
    }

    sys_improvement_values_info = {
        i: {
            j: value_added_info[i][j] - value_lost_info[i]
            for j in a_taskInds[i] if task_selected[j]
        }
        for i in range(agent_num)
    }


    iteration_count = 0
    while True:
        iteration_count += 1
        feasible_choices_map = {
            i : [
                j for j in a_taskInds[i] if task_selected[j] and (
                    (sys_improvement_values_info[i][j] > 0) or (sys_improvement_values_info[i][j] == 0 and move_vals[i][j] > 0)
                )
            ]
            for i in range(0, agent_num)
        }
        feasible_choices = [(i, j) for i in feasible_choices_map for j in feasible_choices_map[i]]
        if len(feasible_choices) == 0:
            break  # reach NE solution
        # when eps = 1, it's Random, when eps = 0, it's Greedy
        if np.random.uniform() <= eps:
            # exploration: random allocation
            a_index, t_index = np.random.choice(feasible_choices)
        else:
            # exploitation: allocationelse based on reputation or efficiency
            a_index, t_index = max(feasible_choices, key=lambda x: (sys_improvement_values_info[x[0]][x[1]], move_vals[x[0]][x[1]]))
            

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
            for n_id in coalition_structure[t_index]:
                task_cons[n_id][t_index] = agent_contribution(agents, tasks, n_id, t_index, coalition_structure[t_index], constraints, gamma)
                cur_con[n_id] = task_cons[n_id][t_index]
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
                if (j in a_taskInds[i] and task_selected[j]) or j == task_num
                else float("-inf")
                for j in range(0, task_num + 1)
            ]

        ## update other agents w.r.t the affected tasks
        for t_ind in affected_t_indexes:
            for i in range(0, agent_num):
                if (i not in coalition_structure[t_ind]) and (t_ind in a_taskInds[i]):
                    task_cons[i][t_ind] = agent_contribution(agents, tasks, i, t_ind, coalition_structure[t_ind], constraints, gamma)
                    move_vals[i][t_ind] = task_cons[i][t_ind] - cur_con[i]

        # update node values
        for n_id, n_value in deflect_nodes_alt_value_info[a_index].items():
            node_value_info[n_id] = n_value

        for n_id, n_value in added_nodes_alt_value_info[a_index][t_index].items():
            node_value_info[n_id] = n_value


        info_get_cur_con = {
            i: get_cur_con_tree(i, node_type_info, children_info, parent_info, task2leaf, allocation_structure, node_value_info, cur_con)
            for i in range(agent_num)
        }

        deflect_nodes_alt_value_info = {
            i: info_get_cur_con[i][0]
            for i in range(agent_num)
        }

        value_lost_info = {
            i: info_get_cur_con[i][1]
            for i in range(agent_num)
        }

        info_get_move_val_tree = {
            i: {
                j: get_move_val_tree(i, j, deflect_nodes_alt_value_info[i], node_type_info, parent_info, task2leaf, node_value_info, task_cons, root_node_id)
                for j in a_taskInds[i] if task_selected[j]
            }
            for i in range(agent_num)
        }

        added_nodes_alt_value_info = {
            i: {
                j: info_get_move_val_tree[i][j][0]
                for j in a_taskInds[i] if task_selected[j]
            }
            for i in range(agent_num)
        }

        value_added_info = {
            i: {
                j: info_get_move_val_tree[i][j][1]
                for j in a_taskInds[i] if task_selected[j]
            }
            for i in range(agent_num)
        }

        sys_improvement_values_info = {
            i: {
                j: value_added_info[i][j] - value_lost_info[i]
                for j in a_taskInds[i] if task_selected[j]
            }
            for i in range(agent_num)
        }



    return (
        coalition_structure,
        node_value_info,
        iteration_count,
        re_assignment_count,
    )


def fastTreeGNE2(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        task2leaf: list[int],
        leaf2task: dict[int, int],
        tasks: list[list[int]],
        agents: list[dict[int, float]],
        constraints,
        coalition_structure : list[list[int]] = [],
        selected_tasks : list[int] = None,
        gamma=1,
        root_node_id=0,
    ):
    
    """
    GreedyNE on an AND-OR tree.

    At each iteration, calculate the change in nodes values when an agent defects from its current task, and the change in nodes values when an agent moves to a new task.

    Quickly and greedily find the best move by benchmarking (bounding) the added value for each node.
    """
    re_assignment_count = 0
    a_taskInds = constraints[0]
    agent_num = len(agents)
    task_num = len(tasks)

    if selected_tasks is None:
        selected_tasks = list(range(len(tasks)))
        task_selected = [True for i in range(len(tasks))]

    else:    
        task_selected = [False for i in range(len(tasks))]
        for j in selected_tasks:
            task_selected[j] = True

    allocation_structure = [task_num for i in range(0, agent_num)]  # each indicate the current task that agent i is allocated to, if = N, means not allocated
    if coalition_structure is None or coalition_structure == []:
        coalition_structure = [[] for j in range(0, task_num)] + [list(range(0, agent_num))]  # default coalition structure, the last one is dummy coalition
        cur_con = [0 for j in range(0, agent_num)]
    else:

        if len(coalition_structure) < task_num:
            coalition_structure.append([])

        for j in range(0, task_num):
            if not task_selected[j]:
                coalition_structure[len(task_num)] += coalition_structure[j]
                coalition_structure[j] = []

        for j in range(0, task_num):
            for n_id in coalition_structure[j]:
                allocation_structure[n_id] = j

        cur_con = [
            agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            if j != task_num and task_selected[j]
            else 0
            for i, j in enumerate(allocation_structure)
        ]

    task_cons = [
        [
            agent_contribution(agents, tasks, i, j, coalition_structure[j], constraints, gamma)
            if j in a_taskInds[i] and task_selected[j]
            else float("-inf")
            for j in range(0, task_num)
        ] + [0]
        for i in range(0, agent_num)
    ]

    move_vals = {
        i : {
            j : task_cons[i][j] - cur_con[i]
            for j in a_taskInds[i] if task_selected[j]
        }
        for i in range(agent_num)
    }


    node_value_info = get_node_value_info(node_type_info, children_info, leaf2task, coalition_structure, root_node_id)

    info_get_cur_con = {
        i: get_cur_con_tree(i, node_type_info, children_info, parent_info, task2leaf, allocation_structure, node_value_info, cur_con, root_node_id)
        for i in range(agent_num)
    }

    deflect_nodes_alt_value_info = {
        i: info_get_cur_con[i][0]
        for i in range(agent_num)
    }

    value_lost_info = {
        i: info_get_cur_con[i][1]
        for i in range(agent_num)
    }

    added_nodes_alt_value_info = {
        i: { }
        for i in range(agent_num)
    }


    best_move_agent, best_move_task = 0, 0
    best_improvement_value = float("-inf")
    best_move_move_val = float("-inf")
    for i in range(agent_num):
        best_added_value = best_improvement_value + value_lost_info[i]
        for j in a_taskInds[i]:
            if not task_selected[j]:
                continue
            added_nodes_alt_values, value_added = get_move_val_tree(i, j, deflect_nodes_alt_value_info[i], node_type_info, parent_info, task2leaf, node_value_info, task_cons, root_node_id, value_added_benchmark=best_added_value)
            move_val = move_vals[i][j]
            if value_added > best_added_value or (value_added == best_added_value and move_val >= best_move_move_val):
                added_nodes_alt_value_info[i][j] = added_nodes_alt_values
                best_added_value = value_added
                best_improvement_value = value_added - value_lost_info[i]
                best_move_agent, best_move_task = i, j


    iteration_count = 0
    while True:
        iteration_count += 1
        if best_improvement_value < 0:
            break
        elif best_improvement_value == 0 and best_move_move_val < 0:
            break

        a_index, t_index = best_move_agent, best_move_task            

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
            for n_id in coalition_structure[t_index]:
                task_cons[n_id][t_index] = agent_contribution(agents, tasks, n_id, t_index, coalition_structure[t_index], constraints, gamma)
                cur_con[n_id] = task_cons[n_id][t_index]
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
                if (j in a_taskInds[i] and task_selected[j]) or j == task_num
                else float("-inf")
                for j in range(0, task_num + 1)
            ]

        ## update other agents w.r.t the affected tasks
        for t_ind in affected_t_indexes:
            for i in range(0, agent_num):
                if (i not in coalition_structure[t_ind]) and (t_ind in a_taskInds[i]):
                    task_cons[i][t_ind] = agent_contribution(agents, tasks, i, t_ind, coalition_structure[t_ind], constraints, gamma)
                    move_vals[i][t_ind] = task_cons[i][t_ind] - cur_con[i]

        # update node values
        for n_id, n_value in deflect_nodes_alt_value_info[a_index].items():
            node_value_info[n_id] = n_value

        for n_id, n_value in added_nodes_alt_value_info[a_index][t_index].items():
            node_value_info[n_id] = n_value


        info_get_cur_con = {
            i: get_cur_con_tree(i, node_type_info, children_info, parent_info, task2leaf, allocation_structure, node_value_info, cur_con)
            for i in range(agent_num)
        }

        deflect_nodes_alt_value_info = {
            i: info_get_cur_con[i][0]
            for i in range(agent_num)
        }

        value_lost_info = {
            i: info_get_cur_con[i][1]
            for i in range(agent_num)
        }

        added_nodes_alt_value_info = {
            i: { }
            for i in range(agent_num)
        }

        best_move_agent, best_move_task = 0, 0
        best_improvement_value = float("-inf")
        best_move_move_val = float("-inf")
        for i in range(agent_num):
            best_added_value = best_improvement_value + value_lost_info[i]
            for j in a_taskInds[i]:
                if not task_selected[j]:
                    continue
                added_nodes_alt_values, value_added = get_move_val_tree(i, j, deflect_nodes_alt_value_info[i], node_type_info, parent_info, task2leaf, node_value_info, task_cons, root_node_id, value_added_benchmark=best_added_value)
                move_val = move_vals[i][j]
                if value_added > best_added_value or (value_added == best_added_value and move_val >= best_move_move_val):
                    added_nodes_alt_value_info[i][j] = added_nodes_alt_values
                    best_added_value = value_added
                    best_improvement_value = value_added - value_lost_info[i]
                    best_move_agent, best_move_task = i, j


    return (
        coalition_structure,
        node_value_info,
        iteration_count,
        re_assignment_count,
    )





def update_leaves_info(
        updated_node : int, 
        leaves_info : dict[int, list[int]],
        children_info : dict[int, list[int]],
        parent_info : dict[int, int],
        node_type_info : dict[int, NodeType]
    ):
    """
    Update the leaves_info for all ancestors of the updated_node.

    updated_node: the node that has just been updated in leaves_info
    """
    current_node = updated_node
    while current_node != 0:
        parent_node = parent_info[current_node]
        if node_type_info[parent_node] == NodeType.AND:
            leaves_info[parent_node] = sum([leaves_info[child_id] for child_id in children_info[parent_node]], [])
        else:
            leaves_info[parent_node] = leaves_info[children_info[parent_node][0]].copy()
        current_node = parent_node

    return leaves_info


def AO_star(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        reward_function: dict[int, float],
        root_node_id=0,
    ):

    def AOS_helper(node_id: int):

        node_type = node_type_info[node_id]
        
        if node_type == NodeType.LEAF:
            return reward_function[node_id], [node_id]

        total_reward = 0 if node_type == NodeType.AND else float('-inf')

        best_solution = []

        if node_type == NodeType.AND:

            for child_id in children_info[node_id]:

                # solution_path_leaves_info[node_id].append(child_id)
                # update_leaves_info(node_id, solution_path_leaves_info, solution_path_children_info, parent_info, node_type_info)
                
                child_reward, child_solution = AOS_helper(child_id)

                total_reward += child_reward
                best_solution += child_solution
                
        else:
            for child_id in children_info[node_id]:
                
                child_reward, child_solution = AOS_helper(child_id)

                if child_reward > total_reward:
                    total_reward = child_reward
                    best_solution = child_solution
                    
        # expanded.append(node_id)
        return total_reward, best_solution
    
    total_reward, best_leafs_solution = AOS_helper(root_node_id)
    
    return total_reward, best_leafs_solution


def treeGNE_extra(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        task2leaf: list[int],
        leaf2task: dict[int, int],
        tasks: list[list[int]],
        agents: list[dict[int, float]],
        constraints,
        coalition_structure : list[list[int]] = [],
        eps=0, 
        gamma=1,
        root_node_id=0,
    ):
    
    """
    GreedyNE on an AND-OR tree.

    Expansion of treeNE where we use aGreedyNE to perform local search after each treeNE iteration.
    """
    total_iteration_count = 0
    total_re_assignment_count = 0
    sys_reward_final = 0
    while True:
        coalition_structure, node_value_info, iteration_count, re_assignment_count = treeGNE(
            node_type_info=node_type_info, 
            children_info=children_info, 
            parent_info=parent_info, 
            task2leaf=task2leaf, 
            leaf2task=leaf2task, 
            tasks=tasks, 
            agents=agents, 
            constraints=constraints, 
            coalition_structure=coalition_structure, 
            eps=eps, 
            gamma=gamma, 
            root_node_id=root_node_id
        )

        sys_reward_current = node_value_info[root_node_id]
        sys_reward_final = sys_reward_current

        total_iteration_count += iteration_count
        total_re_assignment_count += re_assignment_count

        tasks_reward_info = {
            n_id: node_value_info[n_id]
            for n_id in leaf2task
        }

        sys_reward_aostar, current_tasks_solution = AO_star(node_type_info, children_info, parent_info, tasks_reward_info)
        
        selected_tasks = sorted([leaf2task[n_id] for n_id in current_tasks_solution])

        if sys_reward_current != sys_reward_aostar:
            assert False, "AO_star and treeGNE give different results"


        coalition_structure, sys_reward_new, iteration_count, re_assignment_count = aGreedyNE(
            agents=agents,
            tasks=tasks,
            constraints=constraints,
            coalition_structure=coalition_structure,
            selected_tasks=selected_tasks,
            eps=eps,
            gamma=gamma,
        )

        total_iteration_count += iteration_count
        total_re_assignment_count += re_assignment_count

        if sys_reward_new <= sys_reward_current:
            break

        else:
            sys_reward_final = sys_reward_new


    return (
        coalition_structure,
        sys_reward_final,
        total_iteration_count,
        total_re_assignment_count,
    )

