from .node_type import NodeType
from .GreedyNE import aGreedyNE, adGreedyNE
from .upper_bound import upper_bound_subsytem
from .rewards import task_reward, sys_rewards_tasks

import numpy as np


def get_leaves(root_id: int, children_info: dict[int, list[int]]):
    leaves = []
    stack = [root_id]
    while len(stack) > 0:
        current_node = stack.pop()
        if current_node not in children_info or len(children_info[current_node]) == 0:
            leaves.append(current_node)
        else:
            stack += children_info[current_node]
    return leaves


def update_leaves_info(updated_node, leaves_info, children_info, parent_info, node_type_info):
    """
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


def ao_search(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        reward_function: dict[int, float],
        root_node_id=0,
    ):
    
    visited = {}
    # expanded = []
    solution_path_children_info : dict[int, list[int]] = {}
    solution_path_leaves_info : dict[int, list[int]] = {}

    def aos_helper(node_id: int):

        if node_id not in visited:
            visited[node_id] = True
            solution_path_children_info[node_id] = []
            solution_path_leaves_info[node_id] = []

        node_type = node_type_info[node_id]
        
        if node_type == NodeType.LEAF:
            # expanded.append(node_id)
            solution_path_leaves_info[node_id] = [node_id]
            # Update solution_path_leaves_info for all ancestors
            update_leaves_info(node_id, solution_path_leaves_info, solution_path_children_info, parent_info, node_type_info)
            return reward_function[node_id], [node_id]

        total_reward = 0 if node_type == NodeType.AND else float('-inf')

        best_solution = []

        if node_type == NodeType.AND:

            for child_id in children_info[node_id]:

                solution_path_children_info[node_id].append(child_id)
                # solution_path_leaves_info[node_id].append(child_id)
                # update_leaves_info(node_id, solution_path_leaves_info, solution_path_children_info, parent_info, node_type_info)
                
                child_reward, child_solution = aos_helper(child_id)

                total_reward += child_reward
                best_solution += child_solution
                
        else:
            for child_id in children_info[node_id]:

                if solution_path_children_info[node_id] == []:
                    solution_path_children_info[node_id] = [child_id]
                
                child_reward, child_solution = aos_helper(child_id)

                if child_reward > total_reward:
                    total_reward = child_reward
                    best_solution = child_solution
                    solution_path_children_info[node_id] = [child_id]
                    solution_path_leaves_info[node_id] = child_solution
                    # Update solution_path_leaves_info for all ancestors
                    update_leaves_info(node_id, solution_path_leaves_info, solution_path_children_info, parent_info, node_type_info)
                    
        # expanded.append(node_id)
        return total_reward, best_solution
    
    total_reward, best_leafs_solution = aos_helper(root_node_id)
    
    return total_reward, best_leafs_solution, solution_path_children_info, solution_path_leaves_info


def get_leaves(root_id: int, children_info: dict[int, list[int]]):
    leaves = []
    stack = [root_id]
    while len(stack) > 0:
        current_node = stack.pop()
        if current_node not in children_info or len(children_info[current_node]) == 0:
            leaves.append(current_node)
        else:
            stack += children_info[current_node]
    return leaves



def aos_tree(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        reward_function: dict[int, float],
        root_node_id=0,
    ):
    
    visited = {}
    # expanded = []
    st_children_info : dict[int, list[int]] = {
        node_id: [] if node_type_info[node_id] == NodeType.OR else children_list
        for node_id, children_list in children_info.items()
    }

    def aos_helper(node_id: int):

        if node_id not in visited:
            visited[node_id] = True            

        node_type = node_type_info[node_id]
        
        if node_type == NodeType.LEAF:
            return reward_function[node_id], [node_id]

        total_reward = 0 if node_type == NodeType.AND else float('-inf')

        best_solution = []

        if node_type == NodeType.AND:

            for child_id in children_info[node_id]:

                child_reward, child_solution = aos_helper(child_id)

                total_reward += child_reward
                best_solution += child_solution
                
        else:
            for child_id in children_info[node_id]:

                if st_children_info[node_id] == []:
                    st_children_info[node_id] = [child_id]
                
                child_reward, child_solution = aos_helper(child_id)

                if child_reward > total_reward:
                    total_reward = child_reward
                    best_solution = child_solution
                    st_children_info[node_id] = [child_id]

                    
        # expanded.append(node_id)
        return total_reward, best_solution
    
    total_reward, best_leafs_solution = aos_helper(root_node_id)
    
    return total_reward, best_leafs_solution, st_children_info




def simplify_tree_info(children_info : dict[int, list[int]], leaves_info : dict[int, list[int]] = {}, root_node_id: int=0):
    new_children_info = {}
    new_leaves_info = {}
    stack = [0]
    while len(stack) > 0:
        current_node = stack.pop()
        if current_node in children_info and len(children_info[current_node]) > 0:
            new_children_info[current_node] = children_info[current_node].copy()
        if current_node in leaves_info and len(leaves_info[current_node]) > 0:
            new_leaves_info[current_node] = leaves_info[current_node].copy()
        stack += children_info[current_node]
    return new_children_info, new_leaves_info


def AOsearchGNE(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        nodes_upper_bound: dict[int, float],
        nodes_upper_bound_min: dict[int, float],
        leaf2task: dict[int, int],
        capabilities: list[int],
        tasks: list[list[int]],
        agents: list[dict[int, float]],
        constraints,
        coalition_structure : list[list[int]] = [],
        eps=0, 
        gamma=1,
        root_node_id=0,
        # reward_function: dict[int, float],
    ):

    agent_num = len(agents)
    task_num = len(tasks)

    if coalition_structure is None or coalition_structure == []:
        coalition_structure = [[] for j in range(0, task_num)] + [list(range(0, agent_num))]  # default coalition structure, the last one is dummy coalition

    st_children_info : dict[int, list[int]] = {
        node_id: [] if node_type_info[node_id] == NodeType.OR else children_list
        for node_id, children_list in children_info.items()
    }

    def aos_helper(node_id: int, best_coalition_structure : list[list[int]], best_sys_reward : float):

        node_type = node_type_info[node_id]
        
        if node_type == NodeType.LEAF:
            
            current_tasks = [leaf2task[n_id] for n_id in get_leaves(root_node_id, st_children_info) if node_type_info[n_id] == NodeType.LEAF]
            
            n_coalition_structure, n_system_reward, n_iteration_count, n_re_assignment_count = aGreedyNE(agents=agents, tasks=tasks, constraints=constraints, coalition_structure=best_coalition_structure, selected_tasks=current_tasks, eps=eps, gamma=gamma)
            
            if n_system_reward > best_sys_reward:
                best_sys_reward = n_system_reward
                best_coalition_structure = n_coalition_structure
            
            return best_coalition_structure, best_sys_reward, n_iteration_count, n_re_assignment_count
        
        t_iteration_count, t_re_assignment_count = 0, 0

        if node_type == NodeType.AND:

            for child_id in children_info[node_id]:

                child_coalition_structure, child_reward, c_iteration_count, c_reassignment_count = aos_helper(child_id, best_coalition_structure=best_coalition_structure, best_sys_reward=best_sys_reward)

                t_iteration_count += c_iteration_count
                t_re_assignment_count += c_reassignment_count

                if child_reward > best_sys_reward:
                    best_sys_reward = child_reward
                    best_coalition_structure = child_coalition_structure
                
        else:
            for child_id in children_info[node_id]:

                if st_children_info[node_id] == []:
                    st_children_info[node_id] = [child_id]
                    current_child = child_id
                else:
                    current_child = st_children_info[node_id][0]

                st_children_info[node_id] = [child_id]
                sys_reward_upper_bound = upper_bound_subsytem(
                    selected_nodes=get_leaves(root_node_id, st_children_info),
                    nodes_upper_bound=nodes_upper_bound,
                    nodes_upper_bound_min=nodes_upper_bound_min,
                    node_type_info=node_type_info,
                    leaf2task=leaf2task,
                    capabilities=capabilities,
                    tasks=tasks,
                    agents=agents,
                    constraints=constraints,
                )
                if sys_reward_upper_bound <= best_sys_reward:
                    continue

                st_children_info[node_id] = [current_child]
                
                child_coalition_structure, child_reward, c_iteration_count, c_reassignment_count = aos_helper(child_id, best_coalition_structure=best_coalition_structure, best_sys_reward=best_sys_reward)

                t_iteration_count += c_iteration_count
                t_re_assignment_count += c_reassignment_count

                if child_reward > best_sys_reward:
                    best_sys_reward = child_reward
                    best_coalition_structure = child_coalition_structure
                    st_children_info[node_id] = [child_id]

                    
        # expanded.append(node_id)
        return best_coalition_structure, best_sys_reward, t_iteration_count, t_re_assignment_count
    
    best_coalition_structure, best_sys_reward = coalition_structure, sys_rewards_tasks(tasks, agents, coalition_structure, gamma)

    loop_count = 0
    total_iteration_count = 0
    total_re_assignment_count = 0

    while True:
        loop_count += 1
        new_coalition_structure, new_sys_reward, new_iter_count, new_re_assignment_count = aos_helper(root_node_id, best_coalition_structure=best_coalition_structure, best_sys_reward=best_sys_reward)
        total_iteration_count += new_iter_count
        total_re_assignment_count += new_re_assignment_count
        if new_sys_reward > best_sys_reward:
            best_coalition_structure = new_coalition_structure
            best_sys_reward = new_sys_reward
        else:
            break

    return best_coalition_structure, best_sys_reward, total_iteration_count, total_re_assignment_count, loop_count


def ao_search(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]],
        reward_function: dict[int, float],
        root_node_id=0,
    ):
    
    st_children_info : dict[int, list[int]] = {
        node_id: [] if node_type_info[node_id] == NodeType.OR else children_list
        for node_id, children_list in children_info.items()
    }

    def ao_helper(node_id: int):

        node_type = node_type_info[node_id]
        
        if node_type == NodeType.LEAF:
            return reward_function[node_id], [node_id]

        total_reward = 0 if node_type == NodeType.AND else float('-inf')

        best_solution = []

        if node_type == NodeType.AND:

            for child_id in children_info[node_id]:

                child_reward, child_solution = ao_helper(child_id)

                total_reward += child_reward
                best_solution += child_solution
                
        else: # OR node
            for child_id in children_info[node_id]:
                
                child_reward, child_solution = ao_helper(child_id)

                if child_reward > total_reward:
                    total_reward = child_reward
                    best_solution = child_solution
                    st_children_info[node_id] = [child_id]
                    
        # expanded.append(node_id)
        return total_reward, best_solution
    
    total_reward, best_leafs_solution = ao_helper(root_node_id)
    
    return total_reward, best_leafs_solution, st_children_info


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



def OrNE(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        parent_info: dict[int, int],
        nodes_upper_bound: dict[int, float],
        nodes_upper_bound_min: dict[int, float],
        leaf2task: dict[int, int],
        task2leaf: dict[int, int],
        leaves_info: dict[int, list[int]],
        capabilities: list[int],
        tasks: list[list[int]],
        agents: list[dict[int, float]],
        constraints,
        coalition_structure : dict[int, list[int]] = {},
        eps=0, 
        gamma=1,
        root_node_id=0,
    ):

    myGreedyNE = lambda original_coalition_structure, selected_tasks, selected_agents: adGreedyNE(
        agents=agents, 
        tasks=tasks, 
        constraints=constraints, 
        original_coalition_structure=original_coalition_structure, 
        selected_tasks=selected_tasks, 
        selected_agents=selected_agents, 
        eps=eps, 
        gamma=gamma
    )

    my_ao_search = lambda reward_function, root_node_id: ao_search(
        node_type_info=node_type_info,
        children_info=children_info,
        reward_function=reward_function,
        root_node_id=root_node_id,
    )

    if coalition_structure is None or coalition_structure == {}:
        coalition_structure = {j: [] for j in range(0, len(tasks))}
        coalition_structure[len(tasks)] = list(range(0, len(agents)))  # default coalition structure, the last one is dummy coalition


    def aos_helper(node_id, agents_group, coalition_structure_0 = None):

        descendant_leaves = leaves_info[node_id]
        descendant_tasks = [task2leaf[task_id] for task_id in descendant_leaves]

        # Initialize coalition structure
        if coalition_structure_0 is None:
            # Perform GreedyNE on the entire system, not considering the tree structure
            coalition_structure_1, _, _, _ = myGreedyNE(
                original_allocation_structure=coalition_structure_0,
                selected_tasks=descendant_tasks,
                selected_agents=agents_group,
            )
        else:
            coalition_structure_1 = coalition_structure_0

        reward_function = {
            task2leaf[task_id]: task_reward(tasks[task_id], [agents[i] for i in coalition_structure_1[task_id]], gamma)
            for task_id in descendant_tasks
        }

        true_sys_reward, best_leafs_solution, st_children_info = my_ao_search(
            reward_function=reward_function,
            root_node_id=node_id,
        )

        best_tasks_solution = [task2leaf[task_id] for task_id in best_leafs_solution]

        coalition_structure_2, system_reward_2, _, _ = myGreedyNE(
            original_coalition_structure=coalition_structure_1,
            selected_tasks=best_tasks_solution,
            selected_agents=agents_group,
        )

        # Update coalition_structure based on coalition_structure_2
        for j in coalition_structure_2:
            coalition_structure[j] = coalition_structure_2[j]

        node_type = node_type_info[node_id]

        if node_type == NodeType.LEAF:
            return coalition_structure, system_reward_2
        

        elif node_type == NodeType.AND:
            total_reward = 0
            for child_id in children_info[node_id]:
                child_tasks_descendants = [leaf2task[leaf_id] for leaf_id in leaves_info[child_id]]
                child_agents_group = sum([coalition_structure_2[task_id] for task_id in child_tasks_descendants], [])
                child_coalition_solution, child_system_reward = aos_helper(child_id, child_agents_group, coalition_structure)
                # Update coalition_structure based on child_coalition_solution
                for j in child_coalition_solution:
                    coalition_structure[j] = child_coalition_solution[j]
                
                total_reward += child_system_reward
            return coalition_structure, total_reward
        
        else: # if node_type == NodeType.OR:
            final_coalition_solution = {}
            current_child_id = st_children_info[node_id][0]
            total_reward = system_reward_2
            for child_id in children_info[node_id]:
                if child_id == current_child_id:
                    continue
                # TODO: Upper bound
                child_coalition_solution, child_system_reward = aos_helper(child_id, agents_group, None)
                if child_system_reward > total_reward:
                    total_reward = child_system_reward
                    final_coalition_solution = child_coalition_solution

            # Update allocation_solution based on final_allocation_solution
            # Update coalition_structure based on final_coalition_solution
            for j in final_coalition_solution:
                coalition_structure[j] = final_coalition_solution[j]

            return coalition_structure, total_reward

