from .node_type import NodeType
from .GreedyNE import adGreedyNE
from .upper_bound import upper_bound_subsytem
from .rewards import task_reward, sys_rewards_tasks



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

