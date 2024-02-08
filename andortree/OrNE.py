from .node_type import NodeType
from .GreedyNE import adGreedyNE, alGreedyNE
from .rewards import task_reward, sys_rewards_tasks
from .tree_utils import traverse_tree

import numpy as np


def ao_search(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]],
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





def get_upperbound_node_descendants(
        query_nodeId : int,
        agents_group : list[int],
        ubcv_info : dict[int, np.ndarray],
        children_info : dict[int, list[int]],
        node_type_info : dict[int, NodeType],
        capabilities : list[int], 
        agents : list[list[float]],
        nodes_constraints : tuple[list[list[int]], dict[int, list[int]]],
    ):
    """
    Calculate the upper bound of the reward (utility) at each descendant of the queried node in the AND-OR goal tree, including the queried node itself.
    
    Calculate the upper bound of the reward (utility) at each node of the AND-OR goal tree.

    Refine the upper bound by taking the minimum of the upper bound calculated from the children nodes, and the upper bound calculated from the current node.

    """

    agent_selected = {i: False for i in range(len(agents))}
    for i in agents_group:
        agent_selected[i] = True

    def _upperbound_node(node_id):
        """
        Calculate the upper bound of the system reward, i.e. at the root of the AND-OR goal tree.
        """
        nodes_agents = nodes_constraints[1]

        caps_ranked = [sorted([agents[i][c] for i in nodes_agents[node_id] if agent_selected[i]], reverse=True) for c in capabilities]

        cap_req_num = ubcv_info[node_id]
        
        return sum([sum(caps_ranked[c][:int(cap_req_num[c])]) for c in capabilities])

    descendant_nodes = list(traverse_tree(children_info, root_node_id=query_nodeId))
    
    nodes_upper_bound = { node_id : _upperbound_node(node_id) for node_id in descendant_nodes }

    nodes_upper_bound_min = { node_id : 0 for node_id in descendant_nodes }

    def _min_upper_bound(node_id : int):
        node_type = node_type_info[node_id]

        if node_type == NodeType.LEAF:
            nodes_upper_bound_min[node_id] = nodes_upper_bound[node_id]

        elif node_type == NodeType.OR:
            nodes_upper_bound_min[node_id] = max(_min_upper_bound(child_id) for child_id in children_info[node_id])

        elif node_type == NodeType.AND:
            nodes_upper_bound_min[node_id] = sum(_min_upper_bound(child_id) for child_id in children_info[node_id])

        else:
            raise Exception("Unsupported node type")

        nodes_upper_bound_min[node_id] = min(nodes_upper_bound[node_id], nodes_upper_bound_min[node_id])
        return nodes_upper_bound_min[node_id]
        
    _min_upper_bound(query_nodeId)

    return nodes_upper_bound_min



def BnBOrNE(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        ubcv_info : dict[int, np.ndarray],
        leaf2task: dict[int, int],
        task2leaf: dict[int, int],
        leaves_list_info: dict[int, list[int]],
        capabilities: list[int],
        tasks: list[list[int]],
        agents: list[dict[int, float]],
        constraints,
        nodes_constraints : tuple[list[list[int]], dict[int, list[int]]],
        coalition_structure : dict[int, list[int]] = {},
        eps=0, 
        gamma=1,
        root_node_id=0,
        use_branch_and_bound=True,
    ):
    
    task_num = len(tasks)
    agent_num = len(agents)

    def _GreedyNE(original_allocation_structure, selected_tasks, selected_agents):
        return alGreedyNE(
            agents=agents, 
            tasks=tasks, 
            constraints=constraints, 
            original_allocation_structure=original_allocation_structure,
            selected_tasks=selected_tasks, 
            selected_agents=selected_agents, 
            eps=eps, 
            gamma=gamma
        )
    
    def _ao_search(reward_function, root_node_id):
        return ao_search(
            node_type_info=node_type_info,
            children_info=children_info,
            reward_function=reward_function,
            root_node_id=root_node_id,
        )
    

    def _get_upper_bound(node_id, agents_group):
        return get_upperbound_node_descendants(
            query_nodeId=node_id,
            agents_group=agents_group,
            ubcv_info=ubcv_info,
            children_info=children_info,
            node_type_info=node_type_info,
            capabilities=capabilities,
            agents=agents,
            nodes_constraints=nodes_constraints,
        )

    
    if coalition_structure is None or coalition_structure == {}:
        coalition_structure = {j: [] for j in range(0, len(tasks))}
        coalition_structure[len(tasks)] = list(range(0, len(agents)))  # default coalition structure, the last one is dummy coalition

    allocation_structure_global = {i: task_num for i in range(0, agent_num)}
    for j in coalition_structure:
        for i in coalition_structure[j]:
            allocation_structure_global[i] = j

    t_agents = constraints[1]


    def aos_helper(node_id, agents_group, allocation_structure_0 = None):

        total_iterations_count = 0
        total_reassignment_count = 0

        node_type = node_type_info[node_id]

        if node_type == NodeType.LEAF:
            task_id = leaf2task[node_id]
            new_allocation_structure = { i: task_id if i in t_agents[task_id] else task_num for i in agents_group }
            reward_value = task_reward(tasks[task_id], [agents[i] for i in agents_group if i in t_agents[task_id]], gamma)
            return new_allocation_structure, reward_value, total_iterations_count, total_reassignment_count


        descendant_leaves = leaves_list_info[node_id]
        descendant_tasks = [leaf2task[task_id] for task_id in descendant_leaves]

        # Initialization phase

        # Initialize allocation structure
        not_initialized = False
        if allocation_structure_0 is None:
            not_initialized = True
        else:
            not_initialized = True
            for i in agents_group:
                if i in allocation_structure_0 and allocation_structure_0[i] in descendant_tasks:
                    not_initialized = False
                    break

        if not_initialized:
            # Perform GreedyNE on the entire system, not considering the tree structure
            coalition_structure_1, allocation_structure_1, _, iter_count_1, reassign_1 = _GreedyNE(
                original_allocation_structure=None,
                selected_tasks=descendant_tasks,
                selected_agents=agents_group,
            )
        else:
            allocation_structure_1 = allocation_structure_0
            coalition_structure_1 = { j: [] for j in range(0, task_num + 1) }
            for i in allocation_structure_1:
                coalition_structure_1[allocation_structure_1[i]].append(i)
                iter_count_1 = 0
                reassign_1 = 0
                    
        # Initialization phase

        reward_function = {
            task2leaf[task_id]: task_reward(tasks[task_id], [agents[i] for i in coalition_structure_1[task_id]], gamma)
            for task_id in descendant_tasks
        }

        true_sys_reward_1, best_leafs_solution, st_children_info = _ao_search(
            reward_function=reward_function,
            root_node_id=node_id,
        )

        best_tasks_solution = [leaf2task[leaf_id] for leaf_id in best_leafs_solution]

        coalition_structure_2, allocation_structure_2, system_reward_2, iter_count_2, reassign_2  = _GreedyNE(
            original_allocation_structure=allocation_structure_1,
            selected_tasks=best_tasks_solution,
            selected_agents=agents_group,
        )

        total_iterations_count += iter_count_1 + iter_count_2
        total_reassignment_count += reassign_1 + reassign_2

        # Update allocation_solution
        allocation_solution = {}
        for i in allocation_structure_2:
            allocation_solution[i] = allocation_structure_2[i]

        if node_type == NodeType.AND:
            total_reward = 0
            for child_id in children_info[node_id]:
                child_tasks_descendants = [leaf2task[leaf_id] for leaf_id in leaves_list_info[child_id]]
                child_agents_group = sum([coalition_structure_2.get(task_id, []) for task_id in child_tasks_descendants], [])
                child_allocation_solution, child_system_reward, child_iter_count, child_reassign_count = aos_helper(child_id, child_agents_group, allocation_structure_2)
                # Update allocation_solution based on child_allocation_solution
                for j in child_allocation_solution:
                    allocation_solution[j] = child_allocation_solution[j]
                
                total_reward += child_system_reward

                total_iterations_count += child_iter_count
                total_reassignment_count += child_reassign_count
                
            return allocation_solution, total_reward, total_iterations_count, total_reassignment_count
        
        else: # if node_type == NodeType.OR:
            current_child_id = st_children_info[node_id][0]
            total_reward = system_reward_2
            final_allocation_solution = allocation_structure_2.copy()
            for child_id in children_info[node_id]:
                if child_id == current_child_id:
                    continue
                # Bound pruning
                if use_branch_and_bound:
                    nodes_upper_bound_min = _get_upper_bound(child_id, agents_group)
                    reward_upper_bound = nodes_upper_bound_min[child_id]
                    if reward_upper_bound <= total_reward:
                        continue
                # Branch to child_id
                child_allocation_solution, child_system_reward, child_iter_count, child_reassign_count = aos_helper(child_id, agents_group, None)

                total_iterations_count += child_iter_count
                total_reassignment_count += child_reassign_count

                if child_system_reward > total_reward:
                    total_reward = child_system_reward
                    final_allocation_solution = child_allocation_solution

            # Update allocation_solution based on final_allocation_solution
            for j in final_allocation_solution:
                allocation_solution[j] = final_allocation_solution[j]

            return allocation_solution, total_reward, total_iterations_count, total_reassignment_count


    return aos_helper(root_node_id, list(range(0, len(agents))), allocation_structure_global)