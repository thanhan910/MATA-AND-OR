from .node_type import NodeType
from .rewards import task_reward, sys_rewards_tree_tasks
from .GreedyNE import aGreedyNE

def ao_search(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]],
        reward_function: dict[int, float],
        root_node_id=0,
    ):
    
    
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
                
        else:
            for child_id in children_info[node_id]:
                
                child_reward, child_solution = ao_helper(child_id)

                if child_reward > total_reward:
                    total_reward = child_reward
                    best_solution = child_solution
                    
        # expanded.append(node_id)
        return total_reward, best_solution
    
    total_reward, best_leafs_solution = ao_helper(root_node_id)
    
    return total_reward, best_leafs_solution


def simpleGNE(
        node_type_info: dict[int, NodeType],
        children_info: dict[int, list[int]],
        leaf2task: dict[int, int],
        agents : list[list[float]], 
        tasks : list[list[int]],
        constraints : tuple[list[list[int]], list[list[int]]],
        coalition_structure : list[list[int]] = [],
        selected_tasks : list[int] = None,
        root_node_id=0,
        eps=0, 
        gamma=1
    ):
    prev_sys_reward = 0
    true_sys_reward = float('inf')
    iteration_count_1 = 0
    re_assignment_count_1 = 0
    iteration_count_2 = 0
    re_assignment_count_2 = 0
    total_loop_count = 0
    while True:

        total_loop_count += 1

        coalition_structure, system_reward, iteration_count, re_assignment_count = aGreedyNE(
            agents=agents,
            tasks=tasks,
            constraints=constraints,
            coalition_structure=coalition_structure,
            selected_tasks=selected_tasks,
            eps=eps,
            gamma=gamma
        )

        iteration_count_1 += iteration_count
        re_assignment_count_1 += re_assignment_count

        reward_function = {
            leaf_id: task_reward(tasks[leaf2task[leaf_id]], [agents[i] for i in coalition_structure[leaf2task[leaf_id]]], gamma)
            for leaf_id in leaf2task
        }
        
        true_sys_reward, best_leafs_solution = ao_search(node_type_info, children_info, reward_function, root_node_id)

        new_selected_tasks = [leaf2task[leaf_id] for leaf_id in best_leafs_solution]

        coalition_structure, true_sys_reward, iteration_count, re_assignment_count = aGreedyNE(
            agents=agents,
            tasks=tasks,
            constraints=constraints,
            coalition_structure=coalition_structure,
            selected_tasks=new_selected_tasks,
            eps=eps,
            gamma=gamma
        )

        iteration_count_2 += iteration_count
        re_assignment_count_2 += re_assignment_count

        if true_sys_reward <= prev_sys_reward:
            break

        prev_sys_reward = true_sys_reward

    return (
        coalition_structure,
        true_sys_reward,
        iteration_count_1,
        re_assignment_count_1,
        iteration_count_2,
        re_assignment_count_2,
        total_loop_count
    )