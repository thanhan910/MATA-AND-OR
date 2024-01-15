from .node_type import NodeType
from .GreedyNE import aGreedyNE
from .upper_bound import upper_bound_subsytem
from .rewards import task_reward, sys_rewards_tasks


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


def AOGreedyNE(
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

    tasks_reward_info = {
        j: task_reward(tasks[j], [agents[i] for i in coalition_structure[j]], gamma)
        for j in range(0, task_num)
    }

    system_reward, current_tasks_solution, solution_path_children_info, solution_path_leaves_info = ao_search(node_type_info, children_info, parent_info, tasks_reward_info)

    coalition_structure, system_reward, iteration_count, re_assignment_count = aGreedyNE(agents=agents, tasks=tasks, constraints=constraints, coalition_structure=coalition_structure, selected_tasks=current_tasks_solution, eps=eps, gamma=gamma)
    
    # Get all OR nodes in solution_path_children_info, starting from the root node
    or_nodes = [root_node_id]
    stack = [root_node_id]
    while len(stack) > 0:
        current_node = stack.pop()
        if current_node not in children_info or len(children_info[current_node]) == 0:
            continue
        else:
            if node_type_info[current_node] == NodeType.OR:
                or_nodes.append(current_node)
            stack += children_info[current_node]
