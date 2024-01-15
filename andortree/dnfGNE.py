from .node_type import NodeType
from .upper_bound import upper_bound_subsytem
from .GreedyNE import aGreedyNE


def traverse_and_or_tree(node_type_info: dict, children_info: dict, root_node_id: int = 0):
    """
    Traverses an AND-OR goal tree and yields all possible solutions (subsets of leaves that can be completed to fulfill an AND-OR goal tree).
    """

    # skipped_nodes = set()

    def traverse_helper(node_id: int) -> list:
        
        # print("NODE: ", node_id)

        # if node_type_info[node_id] != NodeType.OR:
        #     if random.random() < 0.1 and depth_info[node_id] > 2:
        #         skipped_nodes.add(node_id)
        #         return

        # If the node is a leaf node (no children)
        if node_type_info[node_id] == NodeType.LEAF:
            # print("YIELD: ", node_id)
            yield [node_id], [node_id]
            return

        # For AND nodes, need to combine children subsets
        if node_type_info[node_id] == NodeType.AND:
            
            leaves_subsets = [list(traverse_helper(child)) for child in children_info[node_id]]

            stack = [([], [node_id], 0)]
            
            while stack:
                combination, combination_path, index = stack.pop()
                if index >= len(children_info[node_id]):
                    yield combination, combination_path
                else:
                    for item, path in leaves_subsets[index]:
                    # for item, path in traverse_helper(children_info[node_id][index]):
                        stack.append((combination + item, combination_path + path, index + 1))

        
        # For OR nodes, simply yield from each child
        elif node_type_info[node_id] == NodeType.OR:

            # num_child = len(children_info[node_id])
            
            for child in children_info[node_id]:
                # if random.random() < 0.1 and depth_info[child] > 2 and num_child > 1:
                #     num_child -= 1
                #     skipped_nodes.add(child)
                #     continue
                for item, path in traverse_helper(child):
                    yield item, [node_id] + path

    yield from traverse_helper(root_node_id)


def dnfGNE(
        node_type_info: dict[int, NodeType], 
        children_info: dict[int, list[int]], 
        leaf2task: dict[int, int],
        tasks: list[list[int]],
        agents: list[dict[int, float]],
        constraints,
        capabilities: list[list[float]],
        nodes_upper_bound: dict[int, float],
        nodes_upper_bound_min: dict[int, float],
        coalition_structure : list[list[int]] = [],
        eps=0, 
        gamma=1,
        root_node_id=0,
    ):
    """
    GreedyNE on all possible leaves-subset solutions of an AND-OR goal tree.

    Equivalent to converting the AND-OR goal tree into a Disjunctive Normal Form (DNF) formula and iterating through all possible clauses.
    """
    final_system_reward = 0
    total_assessment_count = 0
    total_iteration_count = 0
    total_re_assignment_count = 0
    for leaves_subset, leaves_subset_path in traverse_and_or_tree(node_type_info=node_type_info, children_info=children_info, root_node_id=root_node_id):
        sys_reward_upper_bound = upper_bound_subsytem(
            selected_nodes=leaves_subset,
            nodes_upper_bound=nodes_upper_bound,
            nodes_upper_bound_min=nodes_upper_bound_min,
            node_type_info=node_type_info,
            leaf2task=leaf2task,
            capabilities=capabilities,
            tasks=tasks,
            agents=agents,
            constraints=constraints,
        )
        total_assessment_count += 1
        if sys_reward_upper_bound < final_system_reward:
            continue
        selected_tasks = [leaf2task[leaf] for leaf in leaves_subset]
        coalition_structure, sys_reward, iteration_count, re_assignment_count = aGreedyNE(
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
        if sys_reward > final_system_reward:
            final_system_reward = sys_reward

    return coalition_structure, final_system_reward, total_assessment_count, total_iteration_count, total_re_assignment_count