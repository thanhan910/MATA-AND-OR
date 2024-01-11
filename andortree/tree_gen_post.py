import random
from .tree_types import NodeType, Node, reverse_node_type


def gen_conventional_node_ids(leaf_nodes : list[int], children_info : dict[int, list[int]], root_node_id : int = 0):
    """
    Generate new node ids, based on this convention:
    - leaf nodes are numbered first, from 0 to len(leaf_nodes) - 1 (inclusive)
    - then, a dummy node is added. Its id is len(leaf_nodes).
    - then non-leaf nodes, from len(leaf_nodes) + 1 to len(leaf_nodes) + len(children_info) - 1 (inclusive)
    - then finally, the root node
    """
    middle_nodes = [node_id for node_id in children_info.keys() if node_id != root_node_id and len(children_info[node_id]) > 0]
    random.shuffle(middle_nodes)
    random.shuffle(leaf_nodes)
    # old_dummy_node_id = -1 # stub
    node_ids_sorted = leaf_nodes + [-1] + middle_nodes + [root_node_id]
    new_node_ids_map = { old_node_id : new_node_id for new_node_id, old_node_id in enumerate(node_ids_sorted) }
    new_root_node_id = len(node_ids_sorted) - 1
    new_leaf_nodes = list(range(len(leaf_nodes)))
    return new_node_ids_map, new_leaf_nodes, new_root_node_id


def reevaluate_tree(depth_info : dict[int, int], parent_info : dict[int, int], children_info : dict[int, int], new_node_ids_map : dict[int, int]):
    """
    Reevaluate tree information based on the new node_id mapping.
    """
    new_parent_info = {}
    new_children_info = {}
    new_depth_info = {}
    for old_node_id, new_node_id in new_node_ids_map.items():
        if old_node_id in parent_info:
            new_parent_info[new_node_id] = new_node_ids_map[parent_info[old_node_id]]
        if old_node_id in children_info:
            new_children_info[new_node_id] = [new_node_ids_map[child_id] for child_id in children_info[old_node_id]]
        if old_node_id in depth_info:
            new_depth_info[new_node_id] = depth_info[old_node_id]
    return new_depth_info, new_parent_info, new_children_info



def assign_node_type(depth_info: dict[int, int], leaf_nodes : list[int], children_info: dict[int, int], root_node_id : int, strict_and_or: bool = True, root_node_type : NodeType = None):
    """
    Randomly assigns a node type to each node in the tree, where the node type is either "AND", "OR", or "LEAF".
    """
    node_type_info = {}
    if root_node_type is None:
        root_node_type = random.choice([NodeType.AND, NodeType.OR])
    reversed_root_node_type = reverse_node_type(root_node_type)
    node_type_info[root_node_id] = root_node_type
    for node_id in leaf_nodes:
        node_type_info[node_id] = NodeType.LEAF
    for node_id, node_children_ids in children_info.items():
        if node_children_ids is None or len(node_children_ids) == 0:
            node_type_info[node_id] = NodeType.LEAF
            continue

        if node_id == root_node_id:
            continue
        
        if strict_and_or:
            node_depth = depth_info[node_id]
            node_type_info[node_id] = root_node_type if node_depth % 2 == 0 else reversed_root_node_type
        else:
            node_type_info[node_id] = random.choice([NodeType.AND, NodeType.OR])

    return node_type_info


def get_tree_info(tree_info_size : int, depth_info : dict[int, int], parent_info : dict[int, int], children_info : dict[int, list[int]], node_type_info : dict[int, NodeType]):
    return [Node(
        node_id=node_id,
        node_type=NodeType(node_type_info[node_id]),
        parent_id=parent_info[node_id] if node_id in parent_info else None,
        children_ids=children_info[node_id] if node_id in children_info else None,
        depth=depth_info[node_id] if node_id in depth_info else None,
    ) for node_id in range(tree_info_size)]
