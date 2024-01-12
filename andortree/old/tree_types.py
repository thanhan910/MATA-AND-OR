from dataclasses import dataclass
from enum import Enum

class NodeType(Enum):
    AND = "AND"
    OR = "OR"
    LEAF = "LEAF"
    DUMMY = "DUMMY"

def reverse_node_type(node_type):
    if node_type == NodeType.AND:
        return NodeType.OR
    elif node_type == NodeType.OR:
        return NodeType.AND
    else:
        return node_type
    

def is_not_leaf(node_type):
    return node_type == NodeType.AND or node_type == NodeType.OR


@dataclass
class Node:
    node_id: int
    node_type: NodeType = None
    parent_id: int = None
    depth: int = None
    children_ids: list[int] = None