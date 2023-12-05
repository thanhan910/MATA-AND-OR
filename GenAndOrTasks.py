import numpy as np

def gen_random_partition(array, flatten: bool = True):
    """
    Generate a random partition of the given array.
    """
    partition = []
    for element in array:
        random_index = np.random.randint(0, len(partition) + 1)
        if random_index == len(partition):
            partition.append([element])
        else:
            partition[random_index].append(element)

    if len(partition) == 1:
        partition = partition[0]

    if flatten:
        for i, subset in enumerate(partition):
            if len(subset) == 1:
                partition[i] = partition[i][0]

    return partition


def gen_and_or_tree(task_num, max_depth=5):
    """
    Generate a random AND/OR tree with `task_num` tasks.
    Returns a list of lists or integers, where each element/subelement is a node.
    Each leaf node is an integer from 0 to `task_num-1`, and each non-leaf node is a list.
    """
    # Fix max_depth to be at least 1 and at most task_num
    max_depth = max(1, min(max_depth, task_num))

    # Generate a random AND/OR tree with task_num tasks
    tree = list(range(task_num))
    for depth in range(max_depth - 1):
        tree = gen_random_partition(tree, flatten=True)

    # Randomly choose the root node type
    root_node_type = np.random.choice(['AND', 'OR'])
    
    return tree, root_node_type