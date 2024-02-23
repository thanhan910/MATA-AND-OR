from functools import reduce  # python3 compatibility
from operator import mul
import json
import math
import os
import random
import statistics
import argparse
import concurrent.futures

from utils.problem import gen_tasks, gen_constraints, gen_agents

from heterogeneous.GreedyNE import eGreedy2
from heterogeneous.solutions import *
from heterogeneous.upper_bound import *

from andortree.tree_gen import gen_tree, assign_node_type
from andortree.tree_utils import get_leaves_list_info, get_nodes_constraints
from andortree.upper_bound import upperbound_node_all, upperbound_node_all_min, upperbound_node, calculate_ubc_vectors, get_cap_vector_all
# from andortree.GreedyNE import *
from andortree.solutions import random_solution_and_or_tree
from andortree.treeGNE import treeGNE, treeGNE2, fastTreeGNE2
from andortree.simpleGNE import simpleGNE
from andortree.dnfGNE import dnfGNE
from andortree.AOGNE import AOsearchGNE
from andortree.OrNE import BnBOrNE



def append_record(record, filename, typ):
    with open(filename, "a") as f:
        if typ != "":
            json.dump(record, f, default=typ)
        else:
            json.dump(record, f)
        f.write(os.linesep)
        f.close()


def generate_tree_problem(tasks):
    """
    Encapsulate the generation of variables that represent the tree.

    This is so that only a number of data structures and variable types can be used as initial input to the MATA problem.

    All other variables must be generated within the solution function and/or must be counted as a procedure within the solution, and thus when calculating the solution speed, we must include the time taken to generate those other variables.
    """
    task_num = len(tasks)

    depth_info, parent_info, children_info, leaves_by_depth = gen_tree(task_num, min_leaf_depth=2)
    
    leaf_nodes = sum(leaves_by_depth, [])

    random.shuffle(leaf_nodes)

    leaf2task = {leaf_id : j for j, leaf_id in enumerate(leaf_nodes)}

    node_type_info = assign_node_type(depth_info, children_info, leaf_nodes)

    return node_type_info, parent_info, children_info, leaf2task, leaf_nodes, leaves_by_depth, depth_info


def solve_get_upper_bound(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    start = time.perf_counter()
    leaves_list_info = get_leaves_list_info(parent_info=parent_info, leaf_nodes=leaf_nodes)
    nodes_constraints = get_nodes_constraints(node_type_info=node_type_info, leaves_list_info=leaves_list_info, leaf2task=leaf2task, constraints=constraints)
    tasks_capVecs = get_cap_vector_all(capabilities, tasks)
    ubcv_info = calculate_ubc_vectors(node_type_info, parent_info, leaves_list_info, leaf2task, tasks_capVecs, capabilities, query_nodeId=0)
    nodes_upper_bound = upperbound_node_all(children_info, ubcv_info, capabilities, agents, nodes_constraints, query_nodeId=0)
    nodes_upper_bound_min = upperbound_node_all_min(nodes_upper_bound, node_type_info, children_info, query_nodeId=0)
    end = time.perf_counter()
    
    print("UPPER BOUND:", nodes_upper_bound_min[0], "\ttime:", end - start)

    return nodes_upper_bound_min[0]


def solve_random_solution(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    start = time.perf_counter()
    rand_sol_alloc, rand_sol_reward = random_solution_and_or_tree(node_type_info, children_info, leaf2task, tasks, agents, constraints, gamma)
    end = time.perf_counter()

    print(f"Random: {rand_sol_reward}\ttime: {end - start}")

    return {
        "reward": rand_sol_reward,
        "time": end - start,
    }


def solve_treeGNE_1(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    start = time.perf_counter()
    result_c = treeGNE(
        node_type_info=node_type_info,
        children_info=children_info,
        parent_info=parent_info,
        task2leaf=leaf_nodes,
        leaf2task=leaf2task,
        tasks=tasks,
        agents=agents,
        constraints=constraints,
        gamma=gamma,
    )
    end = time.perf_counter()

    print(f"TreeGNE: {result_c[1][0]}\ttime: {end - start}\titeration: {result_c[2]}\tre-assignment {result_c[3]}")

    return {
        "reward": result_c[1][0],
        "time": end - start,
        "iteration": result_c[2],
        "re-assignment": result_c[3],
    }

def solve_treeGNE_2(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    
    start = time.perf_counter()
    result_c = treeGNE2(
        node_type_info=node_type_info,
        children_info=children_info,
        parent_info=parent_info,
        task2leaf=leaf_nodes,
        leaf2task=leaf2task,
        tasks=tasks,
        agents=agents,
        constraints=constraints,
        gamma=gamma,
    )
    end = time.perf_counter()

    print(f"TreeGNE2: {result_c[1][0]}\ttime: {end - start}\titeration: {result_c[2]}\tre-assignment {result_c[3]}")

    return {
        "reward": result_c[1][0],
        "time": end - start,
        "iteration": result_c[2],
        "re-assignment": result_c[3],
    }


def solve_fastTreeGNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    start = time.perf_counter()
    result_c = fastTreeGNE2(
        node_type_info=node_type_info,
        children_info=children_info,
        parent_info=parent_info,
        task2leaf=leaf_nodes,
        leaf2task=leaf2task,
        tasks=tasks,
        agents=agents,
        constraints=constraints,
        gamma=gamma,
    )
    end = time.perf_counter()

    print(f"fastTreeGNE2: {result_c[1][0]}\ttime: {end - start}\titeration: {result_c[2]}\tre-assignment {result_c[3]}")

    return {
        "reward": result_c[1][0],
        "time": end - start,
        "iteration": result_c[2],
        "re-assignment": result_c[3],
    }


def solve_simpleGNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    start = time.perf_counter()
    r_coalition_structure, r_sys_reward, r_iteration_count_1, r_re_assignment_count_1, r_iteration_count_2, r_re_assignment_count_2, r_loop_count = simpleGNE(
        node_type_info=node_type_info,
        children_info=children_info,
        leaf2task=leaf2task,
        agents=agents,
        tasks=tasks,
        constraints=constraints,
        gamma=gamma,
    )
    end = time.perf_counter()

    
    print(f"simpleGNE: {r_sys_reward}\ttime: {end - start}\titeration 1: {r_iteration_count_1}\tre-assignment 1 {r_re_assignment_count_1}\titeration 2: {r_iteration_count_2}\tre-assignment 2 {r_re_assignment_count_2}\tloop: {r_loop_count}")

    return {
        "reward": r_sys_reward,
        "time": end - start,
        "iteration_1": r_iteration_count_1,
        "re-assignment_1": r_re_assignment_count_1,
        "iteration_2": r_iteration_count_2,
        "re-assignment_2": r_re_assignment_count_2,
        "loop": r_loop_count,
    }


def solve_AOSearchGNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    start = time.perf_counter()
    leaves_list_info = get_leaves_list_info(parent_info=parent_info, leaf_nodes=leaf_nodes)
    nodes_constraints = get_nodes_constraints(node_type_info=node_type_info, leaves_list_info=leaves_list_info, leaf2task=leaf2task, constraints=constraints)
    tasks_capVecs = get_cap_vector_all(capabilities, tasks)
    ubcv_info = calculate_ubc_vectors(node_type_info, parent_info, leaves_list_info, leaf2task, tasks_capVecs, capabilities, query_nodeId=0)
    nodes_upper_bound = upperbound_node_all(children_info, ubcv_info, capabilities, agents, nodes_constraints, query_nodeId=0)
    nodes_upper_bound_min = upperbound_node_all_min(nodes_upper_bound, node_type_info, children_info, query_nodeId=0)
    raos_coalition_structure, raos_sys_reward, raos_iteration_count, raos_re_assignment_count, raos_loop_count = AOsearchGNE(
        node_type_info=node_type_info,
        children_info=children_info,
        parent_info=parent_info,
        nodes_upper_bound=nodes_upper_bound,
        nodes_upper_bound_min=nodes_upper_bound_min,
        leaf2task=leaf2task,
        capabilities=capabilities,
        tasks=tasks,
        agents=agents,
        constraints=constraints,
        gamma=gamma,
        root_node_id=0,
    )
    end = time.perf_counter()


    print(f"AOsearchGNE: {raos_sys_reward}\ttime: {end - start}\titeration: {raos_iteration_count}\tre-assignment {raos_re_assignment_count}\tloop: {raos_loop_count}")

    return {
        "reward": raos_sys_reward,
        "time": end - start,
        "iteration": raos_iteration_count,
        "re-assignment": raos_re_assignment_count,
        "loop": raos_loop_count,
    }


def solve_OrNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    
    start = time.perf_counter()
    leaves_list_info = get_leaves_list_info(parent_info=parent_info, leaf_nodes=leaf_nodes)
    nodes_constraints = get_nodes_constraints(node_type_info=node_type_info, leaves_list_info=leaves_list_info, leaf2task=leaf2task, constraints=constraints)
    tasks_capVecs = get_cap_vector_all(capabilities, tasks)
    ubcv_info = calculate_ubc_vectors(node_type_info, parent_info, leaves_list_info, leaf2task, tasks_capVecs, capabilities, query_nodeId=0)
    rorne_alloc, rorne_sys_reward, rorne_iteration_count, rorne_re_assignment_count = BnBOrNE(
        node_type_info=node_type_info,
        children_info=children_info,
        ubcv_info=ubcv_info,
        leaf2task=leaf2task,
        task2leaf=leaf_nodes,
        leaves_list_info=leaves_list_info,
        capabilities=capabilities,
        tasks=tasks,
        agents=agents,
        constraints=constraints,
        nodes_constraints=nodes_constraints,
        coalition_structure=None,
        gamma=gamma,
        root_node_id=0,
        use_branch_and_bound=False,
    )
    end = time.perf_counter()

    print(f"OrNE: {rorne_sys_reward}\ttime: {end - start}\titeration: {rorne_iteration_count}\tre-assignment {rorne_re_assignment_count}")

    return {
        "reward": rorne_sys_reward,
        "time": end - start,
        "iteration": rorne_iteration_count,
        "re-assignment": rorne_re_assignment_count,
    }


def solve_BnBOrNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    start = time.perf_counter()
    leaves_list_info = get_leaves_list_info(parent_info=parent_info, leaf_nodes=leaf_nodes)
    nodes_constraints = get_nodes_constraints(node_type_info=node_type_info, leaves_list_info=leaves_list_info, leaf2task=leaf2task, constraints=constraints)
    tasks_capVecs = get_cap_vector_all(capabilities, tasks)
    ubcv_info = calculate_ubc_vectors(node_type_info, parent_info, leaves_list_info, leaf2task, tasks_capVecs, capabilities, query_nodeId=0)
    rorne_alloc, rorne_sys_reward, rorne_iteration_count, rorne_re_assignment_count = BnBOrNE(
        node_type_info=node_type_info,
        children_info=children_info,
        ubcv_info=ubcv_info,
        leaf2task=leaf2task,
        task2leaf=leaf_nodes,
        leaves_list_info=leaves_list_info,
        capabilities=capabilities,
        tasks=tasks,
        agents=agents,
        constraints=constraints,
        nodes_constraints=nodes_constraints,
        coalition_structure=None,
        gamma=gamma,
        root_node_id=0,
    )
    end = time.perf_counter()

    print(f"BnBOrNE: {rorne_sys_reward}\ttime: {end - start}\titeration: {rorne_iteration_count}\tre-assignment {rorne_re_assignment_count}")

    return {
        "reward": rorne_sys_reward,
        "time": end - start,
        "iteration": rorne_iteration_count,
        "re-assignment": rorne_re_assignment_count,
    }


def solve_BnBOrNE_skip(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    
    start = time.perf_counter()
    leaves_list_info = get_leaves_list_info(parent_info=parent_info, leaf_nodes=leaf_nodes)
    nodes_constraints = get_nodes_constraints(node_type_info=node_type_info, leaves_list_info=leaves_list_info, leaf2task=leaf2task, constraints=constraints)
    tasks_capVecs = get_cap_vector_all(capabilities, tasks)
    ubcv_info = calculate_ubc_vectors(node_type_info, parent_info, leaves_list_info, leaf2task, tasks_capVecs, capabilities, query_nodeId=0)
    rorne_alloc, rorne_sys_reward, rorne_iteration_count, rorne_re_assignment_count = BnBOrNE(
        node_type_info=node_type_info,
        children_info=children_info,
        ubcv_info=ubcv_info,
        leaf2task=leaf2task,
        task2leaf=leaf_nodes,
        leaves_list_info=leaves_list_info,
        capabilities=capabilities,
        tasks=tasks,
        agents=agents,
        constraints=constraints,
        nodes_constraints=nodes_constraints,
        coalition_structure=None,
        gamma=gamma,
        root_node_id=0,
        skip_initial_branch=True,
    )
    end = time.perf_counter()

    print(f"BnBOrNEskip: {rorne_sys_reward}\ttime: {end - start}\titeration: {rorne_iteration_count}\tre-assignment {rorne_re_assignment_count}")

    return {
        "reward": rorne_sys_reward,
        "time": end - start,
        "iteration": rorne_iteration_count,
        "re-assignment": rorne_re_assignment_count,
    }


def solve_dnfGNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes):
    start = time.perf_counter()
    leaves_list_info = get_leaves_list_info(parent_info=parent_info, leaf_nodes=leaf_nodes)
    nodes_constraints = get_nodes_constraints(node_type_info=node_type_info, leaves_list_info=leaves_list_info, leaf2task=leaf2task, constraints=constraints)
    tasks_capVecs = get_cap_vector_all(capabilities, tasks)
    ubcv_info = calculate_ubc_vectors(node_type_info, parent_info, leaves_list_info, leaf2task, tasks_capVecs, capabilities, query_nodeId=0)
    nodes_upper_bound = upperbound_node_all(children_info, ubcv_info, capabilities, agents, nodes_constraints, query_nodeId=0)
    nodes_upper_bound_min = upperbound_node_all_min(nodes_upper_bound, node_type_info, children_info, query_nodeId=0)
    rdnf_coalition_structure, rdnf_system_reward, rdnf_total_assessment_count, rdnf_iteration_count, rdnf_re_assignment_count = dnfGNE(
        node_type_info=node_type_info,
        children_info=children_info,
        leaf2task=leaf2task,
        tasks=tasks,
        agents=agents,
        constraints=constraints,
        capabilities=capabilities,
        nodes_upper_bound=nodes_upper_bound,
        nodes_upper_bound_min=nodes_upper_bound_min,
        gamma=gamma,
    )
    end = time.perf_counter()
    print(f"dnfGNE: {rdnf_system_reward}\ttime: {end - start}\tassessment: {rdnf_total_assessment_count}\titeration: {rdnf_iteration_count}\tre-assignment {rdnf_re_assignment_count}")

    return {
        "reward": rdnf_system_reward,
        "time": end - start,
        "assessment": rdnf_total_assessment_count,
        "iteration": rdnf_iteration_count,
        "re-assignment": rdnf_re_assignment_count,
    }


def main_tree(capabilities, tasks, agents, constraints, gamma):

    """
    Driver code for algorithms related to AND-OR goal tree.
    """
    result_row = {}

    task_num = len(tasks)

    result_row["task_num"] = task_num

    node_type_info, parent_info, children_info, leaf2task, leaf_nodes, leaves_by_depth, depth_info = generate_tree_problem(tasks)

    max_depth = len(leaves_by_depth) - 1
    min_depth = 0
    while len(leaves_by_depth[min_depth]) == 0:
        min_depth += 1

    branching_factor = (len(node_type_info) - 1) / len(children_info)
    min_degree = min([len(c) for c in children_info.values()])
    max_degree = max([len(c) for c in children_info.values()])
    num_internal_nodes = len(children_info)

    result_row["tree_info"] = {
        "max_depth": max_depth,
        "min_depth": min_depth,
        "branching_factor": branching_factor,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "num_internal_nodes": num_internal_nodes,
    }

    result_row["upper_bound"] = solve_get_upper_bound(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes)

    result_row["random_solution"] = solve_random_solution(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes)

    result_row["treeGNE"] = solve_treeGNE_1(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes)

    result_row["treeGNE2"] = solve_treeGNE_2(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes)

    result_row["fastTreeGNE2"] = solve_fastTreeGNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes)

    result_row["simpleGNE"] = solve_simpleGNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes)

    result_row["AOsearchGNE"] = solve_AOSearchGNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes)

    result_row["OrNE"] = solve_OrNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes)

    result_row["BnBOrNE"] = solve_BnBOrNE(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes)

    result_row["BnBOrNEskip"] = solve_BnBOrNE_skip(capabilities, tasks, agents, constraints, gamma, node_type_info, parent_info, children_info, leaf2task, leaf_nodes)

    return result_row




def main_original_opd_fms(tasks, agents, constraints, gamma, t_max_edge, result, time_bound):
    """
    Original driver code for algorithms related to OPD and FMS.
    """

    if t_max_edge < 15:
        start = time.perf_counter()
        opd = OPD(agents, tasks, constraints, gamma)
        new_con = opd[0:2]
        result["OPD_t"] = time.perf_counter() - start
        r = FMS(agents, tasks, new_con, gamma=gamma, time_bound=time_bound)
        end = time.perf_counter()
        result["OPD_pruned"] = opd[2]
        print(
            "OPD time used:",
            result["OPD_t"],
            " Edge pruned:",
            result["OPD_pruned"],
        )
        result["BnBFMS"] = r[1]
        result["BnBFMS_iter"] = r[2]
        result["BnBFMS_t"] = end - start
        result["BnBFMS_converge"] = r[4]
        print(
            "BnB FMS time:",
            result["BnBFMS_t"],
            "result:",
            result["BnBFMS"],
            "iteration:",
            result["BnBFMS_iter"],
            "converge?",
            result["BnBFMS_converge"],
        )
    print()

    return t_max_edge, result, time_bound


def main_run(task_num, agent_num, capNum, t_max_edge, a_min_edge, ex_identifier = None, save_to_file = None):
    
    gamma = 1

    max_capVal = capNum
    max_capNum_task = capNum
    max_capNum_agent = capNum
    time_bound = 600

    capabilities = list(range(0, capNum))

    # agent_num = np.random.randint(task_num,3*task_num)
    tasks = gen_tasks(task_num, max_capNum_task, capabilities)
    constraints = gen_constraints(agent_num, task_num, 1, a_min_edge, t_max_edge)
    a_taskInds = constraints[0]
    agents_cap, agents = gen_agents(a_taskInds, tasks, max_capNum_agent, capabilities, max_capVal)
    # num_com = np.prod([1 if a_taskInds[i] == [] else len(a_taskInds[i])+1 for i in range(0,agent_num)])

    num_com = reduce(
        mul,
        [
            1 if a_taskInds[i] == [] else len(a_taskInds[i]) + 1
            for i in range(0, agent_num)
        ],
    )

    up = upperBound(capabilities, tasks, agents)

    up2 = upperBound_ver2(capabilities, tasks, agents, constraints)
    print("UP:", up, "  UP2:", up2)

    result_info = {
        "ex_identifier": ex_identifier,
        "task_num": task_num,
        "agent_num": agent_num,
        "capNum": capNum,
        "up": up,
        "up2": up2,
    }
    #         data = {"ex_identifier":ex_identifier,"tasks":tasks,"constraints":constraints,"agents_cap":agents_cap,"agents":agents}

    a_den = [len(c) for c in constraints[0]]
    t_den = [len(c) for c in constraints[1]]
    result_info["a_den_avg"] = statistics.mean(a_den)
    result_info["a_den_max"] = max(a_den)
    result_info["a_den_min"] = min(a_den)

    result_info["t_den_avg"] = statistics.mean(t_den)
    result_info["t_den_max"] = max(t_den)
    result_info["t_den_min"] = min(t_den)
    result_info["t_max_edge"] = t_max_edge

    print(
        "density", t_max_edge, "task_num:", task_num, "  agent_num:", agent_num
    )
    print(
        "a_den_avg",
        result_info["a_den_avg"],
        "a_den_max:",
        result_info["a_den_max"],
        "  a_den_min:",
        result_info["a_den_min"],
    )

    print(
        "t_den_avg",
        result_info["t_den_avg"],
        "t_den_max:",
        result_info["t_den_max"],
        "  t_den_min:",
        result_info["t_den_min"],
    )
    print("-----------------------------------")
    print("Heterogeneous Tasks")
    print("-----------------------------------")

    start = time.perf_counter()
    r = eGreedy2(agents, tasks, constraints, gamma=gamma)
    end = time.perf_counter()
    result_info["g"] = r[1]
    result_info["g_iter"] = r[2]
    result_info["g_reass"] = r[3]
    result_info["g_t"] = end - start
    print(
        "eGreedy:",
        "\ttime:",
        result_info["g_t"],
        "\tresult:",
        result_info["g"],
        "\titeration:",
        result_info["g_iter"],
        "\tre-assignment",
        result_info["g_reass"],
    )

    start = time.perf_counter()
    rand_sol_a, rand_sol_reward = random_solution_heterogeneous(agents, tasks, constraints, gamma=gamma)
    end = time.perf_counter()
    print("Random Solution:", "\ttime:", end - start, "\tresult:", rand_sol_reward)

    print("-----------------------------------")
    print("AND-OR Tree Tasks")
    print("-----------------------------------")

    result_row = main_tree(capabilities, tasks, agents, constraints, gamma)
    result_row["info"] = result_info

    if save_to_file:
        append_record(result_row, save_to_file, typ="")

    return result_row


def main_run_1(args):
    return main_run(*args)


def main_single(remove_file = False):

    if remove_file:
        if os.path.exists("local-results.jsonl"):
            os.remove("local-results.jsonl")
    
    ex_identifier = 0

    for task_num in range(100, 1000, 100):
        for agent_tasks_ratio in range(2, 5):
            agent_num = task_num * agent_tasks_ratio
            for capNum in range(10, 15):
                run_num = 3
                a_min_edge = 2
                for run in range(0, run_num):
                    # print("----------------------------------------------------------------------")
                    # print("ITERATION:", run)
                    # print("----------------------------------------------------------------------")
                    min_t_max_edge = max(math.ceil((agent_num * a_min_edge) / task_num), 10)
                    max_t_max_edge = min_t_max_edge + 5 * 3
                    for t_max_edge in range(min_t_max_edge, max_t_max_edge + 1, 5):
                        ex_identifier += 1
                        print("----------------------------------------------------------------------")
                        print("EX IDENTIFIER:", ex_identifier)
                        print("----------------------------------------------------------------------")
                        result_row = main_run(task_num, agent_num, capNum, t_max_edge, a_min_edge, ex_identifier)
                        # append data and result
                        files = {"local-results.jsonl": [result_row, ""]}

                        for filename in list(files.keys()):
                            append_record(files[filename][0], filename, typ=files[filename][1])


def main_cli_full_args():

    parser = argparse.ArgumentParser(description="Run the main function.")
    parser.add_argument(
        "--task_num",
        type=int,
        default=100,
        help="Number of tasks to generate.",
    )
    parser.add_argument(
        "--agent_num",
        type=int,
        default=200,
        help="Number of agents to generate.",
    )
    parser.add_argument(
        "--capNum",
        type=int,
        default=10,
        help="Number of capabilities to generate.",
    )
    parser.add_argument(
        "--t_max_edge",
        type=int,
        default=15,
        help="Maximum number of edges to generate.",
    )
    parser.add_argument(
        "--a_min_edge",
        type=int,
        default=2,
        help="Minimum number of edges to generate.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of iterations to run.",
    )

    args = parser.parse_args()

    if args.iterations == 0:
        args.iterations = 3

    for run in range(0, args.iterations):
        result_row = main_run(
            args.task_num,
            args.agent_num,
            args.capNum,
            args.t_max_edge,
            args.a_min_edge,
            None,
            'results-cli.jsonl',
        )


def main_cli():

    parser = argparse.ArgumentParser(description="Run by task_num.")
    parser.add_argument(
        "--task_num",
        type=int,
        default=100,
        help="Number of tasks to generate.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of iterations to run.",
    )

    args = parser.parse_args()

    if args.iterations == 0:
        args.iterations = 3

    task_num = args.task_num

    for agent_tasks_ratio in range(2, 5):
        agent_num = task_num * agent_tasks_ratio
        for capNum in range(10, 15):
            a_min_edge = 2
            min_t_max_edge = max(math.ceil((agent_num * a_min_edge) / task_num), 10)
            max_t_max_edge = max(min_t_max_edge, 50)
            for t_max_edge in range(min_t_max_edge, max_t_max_edge + 1):
                run_num = 3
                for run in range(0, run_num):
                    result_row = main_run(task_num, agent_num, capNum, t_max_edge, a_min_edge)
                    # append data and result
                    files = {"local-results.jsonl": [result_row, ""]}

                    for filename in list(files.keys()):
                        append_record(files[filename][0], filename, typ=files[filename][1])




def main_multiprocessing():
    from multiprocessing import Pool

    args = []

    ex_identifier = 0

    for task_num in range(100, 1000, 100):
        for agent_tasks_ratio in range(2, 5):
            agent_num = task_num * agent_tasks_ratio
            for capNum in range(10, 15):
                a_min_edge = 2
                min_t_max_edge = max(math.ceil((agent_num * a_min_edge) / task_num), 10)
                max_t_max_edge = max(min_t_max_edge, 50)
                for t_max_edge in range(min_t_max_edge, max_t_max_edge + 1):
                    run_num = 3
                    for run in range(0, run_num):
                        # print("----------------------------------------------------------------------")
                        # print("EXPERIMENT")
                        # print("----------------------------------------------------------------------")
                        # result_row = main_run(task_num, agent_num, capNum, t_max_edge, a_min_edge)
                        # # append data and result
                        # files = {"local-results.jsonl": [result_row, ""]}

                        # for filename in list(files.keys()):
                        #     append_record(files[filename][0], filename, typ=files[filename][1])
                        ex_identifier += 1
                        args.append((task_num, agent_num, capNum, t_max_edge, a_min_edge, ex_identifier, "results-multiprocessing.jsonl"))

    with Pool(5) as p:
        p.map(main_run_1, args) 

def main_multithread():

    ex_identifier = 0

    args = []

    for task_num in range(100, 1000, 100):
        for agent_tasks_ratio in range(2, 5):
            agent_num = task_num * agent_tasks_ratio
            for capNum in range(10, 15):
                a_min_edge = 2
                min_t_max_edge = max(math.ceil((agent_num * a_min_edge) / task_num), 10)
                max_t_max_edge = max(min_t_max_edge, 50)
                for t_max_edge in range(min_t_max_edge, max_t_max_edge + 1):
                    run_num = 3
                    for run in range(0, run_num):
                        ex_identifier += 1
                        args.append((task_num, agent_num, capNum, t_max_edge, a_min_edge, ex_identifier, "results-multithread.jsonl"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(main_run_1, args)

if __name__ == "__main__":
    main_single()