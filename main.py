from functools import reduce  # python3 compatibility
from operator import mul
import json
import os
import random
import statistics

from utils.problem import *

from heterogeneous.GreedyNE import * 
from heterogeneous.solutions import *
from heterogeneous.upper_bound import *

from andortree.tree_gen import gen_tree, assign_node_type
from andortree.tree_utils import get_leaves_list_info, get_nodes_constraints
from andortree.upper_bound import upperbound_node_all, upperbound_node_all_min, upperbound_node, calculate_ubc_vectors, get_cap_vector_all
# from andortree.GreedyNE import *
from andortree.solutions import random_solution_and_or_tree
from andortree.treeGNE import treeGNE, treeGNE2
from andortree.naiveGNE import naiveGNE
from andortree.dnfGNE import dnfGNE



def append_record(record, filename, typ):
    with open(filename, "a") as f:
        if typ != "":
            json.dump(record, f, default=typ)
        else:
            json.dump(record, f)
        f.write(os.linesep)
        f.close()


def main_tree(capabilities, tasks, agents, constraints, gamma):

    """
    Driver code for algorithms related to AND-OR goal tree.
    """

    task_num = len(tasks)

    depth_info, parent_info, children_info, leaves_by_depth = gen_tree(task_num, min_leaf_depth=2)
    
    leaf_nodes = sum(leaves_by_depth, [])

    random.shuffle(leaf_nodes)

    leaf2task = {leaf_id : j for j, leaf_id in enumerate(leaf_nodes)}

    node_type_info = assign_node_type(depth_info, children_info, leaf_nodes)

    leaves_list_info = get_leaves_list_info(parent_info=parent_info, leaf_nodes=leaf_nodes)

    nodes_constraints = get_nodes_constraints(node_type_info=node_type_info, leaves_list_info=leaves_list_info, leaf2task=leaf2task, constraints=constraints)

    tasks_capVecs = get_cap_vector_all(capabilities, tasks)

    ubcv_info = calculate_ubc_vectors(node_type_info, parent_info, leaves_list_info, leaf2task, tasks_capVecs, capabilities, query_nodeId=0)

    start = time.perf_counter()
    up_tree_root = upperbound_node(ubcv_info, capabilities, agents, nodes_constraints, query_nodeId=0)
    end = time.perf_counter()

    print("UP Tree:", up_tree_root, "\ttime:", end - start)

    start = time.perf_counter()
    nodes_upper_bound = upperbound_node_all(children_info, ubcv_info, capabilities, agents, nodes_constraints, query_nodeId=0)
    end = time.perf_counter()

    print("UP1:", nodes_upper_bound[0], "\ttime:", end - start)

    start = time.perf_counter()
    nodes_upper_bound_min = upperbound_node_all_min(nodes_upper_bound, node_type_info, children_info, query_nodeId=0)
    end = time.perf_counter()

    print("UP2:", nodes_upper_bound_min[0], "\ttime:", end - start)

    start = time.perf_counter()
    rand_sol_alloc, rand_sol_reward = random_solution_and_or_tree(node_type_info, children_info, leaf2task, tasks, agents, constraints, gamma)
    end = time.perf_counter()
    print(f"Random: {rand_sol_reward}\ttime: {end - start}")

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


    start = time.perf_counter()
    r_coalition_structure, r_sys_reward, r_iteration_count_1, r_re_assignment_count_1, r_iteration_count_2, r_re_assignment_count_2, r_loop_count = naiveGNE(
        node_type_info=node_type_info,
        children_info=children_info,
        leaf2task=leaf2task,
        agents=agents,
        tasks=tasks,
        constraints=constraints,
        gamma=gamma,
    )
    end = time.perf_counter()
    print(f"naiveGNE: {r_sys_reward}\ttime: {end - start}\titeration 1: {r_iteration_count_1}\tre-assignment 1 {r_re_assignment_count_1}\titeration 2: {r_iteration_count_2}\tre-assignment 2 {r_re_assignment_count_2}\tloop: {r_loop_count}")


    start = time.perf_counter()
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


def main():
    run_num = 1000
    gamma = 1
    a_min_edge = 1

    capNum = 10
    max_capVal = 10
    max_capNum_task = 10
    max_capNum_agent = 10
    ex_identifier = 0
    time_bound = 600
    task_num = 100
    agent_num = 200

    capabilities = list(range(0, capNum))

    for run in range(0, run_num):
        print("-----------------------------------")
        print("ITERATION:", run)
        print("-----------------------------------")
        t_max_edge = 3
        while t_max_edge <= 50:
            print("-----------------------------------")
            print("t_max_edge:", t_max_edge)
            print("-----------------------------------")
            ex_identifier += 1

            #         agent_num = np.random.randint(task_num,3*task_num)
            tasks = gen_tasks(task_num, max_capNum_task, capabilities)
            constraints = gen_constraints(
                agent_num, task_num, 1, a_min_edge, t_max_edge
            )
            a_taskInds = constraints[0]
            agents_cap, agents = gen_agents(
                a_taskInds, tasks, max_capNum_agent, capabilities, max_capVal
            )
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

            result = {
                "ex_identifier": ex_identifier,
                "task_num": task_num,
                "agent_num": agent_num,
                "up": up,
                "up2": up2,
            }
            #         data = {"ex_identifier":ex_identifier,"tasks":tasks,"constraints":constraints,"agents_cap":agents_cap,"agents":agents}

            a_den = [len(c) for c in constraints[0]]
            t_den = [len(c) for c in constraints[1]]
            result["a_den_avg"] = statistics.mean(a_den)
            result["a_den_max"] = max(a_den)
            result["a_den_min"] = min(a_den)

            result["t_den_avg"] = statistics.mean(t_den)
            result["t_den_max"] = max(t_den)
            result["t_den_min"] = min(t_den)
            result["t_max_edge"] = t_max_edge

            print(
                "density", t_max_edge, "task_num:", task_num, "  agent_num:", agent_num
            )
            print(
                "a_den_avg",
                result["a_den_avg"],
                "a_den_max:",
                result["a_den_max"],
                "  a_den_min:",
                result["a_den_min"],
            )

            print(
                "t_den_avg",
                result["t_den_avg"],
                "t_den_max:",
                result["t_den_max"],
                "  t_den_min:",
                result["t_den_min"],
            )
            print("-----------------------------------")
            print("Heterogeneous Tasks")
            print("-----------------------------------")

            start = time.perf_counter()
            r = eGreedy2(agents, tasks, constraints, gamma=gamma)
            end = time.perf_counter()
            result["g"] = r[1]
            result["g_iter"] = r[2]
            result["g_reass"] = r[3]
            result["g_t"] = end - start
            print(
                "eGreedy:",
                "\ttime:",
                result["g_t"],
                "\tresult:",
                result["g"],
                "\titeration:",
                result["g_iter"],
                "\tre-assignment",
                result["g_reass"],
            )

            start = time.perf_counter()
            rand_sol_a, rand_sol_reward = random_solution_heterogeneous(agents, tasks, constraints, gamma=gamma)
            end = time.perf_counter()
            print("Random Solution:", "\ttime:", end - start, "\tresult:", rand_sol_reward)

            print("-----------------------------------")
            print("AND-OR Tree Tasks")
            print("-----------------------------------")

            main_tree(capabilities, tasks, agents, constraints, gamma)


            # print("-----------------------------------")
            # print("Heterogeneous Tasks (OPD, FMS)")
            # print("-----------------------------------")

            # t_max_edge, result, time_bound = main_original_opd_fms(tasks, agents, constraints, gamma, t_max_edge, result, time_bound)

            # # append data and result
            # files = {"density100_result_cap": [result, ""]}

            # for filename in list(files.keys()):
            #     append_record(files[filename][0], filename, typ=files[filename][1])

            # increase the task_num
            t_max_edge += 1
        
        break


if __name__ == "__main__":
    main()