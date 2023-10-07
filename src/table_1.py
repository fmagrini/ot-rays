#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:40:14 2023

@author: fabrizio
"""

from time import time
import gc
import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict
import itertools as it
from raytracing_ot import get_finite_idx_and_cost
import numpy as np
from scipy.sparse import lil_array, csr_array
from scipy.sparse.csgraph import dijkstra, bellman_ford
from scipy.optimize import curve_fit
from seislib.utils import save_pickle, load_pickle
import matplotlib.pyplot as plt
from docplex.mp.model import Model
plt.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern Roman'


MESH_SHAPES = [(20, 20),
               (40, 40),
               (60, 60),
               (80, 80),
               (100, 100),
               (130, 130),
               (160, 160),
               (200, 200),
               (250, 250)]
SAVE_RESULTS = '../data/comptimes'
N_SIMULATIONS = 5
RECOMPUTE = False



def ravel_index(index_2d, shape_2d):
    index_2d = np.asarray(index_2d, dtype=int)
    row, col = index_2d.T
    return row * shape_2d[1] + col
 

def get_masses(isource, ireceivers, size):
    mass_excess = ireceivers.size - 1
    mass_source = np.ones(size) + mass_excess
    mass_target = np.ones(size) + mass_excess
    for ireceiver in ireceivers:
        mass_source[isource] += 1
        # mass_target[ireceiver] += 1
        mass_target[ireceiver] += 1
    return mass_source, mass_target


def estimate_complexity(V, E, times):
    # Define models for different complexity classes
    def constant(x, a):
        return a

    def V_model(x, a, b):
        v, e = x
        return a * v + b

    def E_model(x, a, b):
        v, e = x
        return a * e + b

    def VE_model(x, a, b):
        v, e = x
        return a * v * e + b

    def V2_model(x, a, b):
        v, e = x
        return a * v**2 + b

    def E2_model(x, a, b):
        v, e = x
        return a * e**2 + b

    def V_logV_model(x, a, b):
        v, e = x
        return a * v * np.log(v) + b

    def E_logE_model(x, a, b):
        v, e = x
        return a * e * np.log(e) + b

    def VE_logV_model(x, a, b):
        v, e = x
        return a * v * e * np.log(v) + b

    def VE_logE_model(x, a, b):
        v, e = x
        return a * v * e * np.log(e) + b

    def V_ElogV_model(x, a, b):  # Dijkstra with Binary Heap
        v, e = x
        return a * (v + e) * np.log(v) + b

    def E_VlogV_model(x, a, b):  # Dijkstra with Fibonacci Heap
        v, e = x
        return a * (e + v * np.log(v)) + b

    # List of models and their names
    models = [constant, V_model, E_model, VE_model, V2_model, E2_model, 
              V_logV_model, E_logE_model, VE_logV_model, VE_logE_model,
              V_ElogV_model, E_VlogV_model]
    model_names = ["O(1)", "O(V)", "O(E)", "O(VE)", "O(V^2)", "O(E^2)", 
                   "O(V log V)", "O(E log E)", "O(VE log V)", "O(VE log E)",
                   "O((V + E) log V)", "O(E + V log V)"]
    
    V = np.array(V)
    E = np.array(E)
    avg_times = np.mean(times, axis=0)
    best_fit = float('inf')
    best_model = None
    best_model_name = None

    for model, name in zip(models, model_names):
        # Fit the model to the data
        try:
            params, _ = curve_fit(model, (V, E), avg_times)
            predicted_times = model((V, E), *params)
            mse = np.mean((avg_times - predicted_times)**2)  # Mean squared error
            if mse < best_fit:
                best_fit = mse
                best_model = (model, params)
                best_model_name = name
        except:
            # If curve fitting fails for a model, continue to the next one
            continue

    return best_model, best_model_name, best_fit


def solve_ot(mesh_shape, 
             source_index, 
             receiver_indexes, 
             cost_matrix,
             idx_2d):
    
    mass_excess = len(receiver_indexes)
    a, b = get_masses(
        isource=ravel_index(source_index, mesh_shape), 
        ireceivers=ravel_index(receiver_indexes, mesh_shape),
        size=cost_matrix.shape[0]
        )
    mdl = Model(name='OT')
    variable = mdl.continuous_var
    
    x = {(i, j): variable(name=f"x_{i}_{j}", lb=0, ub=mass_excess+1) \
         for i, j in zip(*idx_2d)}
    
    # Flow conservation constraints
    i_to_js = defaultdict(list)
    j_to_is = defaultdict(list)
    for (i, j) in zip(*idx_2d):
        i_to_js[i].append(j)
        j_to_is[j].append(i)   
    
    # Batch add constraints
    constraints = []
    for i, js in i_to_js.items():
        constraints.append(mdl.sum(x[i, j] for j in js) == a[i])
            
    for j, is_ in j_to_is.items():
        constraints.append(mdl.sum(x[i, j] for i in is_) == b[j])
        
    mdl.add_constraints(constraints)
    mdl.minimize(mdl.sum(cost_matrix[i, j] * x[i, j] for (i, j) in x.keys()))    
    mdl.parameters.emphasis.mip.set(1)
    
    t1 = time()
    mdl.solve()
    t2 = time()
            
    ot_plan = lil_array(cost_matrix.shape) 
    for (i, j), var in x.items():
        if var.solution_value:
            ot_plan[i, j] = var.solution_value    
    
    return csr_array(ot_plan), t2 - t1


def solve_dijkstra(mesh_shape, 
                   source_index, 
                   graph):

    isource = np.int32(ravel_index(source_index, mesh_shape))
    
    t1 = time()
    arrivals, predecessors = dijkstra(graph, 
                                      indices=isource,
                                      return_predecessors=True)
    t2 = time()    
    return arrivals, predecessors, t2 - t1


def solve_bellman_ford(mesh_shape, 
                       source_index, 
                       graph):

    isource = np.int32(ravel_index(source_index, mesh_shape))
    
    t1 = time()
    arrivals, predecessors = bellman_ford(graph, 
                                          indices=isource,
                                          return_predecessors=True)
    t2 = time()    
    return arrivals, predecessors, t2 - t1


def get_travel_times(cost_matrix, predecessors):
    travel_times = np.zeros(predecessors.shape)
    for inode in range(predecessors.size):
        if travel_times[inode] != 0:
            continue
        nodes_seen = []
        early_finish = False
        while predecessors[inode] != -9999 and not early_finish:
            nodes_seen.append(inode)
            predecessor = predecessors[inode]
            t = cost_matrix[predecessor, inode]
            if travel_times[predecessor] != 0:
                early_finish = True
                t += travel_times[predecessor]
            travel_times[nodes_seen] += t
            inode = predecessor
    
    return travel_times


def get_predecessors(cost_matrix, ot_plan):
    predecessors = np.full(cost_matrix.shape[0], -9999)
    indexes = np.array([(i, j) for i, j in zip(*ot_plan.nonzero()) if i!=j])
    for i, j in indexes:
        predecessors[j] = i
    return predecessors


def get_path(inode, predecessors):
    predecessor = predecessors[inode]
    if predecessor < 0:
        return [inode]
    return [inode] + get_path(predecessor, predecessors)


def get_travel_cost(path, cost_matrix):
    cost = 0
    for i in range(len(path)-1):
        cost += cost_matrix[path[i], path[i+1]]
    return cost


def get_cpu_times(mesh_shapes):
    results = {}
    for mesh_shape in mesh_shapes:
        n_pixels = mesh_shape[0] * mesh_shape[1]
        single_receiver = [(0, mesh_shape[0]-1)]
        all_receivers = list(it.product(range(mesh_shape[0]),
                                        range(mesh_shape[1])))
        source_index = (mesh_shape[0]-1, 0)
        cost_matrix_shape = (n_pixels,) * 2
        
        velocity = np.random.uniform(0.5, 5, n_pixels)
        slowness = 1 / velocity
        idx, cost = get_finite_idx_and_cost(mesh_shape, slowness)
        idx_2d = np.int32(np.unravel_index(idx, cost_matrix_shape))
        cost_matrix = csr_array((cost, idx_2d))

        i = [i for i in range(cost.size) if idx_2d[0][i] != idx_2d[1][i]]
        graph = csr_array((cost[i], (idx_2d[0][i], idx_2d[1][i])))

        ot_plan_single, t_ot_single = solve_ot(mesh_shape, 
                                               source_index, 
                                               single_receiver,
                                               cost_matrix,
                                               idx_2d)
        ot_plan_all, t_ot_all = solve_ot(mesh_shape, 
                                         source_index, 
                                         all_receivers,
                                         cost_matrix,
                                         idx_2d)
        
        pred_ot_single = get_predecessors(cost_matrix, ot_plan_single)
        pred_ot_all = get_predecessors(cost_matrix, ot_plan_all)
        arrivals_ot_all = get_travel_times(cost_matrix, pred_ot_all)
        
        arrivals_dijkstra, pred_dijkstra, t_dijkstra = solve_dijkstra(mesh_shape, 
                                                                      source_index, 
                                                                      graph)
        arrivals_bell, pred_bell, t_bell = solve_bellman_ford(mesh_shape, 
                                                              source_index, 
                                                              graph)
        
        ireceiver = ravel_index(single_receiver[0], mesh_shape)
        cost_ot = get_travel_cost(get_path(ireceiver, pred_ot_single), cost_matrix)
        cost_dijkstra = get_travel_cost(get_path(ireceiver, pred_dijkstra), cost_matrix)
        cost_bell = get_travel_cost(get_path(ireceiver, pred_bell), cost_matrix)
        assert np.isclose(cost_ot, cost_dijkstra)
        assert np.isclose(cost_ot, cost_bell)
        assert np.allclose(arrivals_ot_all, arrivals_dijkstra)
        assert np.allclose(arrivals_ot_all, arrivals_bell)
        
        print(mesh_shape, 
              'OT single: %.5fs OT all: %.5fs Dijkstra: %.5fs Bell: %.5fs' % (
                  t_ot_single, t_ot_all, t_dijkstra, t_bell
                  )
              )
        
        edges = cost_matrix.count_nonzero()
        nodes = cost_matrix.shape[0]
        results[mesh_shape] = {'V': nodes,
                               'E': edges,
                               't_dijkstra': t_dijkstra,
                               't_bell': t_bell,
                               't_ot_single': t_ot_single,
                               't_ot_all': t_ot_all
                               }
        del cost_matrix
        gc.collect()
    
    return results
    
    
def get_complexity(cpu_times):
    results = {}
    nodes, edges = get_nodes_and_edges(cpu_times)
    for kind in ['t_dijkstra', 't_bell', 't_ot_single', 't_ot_all']:
        times = np.array([v[kind] for _, v in sorted(cpu_times.items())]).T
        results[kind] = estimate_complexity(nodes, edges, times)
    return results
        
    
def get_nodes_and_edges(cpu_times):
    nodes = np.array([v['V'] for k, v in sorted(cpu_times.items())])
    edges = np.array([v['E'] for k, v in sorted(cpu_times.items())])
    return nodes, edges


def get_ensemble_cpu_times(directory, n_partials=10):
    results = {}
    for i in range(n_partials):
        results_i = load_pickle(os.path.join(directory, f'cpu_times_{i}.pickle'))
        for mesh_shape, partial_results in results_i.items():
            if mesh_shape not in results:
                results[mesh_shape] = defaultdict(list)
                results[mesh_shape]['V'] = partial_results['V']
                results[mesh_shape]['E'] = partial_results['E']
            else:
                for k, v in partial_results.items():
                    if k not in ['V', 'E']:
                        results[mesh_shape][k].append(v)
    return results
#%%

if RECOMPUTE:
    for i in range(N_SIMULATIONS):
        cpu_times_partial = get_cpu_times(MESH_SHAPES)
        save_pickle(os.path.join(SAVE_RESULTS, f'cpu_times_{i}.pickle'), 
                    cpu_times_partial)
    cpu_times = get_ensemble_cpu_times(SAVE_RESULTS, n_partials=N_SIMULATIONS)
    
else:
    cpu_times = get_ensemble_cpu_times(SAVE_RESULTS, n_partials=N_SIMULATIONS)
#%%

legend_dict = {'t_dijkstra': 'Dijkstra',
               't_bell': 'Bellman-Ford',
               't_ot_single': 'OT (single)',
               't_ot_all': 'OT (full)'
               }
colors_dict = {'t_dijkstra': 'orange',
               't_bell': 'red',
               't_ot_single': 'blue',
               't_ot_all': 'green'
               }
complexity_dict = get_complexity(cpu_times)

fig = plt.figure(figsize=(8, 6), dpi=300)
ax1 = fig.add_subplot(1, 1, 1)

nodes, edges = get_nodes_and_edges(cpu_times)
for kind in ['t_dijkstra', 't_bell', 't_ot_single', 't_ot_all']:
    times = np.array([v[kind] for k, v in sorted(cpu_times.items())]).T
    avg_times = np.mean(times, axis=0)
    (model, params), complexity, l2 = complexity_dict[kind]
    label_complexity = r'$\mathcal{O}(%s)$'%(complexity.split('O(')[1][:-1])
    label = legend_dict[kind]
    color = colors_dict[kind]
    pred_data = model((nodes, edges), *params)
    ax1.plot(nodes*edges, avg_times, color=color, marker='o', lw=0, label=label)
    ax1.plot(nodes*edges, pred_data, color=color, ls='-', label=label_complexity)
    
ax1.set_ylabel('CPU Time [s]')
ax1.set_xlabel(r'$V \times E$')
ax1.grid()
ax1.tick_params(direction='in')
plt.legend()

plt.tight_layout()
plt.show()


for kind in ['V', 'E', 't_dijkstra', 't_bell', 't_ot_all']:
    print(kind, end='\t')
print()
for mesh_shape, results in cpu_times.items():
    for kind in ['V', 'E', 't_dijkstra', 't_bell', 't_ot_all']:
       if kind in ['V', 'E']:
           print(results[kind], end='\t')
       else:
           avg = np.mean(results[kind]) * 1000
           std = np.std(results[kind]) * 1000
           print(r'%.2f +- %.2f'%(avg, std), end='\t')
    print()
    








