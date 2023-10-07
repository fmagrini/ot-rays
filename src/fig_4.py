#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 00:59:46 2023

@author: fabrizio

Download vp.bin at https://www.geoazur.fr/WIND/bin/view/Main/Data/Marmousi
"""

import os
import sys
sys.path.append(os.getcwd())
import itertools as it
from collections import defaultdict
from raytracing_ot import get_finite_idx_and_cost
import numpy as np
import seislib.colormaps as scm
from docplex.mp.model import Model
from scipy.sparse import csr_array, lil_array
from scipy.sparse.csgraph import dijkstra
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from matplotlib.colorbar import ColorbarBase
import matplotlib.ticker as mticker 
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath,newtxtext,newtxmath}'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 20


MARMOUSI = '/path/to/vp.bin'
SOURCE_INDEX = [650, 100]
RECEIVER_INDEXES = None
DOWNSAMPLE = 2 # Save memory and speed up computations


def ravel_index(index_2d, shape_2d):
    index_2d = np.asarray(index_2d, dtype=int)
    row, col = index_2d.T
    return row * shape_2d[1] + col


def get_marmousi(file):
    # MARMOUSI INFO: https://www.geoazur.fr/WIND/bin/view/Main/Data/Marmousi
    n1 = 751
    n2 = 2301
    d1 = 4
    d2 = 4
    shape = (2301, 751)  # n2 x n1
    with open(file, 'rb') as f:
        velocity = np.fromfile(f, dtype=np.float32).reshape(shape).T
    depth = np.arange(0, n1 * d1, d1)
    distance = np.arange(0, n2 * d2, d2)
    return distance, depth, velocity


def label_format(x, pos): # your custom formatter function: divide by 100.0
    s = '{}'.format(x / 1000)
    return s


def plot_marmousi(ax, velocity_model, fig=None, **kwargs):
    fmt = mticker.FuncFormatter(label_format)  
    shape = velocity_model.T.shape
    # Visualize the velocity model
    img = ax.imshow(velocity_model, 
                    extent=[0, shape[0]*4*DOWNSAMPLE, shape[1]*4*DOWNSAMPLE, 0], 
                    **kwargs)
    cbar = make_colorbar(ax, fig=fig, img=img, orientation='vertical')
    cbar.set_label('Velocity [km/s]', fontsize=22, labelpad=10)
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel('Distance [km]', labelpad=10)
    ax.set_ylabel('Depth [km]', labelpad=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='both'))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune='both'))
    return img, cbar
    
    
def make_colorbar(ax, fig=None, img=None, cmap=None, norm=None, orientation='vertical'):
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="2%", pad=0.1)
    if img is None:
        cbar = ColorbarBase(cbar_ax, 
                            cmap=cmap, 
                            norm=norm, 
                            orientation=orientation)
    else:
        cbar = fig.colorbar(img, 
                            cax=cbar_ax, 
                            norm=norm, 
                            orientation=orientation)
    return cbar


def plot_source_receivers(ax, x, y, source, receivers=None):
    color = 'r'
    ax.scatter(x[source[1]], 
               y[source[0]], 
               color=color, 
               marker='*', 
               s=600,
               ec='k')
    if receivers is not None:
        for receiver in receivers:
            ax.scatter(x[receiver[1]], 
                       y[receiver[0]], 
                       color=color, 
                       marker='^', 
                       s=180,
                       ec='k')


def get_masses(isource, ireceivers, size):
    mass_excess = ireceivers.size - 1
    mass_source = np.ones(size) + mass_excess
    mass_target = np.ones(size) + mass_excess
    for ireceiver in ireceivers:
        mass_source[isource] += 1
        mass_target[ireceiver] += 1
    return mass_source, mass_target


def solve_ot(mesh_shape, source_index, receiver_indexes, slowness=None):
    
    print('Getting Cost Matrix and Mass Vectors')
    cost_matrix_shape = (mesh_shape[0] * mesh_shape[1],) * 2
    idx, cost = get_finite_idx_and_cost(mesh_shape, slowness)
    idx_2d = np.unravel_index(idx, cost_matrix_shape)
    cost_matrix = csr_array((cost, idx_2d))
    
    mass_excess = len(receiver_indexes)
    a, b = get_masses(
        isource=ravel_index(source_index, mesh_shape), 
        ireceivers=ravel_index(receiver_indexes, mesh_shape),
        size=cost_matrix.shape[0]
        )
    mdl = Model(name='OT')
    variable = mdl.continuous_var
    
    print('Defining DOcplex Variables')
    x = {(i, j): variable(name=f"x_{i}_{j}", lb=0, ub=mass_excess+1) \
         for i, j in zip(*idx_2d)}
    
    # Flow conservation constraints
    i_to_js = defaultdict(list)
    j_to_is = defaultdict(list)
    for (i, j) in zip(*idx_2d):
        i_to_js[i].append(j)
        j_to_is[j].append(i)   
    
    print('Adding LP Constraints')        
    # Batch add constraints
    constraints = []
    for i, js in i_to_js.items():
        constraints.append(mdl.sum(x[i, j] for j in js) == a[i])
            
    for j, is_ in j_to_is.items():
        constraints.append(mdl.sum(x[i, j] for i in is_) == b[j])
        
    mdl.add_constraints(constraints)
    mdl.minimize(mdl.sum(cost_matrix[i, j] * x[i, j] for (i, j) in x.keys()))
    mdl.parameters.emphasis.mip.set(1)
    
    mdl.solve()

    ot_plan = lil_array(cost_matrix.shape) 
    for (i, j), var in x.items():
        if var.solution_value:
            ot_plan[i, j] = var.solution_value    
    
    return cost_matrix, csr_array(ot_plan)


def get_path(inode, predecessors):
    predecessor = predecessors[inode]
    if predecessor < 0:
        return [inode]
    return [inode] + get_path(predecessor, predecessors)
    

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
        

def solve_djikstra(mesh_shape, 
                   source_index, 
                   receiver_indexes, 
                   slowness=None):

    cost_matrix_shape = (mesh_shape[0] * mesh_shape[1],) * 2
    idx, cost = get_finite_idx_and_cost(mesh_shape, slowness)
    idx = np.unravel_index(idx, cost_matrix_shape)
    i = [i for i in range(cost.size) if idx[0][i] != idx[1][i]]
    graph = csr_array((cost[i], (np.int32(idx[0][i]), np.int32(idx[1][i]))))
    isource = np.int32(ravel_index(source_index, mesh_shape))
    
    arrivals, predecessors = dijkstra(graph, 
                                      indices=isource,
                                      return_predecessors=True)
    return graph, arrivals, predecessors
#%%

distance, depth, velocity = get_marmousi(MARMOUSI)
velocity = velocity[::DOWNSAMPLE][:, ::DOWNSAMPLE]
distance = distance[::DOWNSAMPLE]
depth = depth[::DOWNSAMPLE]
slowness = (1 / velocity).ravel().astype(float)
source_index = SOURCE_INDEX[0] // DOWNSAMPLE, SOURCE_INDEX[1] // DOWNSAMPLE
if RECEIVER_INDEXES is None:
    receiver_indexes = list(it.product(range(velocity.shape[0]),
                                       range(velocity.shape[1])))
else:
    receiver_indexes = []
    for receiver in RECEIVER_INDEXES:
        receiver_indexes.append(
            (receiver[0]//DOWNSAMPLE, receiver[1]//DOWNSAMPLE)
            )
        
mesh_shape = velocity.shape
#%%
fig = plt.figure(figsize=(15, 7), dpi=200)
ax = fig.add_subplot(1, 1, 1)
plot_marmousi(ax, velocity, fig=fig, cmap=scm.grayC)
plt.show()
#%%

cost_matrix, ot_plan = solve_ot(mesh_shape, 
                                source_index, 
                                receiver_indexes,
                                slowness)
predecessors_ot = get_predecessors(cost_matrix, ot_plan)
travel_times_ot = get_travel_times(cost_matrix, predecessors_ot)
cost_matrix_dj, travel_times_dj, predecessors_dj = solve_djikstra(mesh_shape, 
                                                                  source_index, 
                                                                  receiver_indexes,
                                                                  slowness)
coords = np.zeros((velocity.size, 2))
for inode, (i, j) in enumerate(np.ndindex(velocity.shape)):
    coords[inode] = [j*4*DOWNSAMPLE, i*4*DOWNSAMPLE]

assert np.allclose(travel_times_ot, travel_times_dj)

#%%

fig = plt.figure(figsize=(15, 9.5), dpi=200)
ax1 = fig.add_subplot(2, 1, 1)
img1, cb1 = plot_marmousi(ax1, 1 / (velocity/1000), fig=fig, cmap=scm.grayC)
ax1.set_xlabel('')
ax1.tick_params(labelbottom=False)
ax1.text(x=-0.08, 
         y=0.98, 
         s=r'\textbf{(a)}', 
         transform=ax1.transAxes, 
         fontsize=25)
cb1.set_label('Slowness [s/km]', labelpad=10)
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(2, 1, 2)
img2, cb2 = plot_marmousi(ax2, 
                          travel_times_ot.reshape(velocity.shape), 
                          fig=fig, 
                          cmap='turbo')
contour = ax2.contour(distance, 
                    depth, 
                    travel_times_ot.reshape(velocity.shape),
                    levels=10,
                    colors='k',
                    alpha=0.5,
                    linestyles=[(5, (5, 10))],
                    linewidths=1)
ax2.clabel(contour, inline=True, fontsize=15, inline_spacing=5)

cb2.set_label('Travel Time [s]', labelpad=10)
plot_source_receivers(ax2, 
                      distance, 
                      depth, 
                      source_index)
ax2.text(x=-0.08, 
         y=0.98, 
         s=r'\textbf{(b)}', 
         transform=ax2.transAxes, 
         fontsize=25)
ax2.grid(alpha=0.3)
plt.tight_layout(h_pad=0.75)
plt.show()

