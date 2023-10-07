#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:40:14 2023

@author: fabrizio
"""

import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import bellman_ford
from raytracing_ot import get_finite_idx_and_cost
import seislib.colormaps as scm
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath,newtxtext,newtxmath}'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 20

TOPOGRAPHY = '../data/topo_small.npy'
MESH_SHAPE = (6, 6)
SOURCE_INDEX = (0, 0)
RECEIVERS_INDEXES = [(5, 5)]



def get_cost_from_topo(x, y, topo, E0=1.5, k_up=1, k_down=0.1):
    """
    Parameters
    ----------
    x, y : ndarray
        x and y coordinates
    topo : ndarray
        Topography
    E0 : float
        Energy required to travel a unit of distance in the absence of
        topographic gradient (flat surface).
    k : float
        Multiplicative factor used to calculate the energy required to
        travel in the presence of topographic gradient (g),
            E = E0(1 + g * k)
        Note that, while traveling downhill, E can be negative.

    Returns
    -------
    cost_matrix : ndarray
        Transportation cost matrix
    """
    mesh_shape = topo.shape
    idx, _ = get_finite_idx_and_cost(mesh_shape)
    cost_matrix_shape = (topo.shape[0] * mesh_shape[1],) * 2
    idx_2d = np.array(np.unravel_index(idx, cost_matrix_shape)).T
    cost_matrix = np.full(cost_matrix_shape, np.inf) # Energy required to travel
    
    for icost, (i, j) in enumerate(idx_2d):
        if i == j:
            cost_matrix[i, j] = 0
            continue
        ix, iy = np.unravel_index(i, mesh_shape)
        jx, jy = np.unravel_index(j, mesh_shape)
        
        dist_x = abs(x[i] - x[j])
        dist_y = abs(y[i] - y[j])
        dist = np.sqrt(dist_x**2 + dist_y**2)
        g = (topo[jx, jy] - topo[ix, iy]) / dist # topographic gradient
        if g >= 0:
            cost_matrix[i, j] = E0 * dist + g * k_up
        else:
            cost_matrix[i, j] = E0 * dist + g * k_down

    return cost_matrix


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def shiftedColorMap(cmap, 
                    start=0, 
                    midpoint=0.5, 
                    stop=1.0, 
                    name='shiftedcmap'):

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    # plt.register_cmap(cmap=newcmap)

    return newcmap


def ravel_index(index_2d, shape_2d):
    index_2d = np.asarray(index_2d, dtype=int)
    row, col = index_2d.T
    return row * shape_2d[1] + col
    

def sort_matrix(matrix):
    """
    Sort matrix for visualization purposes, so as to map pixel 1 onto the upper 
    left. (Its original position is the lower right.)
    """    
    original = np.arange(matrix.shape[0]).reshape(MESH_SHAPE)[::-1]
    new = np.arange(matrix.shape[0]).reshape(MESH_SHAPE)
    mapping = dict(zip(original.flatten(), new.flatten()))
    
    matrix_sorted = np.zeros(matrix.shape)
    for irow in range(matrix.shape[0]):
        new_row = mapping[irow]
        for icol in range(matrix.shape[1]):
            new_col = mapping[icol]
            matrix_sorted[new_row, new_col] = matrix[irow, icol]
    return matrix_sorted


def make_colorbar(ax, img=None, cmap=None, norm=None):
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("bottom", size="4%", pad=0.1)
    if img is None:
        cbar = ColorbarBase(cbar_ax, 
                            cmap=cmap, 
                            norm=norm, 
                            orientation='horizontal')
    else:
        cbar = fig.colorbar(img, cax=cbar_ax, norm=norm, orientation='horizontal')
    return cbar
    

def plot_mass_transport(ax, x_coords, y_coords, ot_plan, colorbar=True):
    
    cmap_two_values = ListedColormap([(0, 0, 0, 0.3), 'black'])
    cmap_four_values = ListedColormap([(0, 0, 0, 0.3), 
                                       'yellow',
                                       'r', 
                                       'k'])
    
    unique_values = np.sort(np.unique(ot_plan))
    n_values = unique_values.size
    
    if n_values == 2:
        cmap = cmap_two_values
    elif n_values == 4:
        cmap = cmap_four_values
    else:
        raise NotImplementedError
    
    for row, col in np.argwhere(ot_plan):
        if row == col:
            continue
        x_start, y_start = x_coords[row], y_coords[row]
        x_end, y_end = x_coords[col], y_coords[col]
        gap = 0.3
        if x_start != x_end and y_start != y_end:
            gap = 0.22
        if x_end > x_start:
            x1 = x_start + gap
            x2 = x_end - gap
        elif x_end < x_start:
            x1 = x_start - gap
            x2 = x_end + gap
        else:
            x1 = x_start
            x2 = x_end
        if y_end > y_start:
            y1 = y_start + gap
            y2 = y_end - gap
        elif y_end < y_start:
            y1 = y_start - gap
            y2 = y_end + gap
        else:
            y1 = y_start
            y2 = y_end
        
        color = cmap(int(ot_plan[row, col]))
        # print(ot_plan[row, col], row, col, color)
        ax.arrow(x1, 
                 y1, 
                 (x2-x1) * 0.8, 
                 (y2-y1) * 0.8,
                 head_width=0.18,
                 head_length=1. * 0.15,
                 width=0.08,
                 ec='k',
                 color=color,
                 zorder=1000)
    
    if colorbar:
        norm = plt.Normalize(vmin=ot_plan.min(), vmax=ot_plan.max())
        cbar = make_colorbar(ax, img=None, norm=norm, cmap=cmap)
        cbar.set_label('Mass Transport', fontsize=22, labelpad=10)
        # Compute tick positions to center them within their respective regions
        tick_positions = (np.arange(n_values) * 2 + 1) / (2 * n_values) * ot_plan.max()
        
        # Set the ticks and labels on the colorbar
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(unique_values.astype(str))
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(unique_values.astype(str))


def plot_path(ax, x_coords, y_coords, path, colorbar=True, color='k'):
        
    path = path[::-1]
    for i in range(len(path)-1):
        start = path[i]
        end = path[i+1]
        x_start, y_start = x_coords[start], y_coords[start]
        x_end, y_end = x_coords[end], y_coords[end]
        gap = 0.3
        if x_start != x_end and y_start != y_end:
            gap = 0.22
        if x_end > x_start:
            x1 = x_start + gap
            x2 = x_end - gap
        elif x_end < x_start:
            x1 = x_start - gap
            x2 = x_end + gap
        else:
            x1 = x_start
            x2 = x_end
        if y_end > y_start:
            y1 = y_start + gap
            y2 = y_end - gap
        elif y_end < y_start:
            y1 = y_start - gap
            y2 = y_end + gap
        else:
            y1 = y_start
            y2 = y_end

        ax.arrow(x1, 
                 y1, 
                 (x2-x1) * 0.8, 
                 (y2-y1) * 0.8,
                 head_width=0.18,
                 head_length=1. * 0.15,
                 width=0.08,
                 ec='k',
                 color=color,
                 zorder=1000)
    
    
def get_masses(isource, ireceivers, size):
    mass_excess = ireceivers.size - 1
    mass_source = np.ones(size) + mass_excess
    mass_target = np.ones(size) + mass_excess
    for ireceiver in ireceivers:
        mass_source[isource] += 1
        mass_target[ireceiver] += 1
    return mass_source, mass_target


def solve_ot(cost_matrix,
             mesh_shape, 
             source_index, 
             receivers_indexes):

    mass_excess = len(receivers_indexes)
    a, b = get_masses(
        isource=ravel_index(source_index, mesh_shape), 
        ireceivers=ravel_index(receivers_indexes, mesh_shape),
        size=cost_matrix.shape[0]
        )
    mdl = Model(name='OT')
    variable = mdl.continuous_var
    
    finite_indices = np.argwhere(np.isfinite(cost_matrix))
    x = {(i, j): variable(name=f"x_{i}_{j}", lb=0, ub=mass_excess+1) for i, j in finite_indices}
    
    # Flow conservation constraints
    i_to_js = defaultdict(list)
    j_to_is = defaultdict(list)
    for (i, j) in finite_indices:
        i_to_js[i].append(j)
        j_to_is[j].append(i)   
            
    # Add constraints
    constraints = []
    for i, js in i_to_js.items():
        constraints.append(mdl.sum(x[i, j] for j in js) == a[i])            
    for j, is_ in j_to_is.items():
        constraints.append(mdl.sum(x[i, j] for i in is_) == b[j])    
    mdl.add_constraints(constraints)
    
    # Objective function
    mdl.minimize(mdl.sum(cost_matrix[i, j] * x[i, j] for (i, j) in x.keys()))    
    mdl.solve()

    ot_plan = np.zeros(cost_matrix.shape) # Get ot_plan
    for (i, j), var in x.items():
        ot_plan[i, j] = var.solution_value

    return cost_matrix, ot_plan
    
    
def solve_bellman_ford(cost_matrix, 
                       mesh_shape,
                       source_index, 
                       receivers_indexes, 
                       slowness=None):

    finite_indices = np.argwhere(np.isfinite(cost_matrix))
    cost = np.array([cost_matrix[i, j] for i, j in finite_indices])
    graph = csr_array((cost, [tuple(np.int32(i)) for i in finite_indices.T]))
    isource = ravel_index(source_index, mesh_shape)
    
    arrivals, predecessors = bellman_ford(graph, 
                                          indices=isource,
                                          return_predecessors=True)
    
    return arrivals, predecessors
    

def plot_topography(ax, 
                    X, 
                    Y, 
                    topo_2d, 
                    panel_label, 
                    colorbar=True,
                    enumerate_pixels=True,
                    tick_params_dict={}):
    cmap = truncate_colormap(scm.bamako, 0.2, 1)
    img = ax.pcolormesh(X, Y, topo_2d, cmap=cmap)
    ax.set_aspect('equal')
    labels = (x_edges + 0.5).astype(int)[:-1]
    ax.set_xticks(labels)
    ax.set_xticklabels([r'$x_%d$'%i for i in labels])
    ax.set_yticks(labels[::-1])
    ax.set_yticklabels([r'$y_%d$'%i for i in labels])

    ax.tick_params(**tick_params_dict)
    
    if colorbar:
        cbar = make_colorbar(ax, img)
        cbar.set_label('Topography [m]', fontsize=22, labelpad=10)

    ax.text(x=-0.08, 
            y=1.02, 
            s=r'\textbf{(%s)}'%panel_label, 
            transform=ax.transAxes, 
            fontsize=25)
    if enumerate_pixels:
        for ipixel, (x, y) in enumerate(zip(x_coords, y_coords[::-1]), 1):
            ax.text(x=x+0.42, 
                    y=y-0.44, 
                    s=r'%d'%ipixel, 
                    va='bottom',
                    ha='right', 
                    fontsize=16)
        
    ax.set_aspect('equal')


def plot_cost_matrix(ax,
                     cost_matrix, 
                     panel_label, 
                     colorbar=True,
                     tick_params_dict={}):
    X, Y = np.meshgrid(np.arange(cost_matrix.shape[0]),
                       np.arange(cost_matrix.shape[0]))
    cost_matrix_sorted = sort_matrix(cost_matrix)
    masked = np.ma.masked_where(~np.isfinite(cost_matrix_sorted), cost_matrix_sorted)
    
    vmin, vmax = masked.min(), masked.max()
    abs_max = max(abs(vmax), abs(vmin))
    abs_min = min(abs(vmax), abs(vmin))
    cmap = shiftedColorMap(mpl.cm.RdYlBu, 
                           start=1 - (abs_min + abs_max) / (abs_max * 2),
                           # midpoint=abs(vmin / (vmax - vmin)),
                           stop=1)
    
    # cmap = scm.bam.copy()
    cmap.set_bad('k', 0.3)
    img = ax.pcolormesh(X, Y[::-1], masked, cmap=cmap)
    ax.set_aspect('equal')
    ax.tick_params(**tick_params_dict)
    
    labels = np.arange(2, cost_matrix.shape[0]+1, 3)
    ax.set_xticks(labels-1)
    ax.set_xticklabels(labels)
    ax.set_yticks(cost_matrix.shape[0] - labels)
    ax.set_yticklabels(labels)
    
    if colorbar:
        cbar = make_colorbar(ax, img)
        cbar.set_label(r"Cost Matrix ${\bf C^{'}}$", fontsize=22, labelpad=10)
    ax.text(x=-0.08, 
            y=1.02, 
            s=r'\textbf{(%s)}'%panel_label, 
            transform=ax.transAxes, 
            fontsize=25)
    
    
def plot_ot_plan(ax,
                 ot_plan,
                 panel_label, 
                 colorbar=True,
                 tick_params_dict={}):
    
    X, Y = np.meshgrid(np.arange(ot_plan.shape[0]),
                       np.arange(ot_plan.shape[0]))
    
    cmap_two_values = ListedColormap([(0, 0, 0, 0.3), 'black'])
    cmap_four_values = ListedColormap([(0, 0, 0, 0.3), 
                                       'yellow',
                                       'r', 
                                       'k'])
    
    unique_values = np.sort(np.unique(ot_plan))
    n_values = unique_values.size
    
    if n_values == 2:
        cmap = cmap_two_values
    elif n_values == 4:
        cmap = cmap_four_values
    else:
        raise NotImplementedError
    
    ot_plan_sorted = sort_matrix(ot_plan)
    img = ax.pcolormesh(X, Y[::-1], ot_plan_sorted, cmap=cmap)
    ax.set_aspect('equal')
    ax.tick_params(**tick_params_dict)
    
    labels = np.arange(2, ot_plan.shape[0]+1, 3)
    ax.set_xticks(labels-1)
    ax.set_xticklabels(labels)
    ax.set_yticks(ot_plan.shape[0] - labels)
    ax.set_yticklabels(labels)
    
    if colorbar:
        cbar = make_colorbar(ax, img)
        cbar.set_label(r'Optimal Coupling ${\bf P^*}$', fontsize=22, labelpad=10)
        # Compute tick positions to center them within their respective regions
        tick_positions = (np.arange(n_values) * 2 + 1) / (2 * n_values) * ot_plan.max()
        
        # Set the ticks and labels on the colorbar
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(unique_values.astype(str))
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(unique_values.astype(str))
        
    ax.text(x=-0.08, 
            y=1.02, 
            s=r'\textbf{(%s)}'%panel_label, 
            transform=ax.transAxes, fontsize=25)


def plot_nodes(ax, x_coords, y_coords, source=None, receivers=None):
    color = 'yellow'
    ax.scatter(x_coords, y_coords, color='k', marker='o')
    if source is not None:
        source = source[1] + 1, source[0] + 1 
        ax.scatter(source[0], 
                   source[1], 
                   color=color, 
                   marker='*', 
                   s=960,
                   ec='k')
    if receivers is not None:
        for receiver in receivers:
            receiver = receiver[1] + 1, receiver[0] + 1
            ax.scatter(receiver[0], 
                       receiver[1], 
                       color=color, 
                       marker='^', 
                       s=480,
                       ec='k')


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
#%%

x_source = np.zeros((MESH_SHAPE[0] * MESH_SHAPE[1], 2))
for inode, (i, j) in enumerate(np.ndindex(MESH_SHAPE)):
    x_source[inode] = [j, i]

topo = np.load(TOPOGRAPHY)

cost_matrix_1 = get_cost_from_topo(*x_source.T, topo, E0=5, k_up=0.15, k_down=0.1)


ot_plan_1 = solve_ot(cost_matrix=cost_matrix_1,
                     mesh_shape=MESH_SHAPE, 
                     source_index=SOURCE_INDEX, 
                     receivers_indexes=RECEIVERS_INDEXES)[1]

_, predecessors_bell = solve_bellman_ford(cost_matrix_1,
                                          mesh_shape=MESH_SHAPE, 
                                          source_index=SOURCE_INDEX, 
                                          receivers_indexes=RECEIVERS_INDEXES)
ireceiver = ravel_index(RECEIVERS_INDEXES[0], MESH_SHAPE)
path_ot_1 = get_path(ireceiver, get_predecessors(cost_matrix_1, ot_plan_1))
path_bell = get_path(ireceiver, predecessors_bell)

arrival_ot = get_travel_cost(path_ot_1, cost_matrix_1)
arrival_bell = get_travel_cost(path_bell, cost_matrix_1)


assert np.isclose(arrival_ot, arrival_bell)

#%%
# Start at index 1 for better visualization
x_coords = x_source[:, 0] + 1
y_coords = x_source[:, 1] + 1

x_edges = np.arange(-0.5, MESH_SHAPE[1] + 0.5) + 1
y_edges = np.arange(-0.5, MESH_SHAPE[0] + 0.5) + 1
X, Y = np.meshgrid(x_edges, y_edges)

fig = plt.figure(figsize=(21.52, 8))

ax1 = plt.subplot(1, 3, 1)
plot_topography(ax1, 
              X, 
              Y, 
              topo, 
              panel_label='a', 
              colorbar=True,
              tick_params_dict=dict(labelbottom=False, 
                                    labeltop=True, 
                                    top=True, 
                                    bottom=False)
              )
plot_nodes(ax1, x_coords, y_coords, source=SOURCE_INDEX, receivers=RECEIVERS_INDEXES)
plot_mass_transport(ax1, x_coords, y_coords, ot_plan_1, colorbar=False)
plot_path(ax1, x_coords, y_coords, path_ot_1, color='r')

ax2 = plt.subplot(1, 3, 2)
plot_cost_matrix(ax2, 
                 cost_matrix_1, 
                 panel_label='b',
                 tick_params_dict=dict(labelbottom=False, 
                                        labeltop=True, 
                                        top=True, 
                                        bottom=False)
                  )

ax3 = plt.subplot(1, 3, 3)
plot_ot_plan(ax3,
             ot_plan_1, 
             panel_label='c',
             tick_params_dict=dict(labelbottom=False, 
                                   labeltop=True, 
                                   top=True, 
                                   bottom=False)
             )

plt.tight_layout()
plt.show()
