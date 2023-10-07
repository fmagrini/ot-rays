#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:40:14 2023

@author: fabrizio
"""

import os
import sys
sys.path.append(os.getcwd())
import itertools as it
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_array, lil_array
from scipy.sparse.csgraph import bellman_ford
from raytracing_ot import get_finite_idx_and_cost
import seislib.colormaps as scm
from docplex.mp.model import Model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker 
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath,newtxtext,newtxmath}'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 20


TOPOGRAPHY = '../data/topo_large.npy'
LONMIN, LONMAX = -7, 18
LATMIN, LATMAX = 6.5, 31.5
SOURCE_INDEX = (125, 125)
RECEIVERS_INDEXES = None



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
    cost = np.zeros(idx.shape) # Energy required to travel
    
    for icost, (i, j) in enumerate(idx_2d):
        if i == j:
            continue
        ix, iy = np.unravel_index(i, mesh_shape)
        jx, jy = np.unravel_index(j, mesh_shape)
        dist_x = abs(x[i] - x[j])
        dist_y = abs(y[i] - y[j])
        dist = np.sqrt(dist_x**2 + dist_y**2)
        # dist = gc_distance(lat[ix, iy], lon[ix, iy], lat[jx, jy], lon[jx, jy]) / 1000
        g = (topo[jx, jy] - topo[ix, iy]) / dist # topographic gradient
        if g >= 0:
            cost[icost] = E0 * dist + g * k_up
        else:
            cost[icost] = E0 * dist + g * k_down

    return idx, cost


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
    

def make_colorbar(ax, fig=None, img=None, cmap=None, norm=None, orientation='vertical'):
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("bottom", size="2%", pad=0.1)
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
        

def get_masses(isource, ireceivers, size):
    mass_excess = ireceivers.size - 1
    mass_source = np.ones(size) + mass_excess
    mass_target = np.ones(size) + mass_excess
    for ireceiver in ireceivers:
        mass_source[isource] += 1
        mass_target[ireceiver] += 1
    return mass_source, mass_target


def solve_ot(mesh_shape, 
             source_index, 
             receiver_indexes, 
             cost_matrix,
             idx_2d):
    
    print('Getting Cost Matrix and Mass Vectors')    
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
    print('Solving the LP')
    mdl.solve()

    ot_plan = lil_array(cost_matrix.shape) 
    for (i, j), var in x.items():
        if var.solution_value:
            ot_plan[i, j] = var.solution_value    
    
    return csr_array(ot_plan)
    

def solve_bellman_ford(mesh_shape, 
                       source_index, 
                       receiver_indexes,
                       graph):

    isource = np.int32(ravel_index(source_index, mesh_shape))
    
    arrivals, predecessors = bellman_ford(graph, 
                                          indices=isource,
                                          return_predecessors=True)
    return arrivals, predecessors


def label_format(x, pos): # your custom formatter function: divide by 100.0
    s = '{}'.format(x)
    return s


def plot_topography(ax, x, y, topo, fig=None, **kwargs):
    fmt = mticker.FuncFormatter(label_format)  
    img = ax.imshow(topo, 
                    extent=[x.min(), x.max(), y.min(), y.max()], 
                    **kwargs)
    cbar = make_colorbar(ax, fig=fig, img=img, orientation='horizontal')
    cbar.set_label('Elevation [m]', fontsize=22, labelpad=10)
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel('x [km]', labelpad=10)
    ax.set_ylabel('y [km]')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='both'))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune='both'))
    ax.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
    ax.xaxis.set_label_position('top')
    return img, cbar
    

def plot_source_receivers(ax, x, y, source, receivers=None):
    color = 'r'
    ax.scatter(source[1], 
               source[0], 
               color=color, 
               marker='*', 
               s=600,
               ec='k')
    if receivers is not None:
        for receiver in receivers:
            ax.scatter(receiver[1], 
                       receiver[0], 
                       color=color, 
                       marker='^', 
                       s=180,
                       ec='k')


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

#%%

topo = np.load(TOPOGRAPHY)
mesh_shape = topo.shape
x_source = np.zeros((mesh_shape[0] * mesh_shape[1], 2))
for inode, (i, j) in enumerate(np.ndindex(mesh_shape)):
    x_source[inode] = [j, i]

source_index = SOURCE_INDEX
receiver_indexes = list(it.product(range(topo.shape[0]),
                                   range(topo.shape[1])))

idx, cost = get_cost_from_topo(*x_source.T, topo, E0=5, k_up=1, k_down=0.1)
cost_matrix_shape = (mesh_shape[0] * mesh_shape[1],) * 2
idx_2d = np.int32(np.unravel_index(idx, cost_matrix_shape))
cost_matrix = csr_array((cost, idx_2d))

i = [i for i in range(cost.size) if idx_2d[0][i] != idx_2d[1][i]]
graph = csr_array((cost[i], (idx_2d[0][i], idx_2d[1][i])))
#%%
ot_plan = solve_ot(mesh_shape, 
                   source_index, 
                   receiver_indexes,
                   cost_matrix,
                   idx_2d)
predecessors_ot = get_predecessors(cost_matrix, ot_plan)
travel_times_ot = get_travel_times(cost_matrix, predecessors_ot)
travel_times_bell, predecessors_bell = solve_bellman_ford(mesh_shape, 
                                                          source_index, 
                                                          receiver_indexes,
                                                          graph)

assert np.allclose(travel_times_ot, travel_times_bell)

#%%
x, y = x_source.T.copy()
x -= x.max() / 2
y -= y.max() / 2
source = 0, 0


fig = plt.figure(figsize=(14.5, 7.6))

ax1 = fig.add_subplot(1, 2, 1)
img1, cb1 = plot_topography(ax1, x, y, topo/1000, fig=fig, cmap=scm.grayC)
ax1.text(x=-0.08, 
         y=1.06,
         s=r'\textbf{(a)}', 
         transform=ax1.transAxes, 
         fontsize=25)
cb1.set_label('Elevation [km]', labelpad=10)
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(1, 2, 2)
img2, cb2 = plot_topography(ax2, 
                            x,
                            y,
                            travel_times_ot.reshape(topo.shape), 
                            fig=fig, 
                            cmap='turbo')

ax2.tick_params(left=False, labelleft=False, right=True, labelright=True)
ax2.yaxis.set_label_position('right')

cb2.set_label('Travel Cost', labelpad=10)
plot_source_receivers(ax2, 
                      x, 
                      y, 
                      source)
ax2.text(x=-0.08, 
         y=1.06, 
         s=r'\textbf{(b)}', 
         transform=ax2.transAxes, 
         fontsize=25)
ax2.grid(alpha=0.3)
plt.tight_layout(w_pad=0.75)
plt.show()
