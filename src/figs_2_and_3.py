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
from raytracing_ot import get_cost_matrix
import numpy as np
import seislib.colormaps as scm
from docplex.mp.model import Model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath,newtxtext,newtxmath}'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 20


MESH_SHAPE = (6, 6)
SOURCE_INDEX = (5, 0)
RECEIVERS_INDEXES = [[(5, 5)], [(5, 5), (1, 2), (3, 5)]]
REF_VEL = 3


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
        

def two_gaussians_pattern(x_source, base_velocity=3):

    x_coords = x_source[:, 0]
    y_coords = x_source[:, 1]

    x_gauss_1 = (x_coords.max() + x_coords.min()) / 1.8
    y_gauss_1 = (y_coords.max() + y_coords.min()) / 1.2
    
    x_gauss_2 = (x_coords.max() + x_coords.min()) / 10
    y_gauss_2 = (y_coords.max() + y_coords.min()) / 2.5
    
    anomaly1 = -0.4
    anomaly2 = 1.5
    
    grid_width = x_coords.max() - x_coords.min()
    grid_height = y_coords.max() - y_coords.min()
    sigma_1 = np.sqrt(grid_width**2 + grid_height**2) / 8

    # Compute the Gaussian anomaly values
    distances1 = (x_coords - x_gauss_1)**2 + (y_coords - y_gauss_1)**2
    gaussian1 = np.exp(-distances1 / (2 * sigma_1**2)) * anomaly1
    
    sigma_2_x = 2
    sigma_2_y = 1.2
    gauss_2_x = (x_coords - x_gauss_2)**2 / (2 * sigma_2_x**2)
    gauss_2_y = (y_coords - y_gauss_2)**2 / (2 * sigma_2_y**2)
    
    gaussian2 = np.exp(-(gauss_2_x + gauss_2_y)) * anomaly2
    anomaly = gaussian1 + gaussian2
    
    base_velocity = 4.5 + x_source[:, 1] * -0.3
    return base_velocity + base_velocity * anomaly


def plot_mass_transport(ax, x_coords, y_coords, ot_plan):
    
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


def get_masses(isource, ireceivers, size):
    mass_excess = ireceivers.size - 1
    mass_source = np.ones(size) + mass_excess
    mass_target = np.ones(size) + mass_excess
    for ireceiver in ireceivers:
        mass_source[isource] += 1
        mass_target[ireceiver] += 1
    return mass_source, mass_target


def solve_ot(mesh_shape, source_index, receivers_indexes, slowness=None):
    
    cost_matrix = get_cost_matrix(mesh_shape, 
                                  slowness=slowness,
                                  fill_unallowed=np.inf)
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
    constraints = []
    for i, js in i_to_js.items():
        constraints.append(mdl.sum(x[i, j] for j in js) == a[i])            
    for j, is_ in j_to_is.items():
        constraints.append(mdl.sum(x[i, j] for i in is_) == b[j])    
    mdl.add_constraints(constraints)
    
    mdl.minimize(mdl.sum(cost_matrix[i, j] * x[i, j] for (i, j) in x.keys()))
    mdl.solve()

    ot_plan = np.zeros(cost_matrix.shape) # Get ot_plan
    for (i, j), var in x.items():
        ot_plan[i, j] = var.solution_value

    return cost_matrix, ot_plan
    
    
def plot_slowness(ax, 
                  X, 
                  Y, 
                  slowness_2d, 
                  panel_label, 
                  colorbar=True,
                  enumerate_pixels=True,
                  tick_params_dict={}):
    img = ax.pcolormesh(X, Y, slowness_2d, cmap=scm.roma_r)
    ax.set_aspect('equal')
    labels = (x_edges + 0.5).astype(int)[:-1]
    ax.set_xticks(labels)
    ax.set_xticklabels([r'$x_%d$'%i for i in labels])
    ax.set_yticks(labels[::-1])
    ax.set_yticklabels([r'$y_%d$'%i for i in labels])

    ax.tick_params(**tick_params_dict)
    
    if colorbar:
        cbar = make_colorbar(ax, img)
        cbar.set_label('Slowness [s/m]', fontsize=22, labelpad=10)

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
    cmap = plt.cm.get_cmap("viridis").copy()
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
        cbar.set_label(r'Cost Matrix ${\bf C}$ [s]', fontsize=22, labelpad=10)
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
        
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(unique_values.astype(str))
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(unique_values.astype(str))
        
    ax.text(x=-0.08, 
            y=1.02, 
            s=r'\textbf{(%s)}'%panel_label, 
            transform=ax.transAxes, fontsize=25)


def plot_nodes(ax, x_coords, y_coords, source=None, receivers=None):
    color = 'fuchsia'
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

#%%

x_source = np.zeros((MESH_SHAPE[0] * MESH_SHAPE[1], 2))
for inode, (i, j) in enumerate(np.ndindex(MESH_SHAPE)):
    x_source[inode] = [j, i]

velocity = two_gaussians_pattern(x_source, base_velocity=REF_VEL)
slowness = 1 / velocity

cost_matrix = get_cost_matrix(MESH_SHAPE, 
                              slowness=slowness, 
                              fill_unallowed=np.inf)

#%%
# Start at index 1 for better visualization
x_coords = x_source[:, 0] + 1
y_coords = x_source[:, 1] + 1

x_edges = np.arange(-0.5, MESH_SHAPE[1] + 0.5) + 1
y_edges = np.arange(-0.5, MESH_SHAPE[0] + 0.5) + 1
X, Y = np.meshgrid(x_edges, y_edges)
slowness_2d = slowness.reshape(MESH_SHAPE)

fig = plt.figure(figsize=(15, 8))

ax1 = plt.subplot(1, 2, 1)
plot_slowness(ax1, 
              X, 
              Y, 
              slowness_2d, 
              panel_label='a', 
              colorbar=True,
              tick_params_dict=dict(labelbottom=False, 
                                    labeltop=True, 
                                    top=True, 
                                    bottom=False)
              )
plot_nodes(ax1, x_coords, y_coords)


ax2 = plt.subplot(1, 2, 2)
plot_cost_matrix(ax2, 
                  cost_matrix, 
                  panel_label='b',
                  tick_params_dict=dict(labelbottom=False, 
                                        labeltop=True, 
                                        top=True, 
                                        bottom=False)
                  )

plt.tight_layout()
plt.show()

#%%

ot_plan1 = solve_ot(mesh_shape=MESH_SHAPE, 
                    source_index=SOURCE_INDEX, 
                    receivers_indexes=RECEIVERS_INDEXES[0],
                    slowness=slowness)[1]
ot_plan2 = solve_ot(mesh_shape=MESH_SHAPE, 
                    source_index=SOURCE_INDEX, 
                    receivers_indexes=RECEIVERS_INDEXES[1],
                    slowness=slowness)[1]

fig = plt.figure(figsize=(14.5, 16))

ax1 = plt.subplot(2, 2, 1)
plot_ot_plan(ax1,
             ot_plan1, 
             panel_label='a',
             tick_params_dict=dict(labelbottom=False, 
                                    labeltop=True, 
                                    top=True, 
                                    bottom=False)
              )

ax2 = plt.subplot(2, 2, 2)
plot_slowness(ax2, 
              X, 
              Y, 
              slowness_2d, 
              panel_label='b', 
              colorbar=False,
              tick_params_dict=dict(labelbottom=False, 
                                    labeltop=True, 
                                    top=True, 
                                    bottom=False),
              enumerate_pixels=True
              )
plot_nodes(ax2, 
           x_coords, 
           y_coords, 
           source=SOURCE_INDEX, 
           receivers=RECEIVERS_INDEXES[0])
plot_mass_transport(ax2, x_coords, y_coords, ot_plan1)


    
ax3 = plt.subplot(2, 2, 3)
plot_ot_plan(ax3,
             ot_plan2, 
             panel_label='c',
             tick_params_dict=dict(labelbottom=False, 
                                   labeltop=True, 
                                   top=True, 
                                   bottom=False)
             )

ax4 = plt.subplot(2, 2, 4)
plot_slowness(ax4, 
              X, 
              Y, 
              slowness_2d, 
              panel_label='d', 
              colorbar=False,
              tick_params_dict=dict(labelbottom=False, 
                                    labeltop=True, 
                                    top=True, 
                                    bottom=False),
              enumerate_pixels=True
              )
plot_nodes(ax4, 
           x_coords, 
           y_coords, 
           source=SOURCE_INDEX, 
           receivers=RECEIVERS_INDEXES[1])

plot_mass_transport(ax4, x_coords, y_coords, ot_plan2)

plt.tight_layout()
plt.show()





