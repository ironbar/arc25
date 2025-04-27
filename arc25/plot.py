import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from arc25.training_tasks import Task

def plot_grids_with_shape(grids, suptitle=None, facecolor='white'):
    plt.figure(facecolor=facecolor)
    for plot_idx, grid in enumerate(grids):
        plt.subplot(1, len(grids), plot_idx + 1)
        plot_grid(grid)
        plt.title(f'{len(grid)}x{len(grid[0])}')
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_task(task):
    if isinstance(task, dict):
        samples = task['train'] + task['test']
        for plot_idx, sample in enumerate(samples):
            plt.subplot(2, len(samples), plot_idx + 1)
            plot_grid(sample['input'])
            if plot_idx < len(task['train']):
                plt.xlabel(f'Train {plot_idx}')
            else:
                plt.xlabel(f'Test {plot_idx - len(task["train"])}')
            if 'output' in sample:
                plt.subplot(2, len(samples), plot_idx + 1 + len(samples))
                plot_grid(sample['output'])
            plt.tight_layout()
    if isinstance(task, Task):
        plot_grids_with_shape(task.inputs, suptitle='Inputs')
        plot_grids_with_shape(task.outputs, suptitle='Outputs')
    
    


def plot_grid(grid, write_numbers=False):
    grid = np.array(grid)
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.grid(True,which='both',color='lightgrey', linewidth=0.5)
    plt.xticks(np.arange(-0.5, grid.shape[1]), [])
    plt.yticks(np.arange(-0.5, grid.shape[0]), [])
    plt.xlim(-0.5, grid.shape[1]-0.5)
    if write_numbers:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                plt.text(j, i, str(grid[i, j]), ha='center', va='center')