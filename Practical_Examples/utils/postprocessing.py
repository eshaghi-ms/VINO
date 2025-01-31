import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.ticker as ticker


def plot_field_1d(x, y, title, folder=None):
    fig_font = "DejaVu Serif"
    fig_size = (6, 3)
    plt.rcParams["font.family"] = fig_font
    plt.figure(figsize=fig_size)
    plt.plot(x, y, 'b', linewidth=1.5)
    # plt.title(title)
    # plt.xlabel(x_label)
    title = title.replace('\\', '').replace('$', '')
    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        full_name = folder + '/' + title
        plt.savefig(full_name + '.png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_field_2d(F, L, W, title, folder=None, file=None, mask=None, isError=False):
    """
    Plots a 2D field stored in a 1D tensor F
    """
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    num_pts_v = F.shape[0]
    num_pts_u = F.shape[1]
    x = jnp.linspace(0, L, num_pts_u)
    y = jnp.linspace(0, W, num_pts_v)
    x_2d, y_2d = jnp.meshgrid(x, y, indexing='xy')
    plt.figure()
    color_type = 'jet'
    if isError:
        color_type = 'bwr'

    if mask is None:
        plt.contourf(x_2d, y_2d, F, levels=512, cmap=color_type)
    else:
        masked_f = np.ma.array(F, mask=mask).copy()
        plt.contourf(x_2d, y_2d, masked_f, 255, cmap=color_type)
    # cbar = plt.colorbar(orientation='horizontal', pad=0.2, aspect=40)
    # cbar = plt.colorbar()
    # cbar.locator = ticker.MaxNLocator(nbins=8)
    # cbar.update_ticks()
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.subplots_adjust(top=0.5, bottom=0.2)
    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        full_name = folder + '/' + file
        plt.savefig(full_name + '.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_fields_2d(Fs_top, Fs_bottom, titles_top, titles_bottom, L, W, masks=None, folder=None, file=None, isError=False):
    """
    Plots two series of 2D fields in a 2x3 grid within one figure.
    Fs_top: List of 2D fields for the top row
    Fs_bottom: List of 2D fields for the bottom row
    titles_top: List of titles for each subplot in the top row
    titles_bottom: List of titles for each subplot in the bottom row
    """
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    num_plots = len(Fs_top)
    num_pts_v, num_pts_u = Fs_top[0].shape
    x = jnp.linspace(0, L, num_pts_u)
    y = jnp.linspace(0, W, num_pts_v)
    x_2d, y_2d = jnp.meshgrid(x, y, indexing='xy')

    fig, axs = plt.subplots(2, num_plots, figsize=(6 * num_plots, 9), gridspec_kw={'width_ratios': [1] * num_plots})

    color_types = ['jet', 'jet', 'bwr']

    # Plot top row
    for i, (F, title, ax) in enumerate(zip(Fs_top, titles_top, axs[0])):
        mask = masks[i] if masks is not None else None
        if mask is None:
            contour = ax.contourf(x_2d, y_2d, F, levels=512, cmap=color_types[i])
        else:
            masked_f = np.ma.array(F, mask=mask).copy()
            contour = ax.contourf(x_2d, y_2d, masked_f, 255, cmap=color_types[i])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)

    # Plot bottom row
    for i, (F, title, ax) in enumerate(zip(Fs_bottom, titles_bottom, axs[1])):
        mask = masks[i] if masks is not None else None
        if mask is None:
            contour = ax.contourf(x_2d, y_2d, F, levels=512, cmap=color_types[i])
        else:
            masked_f = np.ma.array(F, mask=mask).copy()
            contour = ax.contourf(x_2d, y_2d, masked_f, 255, cmap=color_types[i])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)

    # Add a single colorbar on the right side of the entire figure
    cbar = fig.colorbar(contour, ax=axs, orientation='vertical', fraction=0.046, pad=0.04)

    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        full_name = folder + '/' + file
        plt.savefig(full_name + '.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_field_deformed(X_displaced, Y_displaced, L, W, title, folder=None, file=None, mask=None, isError=False):
    """
    Plots a deformed 2D field
    """
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    num_pts_v = X_displaced.shape[0]
    num_pts_u = X_displaced.shape[1]
    x = jnp.linspace(0, L, num_pts_u)
    y = jnp.linspace(0, W, num_pts_v)
    _x, _y = jnp.meshgrid(x, y, indexing='xy')
    plt.figure()
    color_type = 'jet'
    if isError:
        color_type = 'bwr'

    if mask is None:
        plt.pcolor(X_displaced, Y_displaced, _x, cmap=color_type)
    else:
        masked_x_disp = np.ma.array(X_displaced, mask=mask).copy()
        masked_y_disp = np.ma.array(Y_displaced, mask=mask).copy()
        masked_x = np.ma.array(_x, mask=mask).copy()
        plt.contourf(masked_x_disp, masked_y_disp, masked_x, 512, cmap=color_type)
    # cbar = plt.colorbar(orientation='horizontal', pad=0.2, aspect=40)
    # cbar.locator = ticker.MaxNLocator(nbins=8)
    # cbar.update_ticks()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.subplots_adjust(top=0.5, bottom=0.2)
    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        full_name = folder + '/' + file
        plt.savefig(full_name + '.png', dpi=600, bbox_inches='tight')
    plt.show()
