# Helper functions for plotting
#
# Author: M.Y. Kingma


from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt


class IndexTracker(object):
    """
    This class is used to track the current slice index of a 3D volume.
    It is used in the plot_scrollable_volume function.
    """

    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title("use scroll wheel to navigate images")

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(np.abs(self.X[:, :, self.ind]))
        self.update()

    def onscroll(self, event):
        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(np.abs(self.X[:, :, self.ind]))
        self.ax.set_ylabel("slice %s" % self.ind)
        self.im.axes.figure.canvas.draw()


def plot_scrollable_volume(volume_data):
    """
    This function plots a 3D volume and allows the user to scroll through the slices of the volume.
    If the user scrolls up, the function increments the current slice index by 1.
    If the user scrolls down, the function decrements the current slice index by 1.
    The function then updates the displayed image to show the new slice.

    Inputs:
    - volume_data: A 3D numpy array representing the volume data to be displayed

    Returns:
    - None
    """

    # Check if values are complex
    if np.iscomplexobj(volume_data):

        # Take absolute value of volume data
        volume_data = np.abs(volume_data)

    # Check dimensions of volume data
    if len(volume_data.shape) != 3:

        # Remove dimension with length 1
        volume_data = np.squeeze(volume_data)

    fig, ax = plt.subplots(1, 1)
    plt.gray()

    tracker = IndexTracker(ax, volume_data)
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)

    plt.show()


def check_plot_inputs(volume, axis, slice_no, complex_volume_required=False):
    """
    This function checks the inputs to the plot_slice and plot_abs_angle_real_imag_from_complex_volume functions.

    Inputs:
    - volume: A 3D numpy array representing the volume data
    - axis: An integer representing the axis along which the slice is to be taken
    - slice_no: An integer representing the slice number to be plotted
    - complex_volume_required: A boolean representing whether the volume must be complex

    Returns:
    - None
    """

    # Check if values are complex
    if complex_volume_required and not np.iscomplexobj(volume):
        raise ValueError("Volume must be complex")
    elif not complex_volume_required and np.iscomplexobj(volume):
        raise ValueError("Volume must not be complex")

    # Check if volume is 3D
    if len(volume.shape) != 3:
        raise ValueError("Volume must be 3D")

    # Check if axis are valid
    if axis not in [0, 1, 2]:
        raise ValueError("Axis must be 0, 1 or 2")

    # Check if slice_no is valid
    if slice_no < 0 or slice_no >= volume.shape[axis]:
        raise ValueError(
            "Slice number must be between 0 and the length of the volume along the axis")


def plot_slice(volume, slice_no, axis=0, save=False, filepath=None):
    """
    This function plots a slice of a 3D volume.

    Inputs:
    - volume: A 3D numpy array representing the volume data
    - slice_no: An integer representing the slice number to be plotted
    - axis: An integer representing the axis along which the slice is to be taken

    Returns:
    - None
    """

    # Check inputs
    check_plot_inputs(volume, axis, slice_no)

    # Get slice
    if axis == 0:
        slice = volume[slice_no, :, :]
    elif axis == 1:
        slice = volume[:, slice_no, :]
    elif axis == 2:
        slice = volume[:, :, slice_no]
    else:
        raise ValueError("Axis must be 0, 1 or 2")

    if save:
        plt.imsave(filepath, slice, cmap="gray")

    # Plot slice
    plt.imshow(slice, cmap="gray")
    plt.show()


def plot_abs_angle_real_imag_from_complex_volume(volume, axis=0, slice_no=0):
    """
    This function plots the absolute value, angle, real, and imaginary parts of a 3D complex volume.

    Inputs:
    - volume: A 3D numpy array representing the volume data
    - axis: An integer representing the axis along which the slice is to be taken
    - slice_no: An integer representing the slice number to be plotted

    Returns:
    - None
    """

    # Check inputs
    check_plot_inputs(volume, axis, slice_no, complex_volume_required=True)

    # Get the absolute value, angle, real, and imaginary parts of the volume
    abs_volume = np.abs(volume)
    angle_volume = np.angle(volume)
    real_volume = np.real(volume)
    imag_volume = np.imag(volume)

    # Get the slice
    if axis == 0:
        abs_volume = abs_volume[slice_no, :, :]
        angle_volume = angle_volume[slice_no, :, :]
        real_volume = real_volume[slice_no, :, :]
        imag_volume = imag_volume[slice_no, :, :]
    elif axis == 1:
        abs_volume = abs_volume[:, slice_no, :]
        angle_volume = angle_volume[:, slice_no, :]
        real_volume = real_volume[:, slice_no, :]
        imag_volume = imag_volume[:, slice_no, :]
    elif axis == 2:
        abs_volume = abs_volume[:, :, slice_no]
        angle_volume = angle_volume[:, :, slice_no]
        real_volume = real_volume[:, :, slice_no]
        imag_volume = imag_volume[:, :, slice_no]

    # Plot the absolute value, angle, real, and imaginary parts of the volume
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(abs_volume, cmap="gray")
    axs[0, 0].set_title("Absolute Value")
    axs[0, 1].imshow(angle_volume, cmap="gray")
    axs[0, 1].set_title("Angle")
    axs[1, 0].imshow(real_volume, cmap="gray")
    axs[1, 0].set_title("Real")
    axs[1, 1].imshow(imag_volume, cmap="gray")
    axs[1, 1].set_title("Imaginary")
    plt.show()


def plot_orientations(volume_data, spacing, filename):
    """
    This function plots the axial, coronal and sagittal slices of a 3D volume.
    It uses the spacing of the volume to set the aspect ratio of the plotted image.
    It plots the center slice per orientation.

    Inputs:
    - volume_data: A 3D numpy array representing the volume data
    - spacing: A tuple representing the voxel spacing
    - filename: A string representing the filename of the nifti file

    Returns:
    - None
    """

    # Check if values are complex
    if np.iscomplexobj(volume_data):

        # Take absolute value of volume data
        volume_data = np.abs(volume_data)

    # Check dimensions of volume data
    if len(volume_data.shape) != 3:
        raise ValueError("Volume must be 3D")

    # Get slice dimension spacing, is highest spacing value
    slice_dimension_spacing = np.max(spacing)

    # Get voxexl spacing, is lowest spacing value
    voxel_spacing = np.min(spacing)

    # Get slice numbers for orientation, center slice
    slice_no_axial = volume_data.shape[2] // 2
    slice_no_coronal = volume_data.shape[1] // 2
    slice_no_sagittal = volume_data.shape[0] // 2

    # Plot Axial, Coronal and Sagittal slices, remove axis ticks and labels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(volume_data[:, :, slice_no_axial], cmap="gray")
    axes[0].set_title("Axial")
    axes[0].axis("off")
    axes[1].imshow(volume_data[:, slice_no_coronal, :], cmap="gray")
    axes[1].set_title("Coronal")
    axes[1].axis("off")
    axes[2].imshow(volume_data[slice_no_sagittal, :, :], cmap="gray")
    axes[2].set_title("Sagittal")
    axes[2].axis("off")

    # Set the spacing of the plotted image
    if "sag" in filename:
        axes[0].set_aspect(slice_dimension_spacing / voxel_spacing)
        axes[1].set_aspect(slice_dimension_spacing / voxel_spacing)
        axes[2].set_aspect(voxel_spacing / voxel_spacing)
    else:
        axes[0].set_aspect(voxel_spacing / voxel_spacing)
        axes[1].set_aspect(voxel_spacing / slice_dimension_spacing)
        axes[2].set_aspect(voxel_spacing / slice_dimension_spacing)

    plt.show()


def get_p_value_significance_symbols(p_value):
    """
    This function returns the significance symbols for a p-value.

    Inputs:
    - p_value: A float representing the p-value

    Returns:
    - A string representing the significance symbols
    """

    # Check if p-value is significant
    if p_value < 0.001:
        significance_symbols = "***"
    elif p_value < 0.01:
        significance_symbols = "**"
    elif p_value < 0.05:
        significance_symbols = "*"
    else:
        significance_symbols = ""

    return significance_symbols


def create_visual_inspection_plots(volumes, axis, slice_no, zoom_x, zoom_y, rotate_degrees=0, blur_scores_dict=None, ssim_scores_dict=None, name_suffix="", x_offset_zoom=0, y_offset_zoom=0.02, offset_title=0.85):
    fig, axs = plt.subplots(1, len(volumes), figsize=(len(volumes)*5, 5))
    if rotate_degrees != 0:
        axs = axs.flatten(order='F')
    for i, vol in enumerate(volumes):

        # Get slice
        if axis == 0:
            vol_data = vol["volume_data"][slice_no, :, :]
        elif axis == 1:
            vol_data = vol["volume_data"][:, slice_no, :]
        elif axis == 2:
            vol_data = vol["volume_data"][:, :, slice_no]

        # Rotate data
        if rotate_degrees != 0:
            vol_data = np.rot90(vol_data, k=int(rotate_degrees/90))

        # Plot slice
        axs[i].imshow(vol_data, cmap='gray', vmin=0, vmax=1)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        if zoom_x is not None and zoom_y is not None:
            zoom_data = vol_data[zoom_y[0]:zoom_y[1], zoom_x[0]:zoom_x[1]]

            zoom_ax = axs[i].inset_axes(
                [x_offset_zoom, y_offset_zoom, 0.5, 0.5])
            zoom_ax.imshow(zoom_data, cmap='gray', vmin=0, vmax=0.7)
            zoom_ax.set_xticks([])
            zoom_ax.set_yticks([])

            # Set border color to yellow
            zoom_ax.spines['bottom'].set_color('yellow')
            zoom_ax.spines['top'].set_color('yellow')
            zoom_ax.spines['right'].set_color('yellow')
            zoom_ax.spines['left'].set_color('yellow')

            # Indicate zoom area
            rect = patches.Rectangle((zoom_x[0], zoom_y[0]), zoom_x[1]-zoom_x[0], zoom_y[1]-zoom_y[0],
                                     linewidth=1, edgecolor='yellow', facecolor='none')
            axs[i].add_patch(rect)

            # Add connect line
            con = patches.ConnectionPatch(xyA=(0, 1), xyB=(zoom_x[0], zoom_y[0]),
                                          coordsA="axes fraction", coordsB="data", axesA=zoom_ax, axesB=axs[i],
                                          color="yellow", linewidth=1)
            axs[i].add_artist(con)
            con = patches.ConnectionPatch(xyA=(1, 0), xyB=(zoom_x[1], zoom_y[1]),
                                          coordsA="axes fraction", coordsB="data", axesA=zoom_ax, axesB=axs[i],
                                          color="yellow", linewidth=1)
            axs[i].add_artist(con)

        axs[i].text(0.01, offset_title, vol['name'], transform=axs[i].transAxes,
                    fontsize=16, fontweight='bold', va='bottom', color='yellow')

        # Check if blur scores are provided
        if vol["name"] in blur_scores_dict.keys():
            axs[i].text(0.01, offset_title - 0.075, f"Blur strength: {blur_scores_dict[vol['name']]:.4g}", transform=axs[i].transAxes,
                        fontsize=16, fontweight='bold', va='bottom', color='yellow')

        # Check if SSIM scores are provided
        if vol["name"] in ssim_scores_dict.keys():
            axs[i].text(0.01, offset_title - 0.125, f"SSIM: {ssim_scores_dict[vol['name']]:.4g}", transform=axs[i].transAxes,
                        fontsize=16, fontweight='bold', va='bottom', color='yellow')

    plt.tight_layout()

    # Save plot
    plt.savefig(f"/Users/Maurice/Downloads/plots/visual_inspection_{name_suffix}.png",
                bbox_inches='tight', dpi=300)
