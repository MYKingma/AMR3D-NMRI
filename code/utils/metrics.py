# Metrics for evaluating reconstruction quality.
#
# Code has been taken from https://github.com/wdika/mridc
#
# Optimized for project by M.Y. Kingma

import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Optional
from skimage.measure import blur_effect
import matplotlib.pyplot as plt
from utils.plot import plot_slice


def mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)  # type: ignore


def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return float(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[np.ndarray] = None) -> float:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = np.max(gt)
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: Optional[np.ndarray] = None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if gt.ndim != 3 and gt.ndim != 2:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if gt.ndim != pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = np.max(gt) if maxval is None else maxval

    if gt.ndim == 2:
        return structural_similarity(gt, pred, data_range=maxval)

    _ssim = sum(
        structural_similarity(gt[slice_num], pred[slice_num], data_range=maxval) for slice_num in range(gt.shape[0])
    )

    return _ssim / gt.shape[0]


METRIC_FUNCS = {"MSE": mse, "NMSE": nmse, "PSNR": psnr, "SSIM": ssim}


class Metrics:
    """Maintains running statistics for a given collection of metrics."""

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        """
        Pushes a new batch of metrics to the running statistics.
        Args:
            target: target image
            recons: reconstructed image
        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        for metric, func in METRIC_FUNCS.items():
            self.metrics_scores[metric].push(func(target, recons))

    def means(self):
        """
        Mean of the means of each metric.
        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        return {metric: stat.mean() for metric, stat in self.metrics_scores.items()}

    def stddevs(self):
        """
        Standard deviation of the means of each metric.
        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        return {metric: stat.stddev() for metric, stat in self.metrics_scores.items()}

    def __repr__(self):
        """
        Representation of the metrics.
        Returns:
            str: A string representation of the metrics.
        """
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))

        res = " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}" for name in metric_names) + "\n"

        return res

    def __csv__(self):
        """
        CSV representation of the metrics.
        Returns:
            str: A CSV string of the metrics.
        """
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))

        res = ",".join(
            f"{means[name]:.4g},{2 * stddevs[name]:.4g}" for name in metric_names) + "\n"

        return res


def calculate_blur_effect_per_slice(volume, axis, h_size):
    """
    Calculate blur effect per slice of volume.

    Inputs:
    - volume: A 3D numpy array representing the volume
    - axis: An integer representing the axis to iterate over
    - h_size: An integer representing the size of the kernel

    Returns:
    - blur_scores: A list of blur scores per slice
    """

    blur_scores = []

    # Loop over slices
    for i in range(volume.shape[axis]):
        # Get slice
        if axis == 0:
            slice = volume[i, :, :]
        elif axis == 1:
            slice = volume[:, i, :]
        else:
            slice = volume[:, :, i]

        with np.errstate(divide='ignore', invalid='ignore'):
            blur_score = blur_effect(slice, h_size=h_size)

        blur_scores.append(blur_score)

    return blur_scores


def calculate_ssim_per_slice(target_volume, ground_truth_volume, axis):
    """
    Calculate SSIM per slice of volume.

    Inputs:
    - target_volume: A 3D numpy array representing the volume
    - ground_truth_volume: A 3D numpy array representing the ground truth volume
    - axis: An integer representing the axis to iterate over

    Returns:
    - ssim_scores: A list of SSIM scores per slice
    """

    ssim_scores = []

    # Loop over slices
    for i in range(target_volume.shape[axis]):
        # Get slice
        if axis == 0:
            target_slice = target_volume[i, :, :]
            ground_truth_slice = ground_truth_volume[i, :, :]
        elif axis == 1:
            target_slice = target_volume[:, i, :]
            ground_truth_slice = ground_truth_volume[:, i, :]
        else:
            target_slice = target_volume[:, :, i]
            ground_truth_slice = ground_truth_volume[:, :, i]

        with np.errstate(divide='ignore', invalid='ignore'):
            ssim_score = ssim(ground_truth_slice, target_slice)

        ssim_scores.append(ssim_score)

    return ssim_scores
