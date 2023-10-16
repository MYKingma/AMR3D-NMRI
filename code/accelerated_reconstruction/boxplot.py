# Script to generate boxplots for the SSIM and PSNR scores of the
# accelerated reconstruction models
#
# Author: M.Y. Kingma

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define the paths to the csv files
csv_l1_1d_20_path = "path/to/scores/l1_1d_20epoch.csv"
csv_ssim_1d_20_path = "path/to/scores/ssim_1d_20epoch.csv"
csv_ssim_2d_20_path = "path/to/scores/ssim_2d_20epoch.csv"
csv_ssim_1d_200_path = "path/to/scores/ssim_1d_200epoch.csv"

# Load all csv files
csv_l1_1d_20 = pd.read_csv(csv_l1_1d_20_path)
csv_ssim_1d_20 = pd.read_csv(csv_ssim_1d_20_path)
csv_ssim_2d_20 = pd.read_csv(csv_ssim_2d_20_path)
csv_ssim_1d_200 = pd.read_csv(csv_ssim_1d_200_path)

# Remode the outlier in l1 1d
csv_l1_1d_20 = csv_l1_1d_20[csv_l1_1d_20["SSIM"] > 0.75]

# Create boxplot for comparing SSIM for all 20 epoch simulations
fig, ax = plt.subplots(dpi=300)
boxplot = ax.boxplot([csv_ssim_1d_20["SSIM"], csv_l1_1d_20["SSIM"],
                      csv_ssim_2d_20["SSIM"]], patch_artist=True)

# Colour of the median lines
plt.setp(boxplot["medians"], color="k")

# Change the background colour of the boxes to Seaborn's "pastel" palette
colors = sns.color_palette("pastel")

# Change the colour of the boxes
for patch, color in zip(boxplot["boxes"], colors):
    patch.set_facecolor(color)

# Set the axis labels
ax.set_xticklabels(["1D Gaussian SSIM", "1D Gaussian L1", "2D Gaussian SSIM"])
ax.set_ylabel("SSIM")

# Set the title
ax.set_title("SSIM for 20 epoch simulations")

# Set y axis limits
ax.set_ylim(0.9, 1)

# Add more ticks
ax.set_yticks([0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
               0.96, 0.97, 0.98, 0.99, 1])

# Save the plot
plt.savefig("/Users/Maurice/Downloads/plots/SSIM for 20 epoch simulations.png")


# Create boxplot for comparing PSNR for all 20 epoch simulations
fig, ax = plt.subplots(dpi=300)
boxplot = ax.boxplot([csv_ssim_1d_20["PSNR"], csv_l1_1d_20["PSNR"],
                      csv_ssim_2d_20["PSNR"]], patch_artist=True)

# Colour of the median lines
plt.setp(boxplot["medians"], color="k")

# Change the background colour of the boxes to Seaborn's "pastel" palette
colors = sns.color_palette("pastel")

# Change the colour of the boxes
for patch, color in zip(boxplot["boxes"], colors):
    patch.set_facecolor(color)

# Set the axis labels
ax.set_xticklabels(["1D Gaussian SSIM", "1D Gaussian L1", "2D Gaussian SSIM"])
ax.set_ylabel("PSNR")

# Set the title
ax.set_title("PSNR for 20 epoch simulations")

# Set y axis limits
ax.set_ylim(25, 47.5)

# Add more ticks
ax.set_yticks([25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5])

# Save the plot
plt.savefig("/output/path/for/PSNR for 20 epoch simulations.png")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2), dpi=300)

# Create boxplots for each subplot
boxplot1 = ax1.boxplot(csv_ssim_1d_200["SSIM"], patch_artist=True, widths=0.5)
boxplot2 = ax2.boxplot(csv_ssim_1d_200["PSNR"], patch_artist=True, widths=0.5)

# Colour of the median lines
plt.setp(boxplot1["medians"], color="k")

# Change the background colour of the boxes to Seaborn's "pastel" palette
colors = sns.color_palette("pastel")

# Change the colour of the boxes
for patch, color in zip(boxplot1["boxes"], colors):
    patch.set_facecolor(color)

# Colour of the median lines
plt.setp(boxplot2["medians"], color="k")

# Change the background colour of the boxes to Seaborn"s "pastel" palette
colors = sns.color_palette("pastel")

# Change the colour of the boxes
for patch in boxplot2["boxes"]:
    patch.set_facecolor(colors[1])

# Add titles and axis labels
ax1.set_ylabel('SSIM')

# Set y axis limits
ax1.set_ylim(0.96, 1.0)

# Add more ticks
ax1.set_yticks([0.96, 0.97, 0.98, 0.99, 1.0])

# Add titles and axis labels
ax2.set_ylabel('PSNR')

# Set y axis limits
ax2.set_ylim(31, 43)

# Add more ticks
ax2.set_yticks([31, 33, 35, 37, 39, 41, 43])

# Remove the x-axis ticks and labels
ax1.set_xticks([])
ax2.set_xticks([])

# Tichten the layout and save
fig.tight_layout()

# Round axis labels to 2 decimal places
ax1.set_yticklabels(["{:.2f}".format(x) for x in ax1.get_yticks()])

# Save the plot
plt.savefig(
    "/output/path/for/SSIM and PSNR for 200 epoch simulation.png")
