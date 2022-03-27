from operator import index
import re, seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import sys

""" 
    This is a simple Python code which prints a
    scatterplot of the results computed by the C 
    program. "N_COLORS" can be adjusted in order
    to match with the number of clusters.
    Disclaimer: this code works for 3D dataset only.

"""

# Number of color to adjust in function of the number of clusters
N_COLORS = int(sys.argv[1])
DATASET = sys.argv[2]

# Setting figure
fig = plt.figure(figsize=(10,6))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
cmap=ListedColormap(sns.color_palette("husl", n_colors=N_COLORS))

# Retrieving results and data manupulation
df_points = pd.read_csv(DATASET, header=None, delim_whitespace=True);
df_clusters = pd.read_csv("membership.csv", header=None)
df_result = pd.concat([df_points, df_clusters], axis=1)
df_result.columns = ["id", "x", "y", "z", "c"]

# Setting scatter
sc = ax.scatter(df_result.x, df_result.y, df_result.z, s=40, c=df_result.c, cmap=cmap)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Setting legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

# Show
plt.show()
