import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

years = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
ieee_data = np.array([40, 53, 66, 83, 109, 167, 227, 358, 464, 405])
wos_data = np.array([71, 94, 95, 137, 170, 224, 321, 332, 449, 484])
scopus_data = np.array([119, 139, 141, 222, 259, 359, 532, 645, 698, 906])

data_matrix = [ieee_data, wos_data, scopus_data]
row_names = ['IEEE Xplore', 'Web of Science', 'Scopus']

colors = ["#A7FFF0", "#FFB8B4", "#A8E6FF"]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

fig = plt.figure(figsize=(14, 10), dpi=150)
ax = fig.add_subplot(111, projection='3d')

ls = LightSource(azdeg=-60, altdeg=65)

x_pos = np.arange(len(years))
bar_width = 0.4
bar_depth = 0.2

for i, data in enumerate(data_matrix):
    y_pos = np.full_like(x_pos, i)
    z_pos = np.zeros_like(x_pos)

    dx = bar_width
    dy = bar_depth
    dz = data

    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz,
             color=colors[i],
             shade=True,
             lightsource=ls,
             alpha=0.9,
             zsort='average')

    for j, val in enumerate(data):
        offset = 25
        ax.text(x_pos[j] + dx / 2, i + dy / 2, val + offset,
                str(val), ha='center', va='bottom', fontsize=9, color='black', zorder=20)

ax.set_xticks(x_pos + bar_width / 2)
ax.set_xticklabels(years)
ax.set_xlabel('\nYears', fontsize=14, labelpad=10)

ax.set_yticks(np.arange(len(row_names)) + bar_depth / 2)
ax.set_yticklabels(row_names, verticalalignment='bottom', horizontalalignment='left', fontsize=11)

ax.set_zlabel('Number of papers', fontsize=14, labelpad=15)
ax.set_zlim(0, 1000)

ax.view_init(elev=30, azim=-55)
ax.set_box_aspect((10, 6, 8))

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.grid(True)
grid_color = (0.9, 0.9, 0.9, 1)
ax.xaxis._axinfo["grid"]['color'] = grid_color
ax.yaxis._axinfo["grid"]['color'] = grid_color
ax.zaxis._axinfo["grid"]['color'] = grid_color

plt.tight_layout()
plt.show()
