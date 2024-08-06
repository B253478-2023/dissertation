import pandas as pd
import  numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

data_folder = '../grid_search_results'

# Initialize a dictionary to store separated data
data_dict = {}

# Iterate through all files in the folder
for filename in os.listdir(data_folder):
    if filename.endswith('.txt'):
        filepath = os.path.join(data_folder, filename)

        # Parse the filename
        parts = filename.split('_')
        dataset_name = 'terrapene'

        # Handle method names starting with 'un' and others
        if parts[2].startswith('un'):
            method_name = '_'.join(parts[2:parts.index('grid')])
        else:
            method_name = parts[2]

        with open(filepath, 'r') as file:
            # Each line is a JSON object
            for line in file:
                if line.strip():
                    try:
                        line = re.sub(r'{"nmi best params: Npc":', '{"Npc":', line)
                        data = json.loads(line.strip())

                        # Create a unique key
                        key = (dataset_name, method_name)
                        if key not in data_dict:
                            data_dict[key] = []

                        data_dict[key].append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {filename}: {e}")
                        continue

# Define the order of methods to ensure consistency
method_order = [
    'un_rtlda', 'un_trlda', 'un_lda', 'swulda', 'un_rtcdlda', 'un_trcdlda',
    'un_kfdapc', 'un_rtalda', 'un_tralda'
]
method_name_mapping = {
    'swulda': 'SWULDA',
    'un_kfdapc': 'Un-KFDAPC',
    'un_lda': 'Un-LDA-Km',
    'un_rtlda': 'Un-RTLDA',
    'un_rtcdlda': 'Un-RTLDA(CD)',
    'un_trlda': 'Un-TRLDA',
    'un_tralda': 'Un-TRLDA(A)',
    'un_trcdlda': 'Un-TRLDA(CD)',
    'un_rtalda': 'Un-RTLDA(A)'
}

# Plotting combined heatmaps
for dataset_name in set(k[0] for k in data_dict.keys()):
    # Sort methods based on the defined order
    methods = [key for key in data_dict.keys() if key[0] == dataset_name]
    methods.sort(key=lambda x: method_order.index(x[1]) if x[1] in method_order else len(method_order))
    num_methods = len(methods)

    # Create a 3x3 grid without constrained layout
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    # Define color range
    vmin, vmax = 0, 100

    # Create a separate axis for the color bar
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i, key in enumerate(methods):
        data_list = data_dict[key]
        df = pd.DataFrame(data_list)
        df_grouped = df.groupby(['Npc', 'k']).agg({'Silhouette': 'mean'}).reset_index()
        pivot_df = df_grouped.pivot(index='Npc', columns='k', values='Silhouette')

        # Calculate maximum value using original scores
        max_val = pivot_df.max().max()

        # Create a rounded version of the DataFrame for annotation
        pivot_df_rounded = (pivot_df * 100).round(2)

        # Plot the heatmap
        ax = axes[i]
        sns.heatmap(pivot_df_rounded, annot=True, fmt=".2f", cmap='viridis', ax=ax,
                    cbar=i == 0, cbar_ax=cbar_ax if i == 0 else None,
                    vmin=vmin, vmax=vmax, annot_kws={"size": 12, "weight": "bold"})

        # Highlight the maximum value in red using original scores for comparison
        max_indices = pivot_df[pivot_df == max_val].stack().index.tolist()
        for idx in max_indices:
            ax.text(idx[1] + 0.5, idx[0] + 0.5, f"{pivot_df_rounded.at[idx]:.2f}", color='red',
                    ha='center', va='center', fontsize=12, weight='bold')

        # Map the method name using the dictionary
        method_display_name = method_name_mapping.get(key[1], key[1])
        ax.set_title(f"Method: {method_display_name}", fontsize=16, weight='bold')

        # Set labels only on the outer left and bottom
        if i % 3 == 0:
            ax.set_ylabel('Npc (Components)', fontsize=16, weight='bold')
        else:
            ax.set_ylabel('')

        if i >= 6:
            ax.set_xlabel('k (Clusters)', fontsize=16, weight='bold')
        else:
            ax.set_xlabel('')

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Silhouette Heatmaps for Dataset: {dataset_name}", fontsize=24, weight='bold')
    plt.subplots_adjust(right=0.9, top=0.9)
    plt.show()


for dataset_name in set(k[0] for k in data_dict.keys()):
    # Sort methods based on the defined order
    methods = [key for key in data_dict.keys() if key[0] == dataset_name]
    methods.sort(key=lambda x: method_order.index(x[1]) if x[1] in method_order else len(method_order))
    num_methods = len(methods)

    # Adjust the figure size to increase height and reduce spacing
    fig = plt.figure(figsize=(18, 18))  # Increased height for better visibility
    fig.suptitle(f"Silhouette Sensitivity for Dataset: {dataset_name}", fontsize=24, weight='bold')

    for i, key in enumerate(methods):
        data_list = data_dict[key]
        df = pd.DataFrame(data_list)
        df_grouped = df.groupby(['Npc', 'k']).agg({'Silhouette': 'mean'}).reset_index()
        pivot_df = df_grouped.pivot(index='Npc', columns='k', values='Silhouette')

        # Create a 3D subplot
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')

        # Get X, Y, Z data
        xpos, ypos = np.meshgrid(pivot_df.columns, pivot_df.index)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        dx = dy = 0.3  # reduced width of the bars to prevent overlap
        dz = pivot_df.values.flatten() * 100  # scale to percentage

        # Determine the colors for bars, highlighting the maximum value in red
        max_value = dz.max()
        colors = ['red' if value == max_value else 'blue' for value in dz]

        # Plot 3D bars
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')

        # Adjust view angle
        ax.view_init(elev=30, azim=60)

        # Set axis labels and title
        ax.set_xlabel('k (Clusters)', fontsize=16, weight='bold')
        ax.set_ylabel('Npc (Components)', fontsize=16, weight='bold')
        ax.set_zlabel('Silhouette (%)', fontsize=16, weight='bold')  # Corrected label to "Silhouette (%)"
        method_display_name = method_name_mapping.get(key[1], key[1])
        ax.set_title(f"Method: {method_display_name}", fontsize=20, weight='bold')

        # Adjust tick parameters for font size
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='z', labelsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.05, top=0.90, hspace=0.15, wspace=0.2)  # Adjusted spacing
    plt.show()