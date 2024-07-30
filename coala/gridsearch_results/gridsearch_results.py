import pandas as pd
import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns


data_folder = '../gridsearch_results'

# Initialize a dictionary to store separated data
data_dict = {}

# Iterate through all files in the folder
for filename in os.listdir(data_folder):
    if filename.endswith('.txt'):
        filepath = os.path.join(data_folder, filename)

        # Parse the filename
        parts = filename.split('_')
        dataset_name = parts[0]

        # Handle method names starting with 'un' and others
        if parts[1].startswith('un'):
            method_name = '_'.join(parts[1:parts.index('grid')])
        else:
            method_name = parts[1]

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

        # Convert Silhouette to percentage and round to one decimal place
        pivot_df = (pivot_df * 100).round(1)

        # Plot the heatmap
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap='viridis', ax=axes[i],
                    cbar=i == 0, cbar_ax=cbar_ax if i == 0 else None,
                    vmin=vmin, vmax=vmax, annot_kws={"size": 12, "weight": "bold"})

        # Map the method name using the dictionary
        method_display_name = method_name_mapping.get(key[1], key[1])
        axes[i].set_title(f"Method: {method_display_name}", fontsize=16, weight='bold')

        # Set labels only on the outer left and bottom
        if i % 3 == 0:
            axes[i].set_ylabel('Npc (Components)', fontsize=16, weight='bold')
        else:
            axes[i].set_ylabel('')

        if i >= 6:
            axes[i].set_xlabel('k (Clusters)', fontsize=16, weight='bold')
        else:
            axes[i].set_xlabel('')

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Silhouette Heatmaps for Dataset: {dataset_name}", fontsize=24, weight='bold')
    plt.subplots_adjust(right=0.9, top=0.9)
    plt.show()