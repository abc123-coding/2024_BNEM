import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

solution = ['N5', 'N50', 'N500', 'N5M10', 'N5M100']
base = ['A', 'G', 'C', 'T']

# Load CSV as pandas DataFrame
def load_data_from_csv_as_df(input_file_path):
    return pd.read_csv(input_file_path, low_memory=False)

# Generate all 5-mer patterns using A, G, C, T and wildcard '.'
def generate_all_patterns(length):
    bases = ['A', 'G', 'C', 'T', '.']
    patterns = []
    generate_patterns_recursive('', length, bases, patterns)
    return patterns

# Recursive helper for pattern generation
def generate_patterns_recursive(current_seq, length, bases, patterns):
    if length == 0:
        patterns.append(current_seq)
    else:
        for base in bases:
            generate_patterns_recursive(current_seq + base, length - 1, bases, patterns)

# Separate patterns by number of wildcards
def find_patterns_according_to_the_number_of_points(length):
    all_patterns = generate_all_patterns(length)
    p1, p2, p3, p4 = [], [], [], []

    for pattern in all_patterns:
        dot_cnt = pattern.count('.')
        if dot_cnt == 1:
            p1.append(pattern)
        elif dot_cnt == 2:
            p2.append(pattern)
        elif dot_cnt == 3:
            p3.append(pattern)
        elif dot_cnt == 4:
            p4.append(pattern)

    return p1, p2, p3, p4

# Merge two patterns if compatible (no conflicting bases)
def concat_strings(str1, str2):

    if len(str1) != len(str2):
        print(f"concat_strs: error, len(str1)({len(str1)}) != len(str2)({len(str2)})")
        return False

    concat_str = ""
    for i in range(len(str1)):
        if str1[i] != '.' and str2[i] != '.':
            return False
        elif str1[i] != '.':
            concat_str += str1[i]
        elif str2[i] != '.':
            concat_str += str2[i]
        else:
            concat_str += '.'
    return concat_str

# Compute mean FRET of rows that match the pattern
def compute_mean_according_to_pattern(df, pattern, sol):
    filtered_df = df[df[pattern] == True]
    df_FRET = filtered_df[f'{sol}_FRET'].dropna()
    return df_FRET.mean()

# Plot heatmaps for 3-dot pattern combinations
def plot_heatmap(solution):
    
    idx = 0

    for sol in solution:
        
        idx += 1
        p4 = find_patterns_according_to_the_number_of_points(length=5)

        input_file_path = f'data/data_{sol}_pattern_with_three_dots.csv'
        directory_path = f'result/jul_16/'
        heatmap_title = f'{sol}_1vs1'
        df = load_data_from_csv_as_df(input_file_path)

        patterns_for_row = p4
        patterns_for_col = p4

        # Initialize heatmap matrix
        heatmap_data = [[0 for _ in patterns_for_col] for _ in patterns_for_row]

        # Fill in heatmap data
        for i in range(len(patterns_for_row)):
            for j in range(len(patterns_for_col)):
                target_pattern = concat_strings(patterns_for_row[i], patterns_for_col[j])
                if target_pattern:
                    heatmap_data[i][j] = compute_mean_according_to_pattern(df, target_pattern, sol)

        # Mask zero entries (missing values)
        mask = np.array([[value == 0 for value in row] for row in heatmap_data])

        # Plot the heatmap
        plt.figure(figsize=(30, 16))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".4g",
            cmap='coolwarm',
            mask=mask,
            annot_kws={'size': 16},
        )

        ax.set_xticklabels(patterns_for_col, rotation=60, ha='right', size=16)
        ax.set_yticklabels(patterns_for_row, rotation=0, size=16)
        plt.title(heatmap_title, size=24)

        output_file_path = directory_path + f"{idx}_{heatmap_title}.png"
        plt.savefig(output_file_path)

# Generate heatmaps
plot_heatmap(solution=solution)
