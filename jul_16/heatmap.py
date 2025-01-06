import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

solution = ['N5', 'N50', 'N500', 'N5M10', 'N5M100']
base = ['A', 'G', 'C', 'T']

def load_data_from_csv_as_df(input_file_path):
    return pd.read_csv(input_file_path, low_memory=False)

def generate_all_patterns(length):
    bases = base = ['A', 'G', 'C', 'T', '.']
    patterns = []
    generate_patterns_recursive('', length, bases, patterns)
    return patterns

def generate_patterns_recursive(current_seq, length, bases, patterns):
    if length == 0:
        patterns.append(current_seq)
    else:
        for base in bases:
            generate_patterns_recursive(current_seq + base, length - 1, bases, patterns)

def find_patterns_according_to_the_number_of_points(length):

    all_patterns = generate_all_patterns(length)
    patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots = [], [], [], []

    for pattern in all_patterns:
        dot_cnt = 0

        for p in pattern:
            if p == '.' :
                dot_cnt += 1

        if dot_cnt == 1 :
            patterns_with_one_dot.append(pattern)
        elif dot_cnt == 2 :
            patterns_with_two_dots.append(pattern)
        elif dot_cnt == 3 :
            patterns_with_three_dots.append(pattern)
        elif dot_cnt == 4 :
            patterns_with_four_dots.append(pattern)

    return patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots

def concat_strings(str1, str2):

    concat_str = ""

    if len(str1) == len(str2):

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

    else:
        print(f"concat_strs: error, len(str1)({len(str1)}) != len(str2)({len(str2)})")
        return False

def compute_mean_according_to_pattern(df, pattern, sol):

    filtered_df = df[df[pattern] == True]
    df_FRET = filtered_df[f'{sol}_FRET'].dropna()
    FRET_mean = df_FRET.mean()
    return FRET_mean

def plot_heatmap(solution, number_of_dots):

    idx = 0

    for sol in solution:

        idx += 1
        patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots = find_patterns_according_to_the_number_of_points(length=5)


        if number_of_dots == 3:

            # load file
            input_file_path = f'C:\\Users\\chw10\\2024BNEM\\data\\data_{sol}_pattern_with_three_dots.csv'
            directory_path = f'C:\\Users\\chw10\\2024_BNEM\\jul_16\\'
            heatmap_title =  f'{sol}_1vs1'
            df = load_data_from_csv_as_df(input_file_path)

            # data preparation for heatmap
            patterns_for_row = [pattern for pattern in patterns_with_four_dots]
            patterns_for_col = [pattern for pattern in patterns_with_four_dots]

            heatmap_data = [[0 for _ in range(len(patterns_for_row))] for _ in range(len(patterns_for_col))]

            for i in range(len(patterns_for_row)):
                for j in range(len(patterns_for_col)):
                    target_pattern = concat_strings(patterns_for_row[i], patterns_for_col[j])

                    if target_pattern:
                        heatmap_data[i][j] = compute_mean_according_to_pattern(df=df, pattern=target_pattern, sol=sol)

            # plot heatmap

            mask = np.array([[value == 0 for value in row] for row in heatmap_data])
            plt.figure(figsize=(30, 16))
            ax = sns.heatmap(heatmap_data, annot=True, fmt=".4g", cmap='coolwarm', annot_kws={'size':16}, mask=mask)

            ax.set_xticklabels(patterns_for_col, rotation=60, ha='right',  size = 16)
            ax.set_yticklabels(patterns_for_row, rotation=0, size = 16)

            plt.title(heatmap_title, size = 24)

            output_file_path = directory_path + str(idx) + '_' + heatmap_title + '.png'
            plt.savefig(output_file_path)


        elif number_of_dots == 2:

            # load file
            idx += 1
            input_file_path = f'C:\\Users\\chw10\\2024BNEM\\data\\data_{sol}_pattern_with_two_dots.csv'
            directory_path = f'C:\\Users\\chw10\\2024_BNEM\\jul_16\\'
            heatmap_title = f'{sol}_2vs1'
            df = load_data_from_csv_as_df(input_file_path)

            # data preparation for heatmap
            patterns_for_row = [pattern for pattern in patterns_with_four_dots] # 20
            patterns_for_col = [pattern for pattern in patterns_with_three_dots if pattern.endswith('.')] # 4C2 * 4^2 = 96

            heatmap_data = [[0 for _ in range(len(patterns_for_col))] for _ in range(len(patterns_for_row))]

            for i in range(len(patterns_for_row)):
                for j in range(len(patterns_for_col)):

                    target_pattern = concat_strings(patterns_for_row[i], patterns_for_col[j])

                    if target_pattern:
                        heatmap_data[i][j] = compute_mean_according_to_pattern(df=df, pattern=target_pattern, sol=sol)

            # plot heatmap
            mask = np.array([[value == 0 for value in row] for row in heatmap_data])
            plt.figure(figsize=(80, 24))

            ax = sns.heatmap(heatmap_data, annot=True, fmt=".4g", cmap='coolwarm', vmin=0.47, vmax=0.8, annot_kws={'size': 11}, mask=mask)
            ax.set_xticklabels(patterns_for_col, rotation=60, ha='right', size = 16)
            ax.set_yticklabels(patterns_for_row, rotation=0, size = 16)

            plt.title(heatmap_title, size = 24)

            output_file_path = directory_path + str(idx) + '_' + heatmap_title + '.png'
            plt.savefig(output_file_path)


plot_heatmap(solution=solution, number_of_dots=3)
plot_heatmap(solution=solution, number_of_dots=2)
