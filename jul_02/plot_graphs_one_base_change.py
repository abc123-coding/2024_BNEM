import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

input_file_path = r'C:\Users\chw10\2024BNEM\data\sorted.csv'
output_directory_path = r'C:\Users\chw10\2024BNEM\data'
sol = ['N5', 'N50', 'N500', 'N5M10', 'N5M100']
base = ['A', 'C', 'G', 'T']

def generate_all_patterns_with_a_wildcard(base):
    patterns = []
    generate_a_pattern_with_a_wildcard('', 0, base, patterns)
    return patterns

def generate_a_pattern_with_a_wildcard(current_seq, seq_length, base, patterns):
    if seq_length == 5:
        if '.' in current_seq:
            patterns.append(current_seq)
    else:
        if '.' not in current_seq:
            generate_a_pattern_with_a_wildcard(current_seq + '.', seq_length + 1, base, patterns)
            for b in base:
                generate_a_pattern_with_a_wildcard(current_seq + b, seq_length + 1, base, patterns)
        else:
            for b in base:
                generate_a_pattern_with_a_wildcard(current_seq + b, seq_length + 1, base, patterns)

def load_data_from_csv_as_df(input_file_path):
    return pd.read_csv(input_file_path, low_memory=False)

def sort_out_FRET_on_basis_of_sequence(df, solution):

    dict = {}
    seq, fret = df[f'{solution}_seq'].dropna(), (df[f'{solution}_FRET'].dropna()).round(4)
    for key, value in zip(seq, fret):
        if key not in dict:
            dict[key] = []
        if value < 1: # rule out error
            dict[key].append(value)
    return dict


def search_seq_pattern(dict, pattern, base):

    # {
    # 'AAAA.': {'AAAAA':[0.xxxx, ...], 'AAAAC':[0.xxxx, ...], 'AAAAG':[0.xxxx, ...], 'AAAAT':[0.xxxx, ...]},
    # ...
    # '.TTTT': {'ATTTT':[0.xxxx, ...], 'CTTTT':[0.xxxx, ...], 'GTTTT':[0.xxxx, ...], TTTTT':[0.xxxx, ...]},
    # }

    dict_pattern_seq_fret = {}

    for p in pattern:
        dict_seq_fret = {}

        for b in base:
            sequence = p.replace('.', b)
            dict_seq_fret[sequence] = dict['T'+sequence+'TT']

        # df_seq_fret = convert_to_df_from_dict(dict_seq_fret)
        dict_pattern_seq_fret[p] = dict_seq_fret

    return dict_pattern_seq_fret

def ceil(x):
    return int(x) + (x > int(x))

def plot_graphs_per_page(dict_pattern_seq_fret, graph_type, row_of_graphs, col_of_graphs, output_file_path, solution):

    output_file_path = output_file_path + '/' + graph_type
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    total_graphs = len(dict_pattern_seq_fret)
    num_of_graphs_per_page = row_of_graphs * col_of_graphs
    total_pages = ceil(total_graphs / num_of_graphs_per_page)

    plot_num = 0
    for page in range(total_pages):
        fig, axes = plt.subplots(row_of_graphs, col_of_graphs, figsize=(16, 16))
        axes = axes.flatten()

        for i in range(num_of_graphs_per_page):
            if plot_num >= total_graphs:
                break

            pattern = list(dict_pattern_seq_fret.keys())[plot_num]
            seq_fret = dict_pattern_seq_fret[pattern]


            # Prepare plot data by converting to dataframe
            plot_data = []
            for seq, fret in seq_fret.items():
                for f in fret:
                    plot_data.append((f, seq, pattern))

            if plot_data:

                plot_df = pd.DataFrame(plot_data, columns=['fret', 'sequence', 'pattern'])
                axes[i].set_ylim(0, 1)
                axes[i].set_title(pattern)

                if graph_type == 'violin':

                    sns.violinplot(ax=axes[i], x='sequence', y='fret', data=plot_df)

                elif graph_type == 'strip':

                    sns.stripplot(ax=axes[i], x='sequence', y='fret', data=plot_df, hue='sequence', legend=False)

                elif graph_type == 'violin_strip':

                    sns.violinplot(ax=axes[i], x="sequence", y="fret", data=plot_df, color="0.8", legend=False)
                    sns.stripplot(ax=axes[i], x="sequence", y="fret", data=plot_df, jitter=True, zorder=1, hue='sequence', legend=False)

            plot_num += 1

        plt.tight_layout()
        output_filename = os.path.join(output_file_path, f'{graph_type}_{solution}_{page + 1}.png')
        plt.savefig(output_filename)
        plt.close(fig)

        print(f"Saved plot page {page + 1} as '{output_filename}'")



df = load_data_from_csv_as_df(input_file_path)
list_patterns = generate_all_patterns_with_a_wildcard(base)

for s in sol:
    seq_fret = sort_out_FRET_on_basis_of_sequence(df, s)
    pattern_seq_fret = search_seq_pattern(seq_fret, list_patterns, base)
    plot_graphs_per_page(dict_pattern_seq_fret=pattern_seq_fret, graph_type = 'violin_strip', row_of_graphs=4, col_of_graphs=2, output_file_path=f'./{s}_plots', solution = s)
    plot_graphs_per_page(dict_pattern_seq_fret=pattern_seq_fret, graph_type = 'violin', row_of_graphs=4, col_of_graphs=2, output_file_path=f'./{s}_plots', solution = s)
    plot_graphs_per_page(dict_pattern_seq_fret=pattern_seq_fret, graph_type = 'strip', row_of_graphs=4, col_of_graphs=2, output_file_path=f'./{s}_plots', solution = s)
