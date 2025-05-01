import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Input/output file paths (modify as needed)
input_file_path = "data/sorted.csv"
output_directory_path = "data"
sol = ['N5', 'N50', 'N500', 'N5M10', 'N5M100']
base = ['A', 'C', 'G', 'T']

# Generate all patterns with one wildcard ('.') in a 5-mer sequence
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


# Read CSV file as DataFrame
def load_data_from_csv_as_df(input_file_path):
    return pd.read_csv(input_file_path, low_memory=False)


# Extract FRET values for each sequence, grouped by sequence
def sort_out_FRET_on_basis_of_sequence(df, solution):
    seq_fret_dict = {}
    seq, fret = df[f'{solution}_seq'].dropna(), (df[f'{solution}_FRET'].dropna()).round(4)
    for key, value in zip(seq, fret):
        if 0 < value < 1:  # exclude invalid values
            if key not in seq_fret_dict:
                seq_fret_dict[key] = []
            seq_fret_dict[key].append(value)
    return seq_fret_dict


# Match patterns with one wildcard and collect matching FRET data
def search_seq_pattern(seq_fret_dict, pattern, base):
    dict_pattern_seq_fret = {}
    for p in pattern:
        dict_seq_fret = {}
        for b in base:
            sequence = p.replace('.', b)
            dict_seq_fret[sequence] = seq_fret_dict['T' + sequence + 'TT']  # assumes sequences are flanked
        dict_pattern_seq_fret[p] = dict_seq_fret
    return dict_pattern_seq_fret


# Round up to nearest integer
def ceil(x):
    return int(x) + (x > int(x))


# Generate and save multiple plots per page for given pattern-FRET mappings
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

            plot_data = [(f, seq, pattern) for seq, frets in seq_fret.items() for f in frets]

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

# Run full analysis and plotting for all solution types
df = load_data_from_csv_as_df(input_file_path)
list_patterns = generate_all_patterns_with_a_wildcard(base)

for s in sol:
    seq_fret = sort_out_FRET_on_basis_of_sequence(df, s)
    pattern_seq_fret = search_seq_pattern(seq_fret, list_patterns, base)

    plot_graphs_per_page(pattern_seq_fret, 'violin_strip', 4, 2, f'./{s}_plots', s)
    plot_graphs_per_page(pattern_seq_fret, 'violin', 4, 2, f'./{s}_plots', s)
    plot_graphs_per_page(pattern_seq_fret, 'strip', 4, 2, f'./{s}_plots', s)