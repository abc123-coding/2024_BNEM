import pandas as pd
import re

# Define input/output file paths (modify as needed)
input_file_path = "data/sorted.csv"
output_directory_path = "data"

# List of solution types to process
sol = ['N5', 'N50', 'N500', 'N5M10', 'N5M100']

# Generates all base patterns with a specific number of wildcards (dots)
def find_patterns_according_to_the_number_of_points(length):

    def generate_all_patterns(length):
        def generate_patterns_recursive(current_seq, length, bases, patterns):
            if length == 0:
                patterns.append(current_seq)
            else:
                for base in bases:
                    generate_patterns_recursive(current_seq + base, length - 1, bases, patterns)

        bases = ['A', 'T', 'G', 'C', '.']
        patterns = []
        generate_patterns_recursive('', length, bases, patterns)
        return patterns

    all_patterns = generate_all_patterns(length)

    # Categorize patterns by number of wildcards
    patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots = [], [], [], []

    for pattern in all_patterns:
        dot_cnt = pattern.count('.')
        if dot_cnt == 1:
            patterns_with_one_dot.append(pattern)
        elif dot_cnt == 2:
            patterns_with_two_dots.append(pattern)
        elif dot_cnt == 3:
            patterns_with_three_dots.append(pattern)
        elif dot_cnt == 4:
            patterns_with_four_dots.append(pattern)

    return patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots

# Set pattern length
length = 5

# Generate patterns with wildcards
patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots = find_patterns_according_to_the_number_of_points(length)

# Read input data
df = pd.read_csv(input_file_path)

# Apply pattern search to each sequence group
for s in sol:

    df_seq = df[f'{s}_seq'].dropna()
    df_fret = df[f'{s}_FRET'].dropna()

    pattern_results_with_four_dots = []

    # Apply regex match for each 4-dot pattern to each sequence
    for pattern in patterns_with_four_dots:
        pattern_results_with_four_dots.append(
            df_seq.apply(lambda x: bool(re.search(pattern, x[1:6])))
        )

    # Compile results into DataFrame
    df_pattern_with_four_dots = pd.concat(pattern_results_with_four_dots, axis=1)
    df_pattern_with_four_dots.columns = patterns_with_four_dots

    # Merge with original DataFrame
    df_combined = pd.concat([df, df_pattern_with_four_dots], axis=1)

    # Export result
    df_combined.to_csv(f'{output_directory_path}_{s}_pattern_with_four_dots.csv', index=False)
