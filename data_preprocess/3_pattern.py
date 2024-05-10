import pandas as pd
import re

input_file_path = r'C:\Users\chw10\2024BNEM\data\mean.csv'
output_directory_path = r'C:\Users\chw10\2024BNEM\data'

def generate_all_combinations(length):
    bases = ['A', 'T', 'G', 'C', '.']
    combinations = []
    generate_combinations_recursive('', length, bases, combinations)
    return combinations

def generate_combinations_recursive(current_seq, length, bases, combinations):
    if length == 0:
        combinations.append(current_seq)
    else:
        for base in bases:
            generate_combinations_recursive(current_seq + base, length - 1, bases, combinations)

length = 5
seq_features = generate_all_combinations(length)

df = pd.read_csv(input_file_path)
df_seq = df['seq'].dropna()

pattern_results = []
for pattern in seq_features:
    pattern_results.append(df_seq.apply(lambda x: bool(re.search(pattern, x[1:6]))))

df_seq_pattern = pd.concat(pattern_results, axis=1)
df_seq_pattern.columns = seq_features  # 열 이름을 패턴으로 설정

df_combined = pd.concat([df, df_seq_pattern], axis=1)
df_combined.to_csv(f'{output_directory_path}_pattern_mean.csv', index=False)

