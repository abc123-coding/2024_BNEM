import pandas as pd
import numpy as np

# Read the sorted input CSV file
df = pd.read_csv("sorted.csv")  # ← replace with your actual input file if needed

# Compute mean and standard deviation of FRET values for each sequence type
mean_std_N5 = df.groupby('N5_seq')['N5_FRET'].agg(['mean', 'std']).reset_index()
mean_std_N50 = df.groupby('N50_seq')['N50_FRET'].agg(['mean', 'std']).reset_index()
mean_std_N500 = df.groupby('N500_seq')['N500_FRET'].agg(['mean', 'std']).reset_index()
mean_std_N5M10 = df.groupby('N5M10_seq')['N5M10_FRET'].agg(['mean', 'std']).reset_index()
mean_std_N5M100 = df.groupby('N5M100_seq')['N5M100_FRET'].agg(['mean', 'std']).reset_index()

# Merge all mean/std DataFrames side by side
data_frames = [mean_std_N5, mean_std_N50, mean_std_N500, mean_std_N5M10, mean_std_N5M100]
merged_df = pd.concat(data_frames, axis=1)

# Rename columns for clarity
merged_df.columns = ['N5_seq', 'N5_mean', 'N5_std',
                     'N50_seq', 'N50_mean', 'N50_std',
                     'N500_seq', 'N500_mean', 'N500_std',
                     'N5M10_seq', 'N5M10_mean', 'N5M10_std',
                     'N5M100_seq', 'N5M100_mean', 'N5M100_std']

# Select only the mean values for the final summary table
final_df = merged_df[['N5_seq', 'N5_mean', 'N50_mean','N500_mean', 'N5M10_mean','N5M100_mean']]

# Save the result as a CSV file
final_df.to_csv("mean.csv", index=False)  # ← replace with your desired output path