import pandas as pd
import numpy as np

# read .csv file
df = pd.read_csv('sorted.csv')

# compute mean and standard deviation based on the sequence
mean_std_N5 = df.groupby('N5_seq')['N5_FRET'].agg(['mean', 'std']).reset_index()
mean_std_N50 = df.groupby('N50_seq')['N50_FRET'].agg(['mean', 'std']).reset_index()
mean_std_N500 = df.groupby('N500_seq')['N500_FRET'].agg(['mean', 'std']).reset_index()
mean_std_N5M10 = df.groupby('N5M10_seq')['N5M10_FRET'].agg(['mean', 'std']).reset_index()
mean_std_N5M100 = df.groupby('N5M100_seq')['N5M100_FRET'].agg(['mean', 'std']).reset_index()

# concat the dataframe by columns
data_frames = [mean_std_N5, mean_std_N50, mean_std_N500, mean_std_N5M10, mean_std_N5M100]
merged_df = pd.concat(data_frames, axis=1)

# column names
merged_df.columns = ['N5_seq', 'N5_mean', 'N5_std',
                     'N50_seq', 'N50_mean', 'N50_std',
                     'N500_seq', 'N500_mean', 'N500_std',
                     'N5M10_seq', 'N5M10_mean', 'N5M10_std',
                     'N5M100_seq', 'N5M100_mean', 'N5M100_std']

# Make the final dataframe by selecting necessary columns 
final_df = merged_df[['N5_seq', 'N5_mean', 'N50_mean','N500_mean', 'N5M10_mean','N5M100_mean']]

# export the final dataframe to csv
final_df.to_csv('mean.csv', index=False)