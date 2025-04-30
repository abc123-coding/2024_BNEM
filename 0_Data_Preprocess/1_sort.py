import pandas as pd

file_path = r'C:\Users\chw10\2024BNEM\data.csv'

# read csv file and load as dataframe
df = pd.read_csv(file_path)

df_N5 = pd.concat([df['N5_seq'], df['N5_FRET']], axis =1).dropna()
df_N50 = pd.concat([df['N50_seq'], df['N50_FRET']], axis =1).dropna()
df_N500 = pd.concat([df['N500_seq'], df['N500_FRET']], axis =1).dropna()
df_N5M10 = pd.concat([df['N5M10_seq'], df['N5M10_FRET']], axis =1).dropna()
df_N5M100 = pd.concat([df['N5M100_seq'], df['N5M100_FRET']], axis =1).dropna()


# sort in alphabetical order
sorted_df_N5 = df_N5.sort_values(by='N5_seq', ignore_index = True)
sorted_df_N50 = df_N50.sort_values(by='N50_seq', ignore_index = True)
sorted_df_N500 = df_N500.sort_values(by='N500_seq', ignore_index = True)
sorted_df_N5M10 = df_N5M10.sort_values(by='N5M10_seq', ignore_index = True)
sorted_df_N5M100 = df_N5M100.sort_values(by='N5M100_seq', ignore_index = True)

# make the dataframe
sorted_df = pd.concat(
    [sorted_df_N5, sorted_df_N50, sorted_df_N500, sorted_df_N5M10, sorted_df_N5M100], axis = 1
)

new_file_path = r'C:\Users\chw10\2024BNEM\sorted.csv'
sorted_df.to_csv(new_file_path, index=False)
