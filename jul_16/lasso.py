from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd


def load_data_from_csv_as_df(input_file_path):
    return pd.read_csv(input_file_path, low_memory=False)

def generate_all_patterns(length):
    bases = ['A', 'T', 'G', 'C', '.']
    patterns = []
    generate_patterns_recursive('', length, bases, patterns)
    return patterns


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


def generate_patterns_recursive(current_seq, length, bases, patterns):
    if length == 0:
        patterns.append(current_seq)
    else:
        for base in bases:
            generate_patterns_recursive(current_seq + base, length - 1, bases, patterns)


output_directory_path = r'C:\Users\chw10\2024BNEM\data'
sol = ['N5', 'N50', 'N500', 'N5M10', 'N5M100']
base = ['A', 'C', 'G', 'T']

patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots = find_patterns_according_to_the_number_of_points(length=5)

def lasso(sol, base, ):

    for s in sol:

        feature = []
        X, y = [], []

        input_file_path = 'input_file_path'
        df = load_data_from_csv_as_df(input_file_path)

        for b in base:
            for i in range(1, 6):
                X = df[patterns_with_two_dots] # change 'patterns_with_two_dots' if you want to see other patterns result
                y = df[f'{s}_FRET']

                X_clean = X.dropna(axis=0)
                y_clean = y[X.index.isin(X_clean.index)]

        # Lasso Regression 
        lasso = Lasso(alpha=0.001)  # alpha for controlling the degree of regularization
        lasso.fit(X_clean, y_clean)


        # show coefficients
        coef_abs = np.abs(lasso.coef_)
        feature_importance = sorted(zip(coef_abs, patterns_with_two_dots), reverse=True) # change 'patterns_with_two_dots' if you want to see other patterns result


        print(f'{s}')
        for importance, feature in feature_importance:
            if importance != 0:
                print(f"{feature}: {importance}")
        print('\n')



lasso(sol, base, data_type='all')
