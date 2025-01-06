import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def load_data_from_csv_as_df(input_file_path):
    return pd.read_csv(input_file_path, low_memory=False)


def generate_all_patterns(length):
    bases = ['A', 'T', 'G', 'C', '.']
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
            if p == '.':
                dot_cnt += 1

        if dot_cnt == 1:
            patterns_with_one_dot.append(pattern)
        elif dot_cnt == 2:
            patterns_with_two_dots.append(pattern)
        elif dot_cnt == 3:
            patterns_with_three_dots.append(pattern)
        elif dot_cnt == 4:
            patterns_with_four_dots.append(pattern)

    return patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots


def plot_graph(sol, y_test, y_pred):

    y_pred = np.array(y_pred)
    y_test = np.array(y_test)

    sample_size = 100

    np.random.seed(42)
    indices = np.random.choice(len(y_test), size=sample_size, replace=False)

    y_test_sample = y_test[indices]
    y_pred_sample = y_pred[indices]
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_sample, y_pred_sample, color='blue', alpha=0.5)
    plt.plot([min(y_test_sample), max(y_test_sample)], [min(y_test_sample), max(y_test_sample)], color='red', linestyle='--')
    plt.title(f"{sol} FRET")
    plt.xlabel("Actual FRET")
    plt.ylabel("Predicted FRET")
    plt.grid(True)
    plt.show()


output_directory_path = r'C:\Users\chw10\2024BNEM\data'
sol = ['N5', 'N50', 'N500', 'N5M10', 'N5M100']
base = ['A', 'C', 'G', 'T']


patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots = find_patterns_according_to_the_number_of_points(length=5)

def lasso(sol, base):

    pattern_with = patterns_with_four_dots

    for s in sol:

        input_file_path = f'C:\\Users\\chw10\\2024BNEM\data\\data_{s}_pattern_with_four_dots.csv'
        
        df = load_data_from_csv_as_df(input_file_path)

        for b in base:
            for i in range(1, 6):
                X = df[pattern_with]
                y = df[f'{s}_FRET']

                X_clean = X.dropna(axis=0)
                y_clean = y[X.index.isin(X_clean.index)]

        # Lasso Regression
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

        lasso = Lasso(alpha=0.001)  # alpha for controlling the degree of regularization
        lasso.fit(X_train, y_train)

        y_pred = lasso.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(s)
        print(f'MAE: {mae}')
        print(f'RMSE: {rmse}')
        print(f"R^2 score: {r2}")



        # # show coefficients
        # #coef_abs = np.abs(lasso.coef_)
        # feature_importance = sorted(zip(lasso.coef_, pattern_with), reverse=True)
        #
        # print(f'{s}')
        # for importance, feature in feature_importance:
        #     if importance != 0:
        #         print(f"{feature}: {importance}")
        # print('\n')

        plot_graph(s, y_test, y_pred)


lasso(sol, base)