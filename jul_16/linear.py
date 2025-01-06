import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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
    patterns_with_no_dot, patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots = [], [], [], [], []

    for pattern in all_patterns:
        dot_cnt = 0

        for p in pattern:
            if p == '.':
                dot_cnt += 1
        if dot_cnt == 0:
            patterns_with_no_dot.append(pattern)
        elif dot_cnt == 1:
            patterns_with_one_dot.append(pattern)
        elif dot_cnt == 2:
            patterns_with_two_dots.append(pattern)
        elif dot_cnt == 3:
            patterns_with_three_dots.append(pattern)
        elif dot_cnt == 4:
            patterns_with_four_dots.append(pattern)

    return patterns_with_no_dot, patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots


output_directory_path = r'C:\Users\chw10\2024BNEM\data'
sol = ['N5', 'N50', 'N500', 'N5M10', 'N5M100']
base = ['A', 'C', 'G', 'T']

patterns_with_no_dot, patterns_with_one_dot, patterns_with_two_dots, patterns_with_three_dots, patterns_with_four_dots = find_patterns_according_to_the_number_of_points(
    length=5)

def linear_regression(sol, base):

    pattern_with = patterns_with_four_dots  # 패턴 정의는 유지

    for s in sol:

        input_file_path = f'C:\\Users\\chw10\\2024BNEM\\data\\data_{s}_pattern_with_four_dots.csv'
        df = load_data_from_csv_as_df(input_file_path)  # 데이터 로드

        for b in base:
            for i in range(1, 6):

                X = df[pattern_with]
                y = df[f'{s}_FRET']

                X_clean = X.dropna(axis=0)
                y_clean = y[X.index.isin(X_clean.index)]

        # Linear Regression 적용
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

        lin_reg = LinearRegression()  # Linear Regression 객체 생성
        lin_reg.fit(X_train, y_train)  # 학습

        y_pred = lin_reg.predict(X_test)  # 예측

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(s)
        print(f'MAE: {mae}')
        print(f'RMSE: {rmse}')
        print(f"R^2 score: {r2}")

linear_regression(sol, base)

