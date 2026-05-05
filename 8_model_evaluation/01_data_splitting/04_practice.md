# Practice Exercises: Data Splitting

These exercises are designed to help you build muscle memory for splitting data correctly.

## Exercise 1: The Basic Split
1. Load the `iris` dataset from `sklearn.datasets`.
2. Assign the features to `X` and the target to `y`.
3. Use `train_test_split` to create a 75% train and 25% test split.
4. Set `random_state=42`.
5. Print the shape of `X_train` and `X_test` to verify the math (150 total rows * 0.25 = 37.5).

## Exercise 2: Understanding Random State
1. Create a simple list of numbers: `X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]` and `y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]`.
2. Run `train_test_split(X, y, test_size=0.2, random_state=1)` and print `y_train`.
3. Run it again with `random_state=1`. Did it change?
4. Run it again with `random_state=99`. What happened?
5. **Question**: Why is `random_state` so important when sharing code with a colleague?

## Exercise 3: The Stratification Problem
1. Look at the `y` array you created in Exercise 2. It has exactly 50% `0`s and 50% `1`s.
2. Run `train_test_split(X, y, test_size=0.4, random_state=4)`.
3. Look at `y_test`. Are there two `0`s and two `1`s? (Hint: probably not).
4. Now, run `train_test_split(X, y, test_size=0.4, random_state=4, stratify=y)`.
5. Look at `y_test`. What did `stratify` fix?

## Exercise 4: Cross-Validation Integration
1. Load the `diabetes` dataset (`load_diabetes()`) from sklearn.
2. Initialize a `LinearRegression` model.
3. Instead of splitting the data yourself, use `cross_val_score(model, X, y, cv=5)`.
4. Print the 5 scores.
5. Print the mean and standard deviation of the scores.
