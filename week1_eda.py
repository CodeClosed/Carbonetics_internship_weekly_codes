"""
Week 1 - Day 1
"""
import pandas, numpy, sklearn, xgboost, fastapi
print("All libraries loaded successfully.")

"""
Week 1 - Day 2
"""
"""
NumPy Fundamentals

Topics Covered:
1. Array Creation
2. Array Attributes
3. Reshaping Arrays
4. Indexing and Slicing
5. Arithmetic Operations
6. Aggregation Functions
7. Axis Operations
8. Broadcasting
9. Boolean Masking
10. Random Number Generation
11. Sorting and Unique Values
12. Combining Arrays
13. Matrix Operations
14. Linear Algebra
15. Handling Missing Values
"""

import numpy as np


# ============================================================
# 1. ARRAY CREATION
# ============================================================

print("\n=== ARRAY CREATION ===")

one_d_array = np.array([1, 2, 3, 4])
two_d_array = np.array([[1, 2], [3, 4]])

print("1D Array:")
print(one_d_array)

print("\n2D Array:")
print(two_d_array)

print("\nZeros Array:")
print(np.zeros((3, 3)))

print("\nOnes Array:")
print(np.ones((2, 4)))

print("\nIdentity Matrix:")
print(np.eye(3))

print("\nFilled Array:")
print(np.full((2, 2), 5))


# ============================================================
# 2. ARRAY ATTRIBUTES
# ============================================================

print("\n=== ARRAY ATTRIBUTES ===")

sample_array = np.array([[1, 2], [3, 4]])

print("Shape:", sample_array.shape)
print("Dimensions:", sample_array.ndim)
print("Size:", sample_array.size)
print("Data Type:", sample_array.dtype)


# ============================================================
# 3. RESHAPING ARRAYS
# ============================================================

print("\n=== RESHAPING ARRAYS ===")

print("Reshape to (1,4):")
print(sample_array.reshape(1, 4))

print("\nReshape to (4,1):")
print(sample_array.reshape(4, 1))

print("\nFlattened Array:")
print(sample_array.flatten())


# ============================================================
# 4. INDEXING AND SLICING
# ============================================================

print("\n=== INDEXING AND SLICING ===")

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Element at row 0, column 1:", matrix[0, 1])
print("First Column:", matrix[:, 0])
print("Second Row:", matrix[1, :])


# ============================================================
# 5. ARITHMETIC OPERATIONS
# ============================================================

print("\n=== ARITHMETIC OPERATIONS ===")

numbers = np.array([9, 3, 5])

print("Add 5:", numbers + 5)
print("Subtract 6:", numbers - 6)
print("Divide by 9:", numbers / 9)
print("Multiply by 18:", numbers * 18)


# ============================================================
# 6. ARRAY-TO-ARRAY OPERATIONS
# ============================================================

print("\n=== ARRAY TO ARRAY OPERATIONS ===")

array_a = np.array([1, 2, 3])
array_b = np.array([4, 5, 6])

print("Addition:", array_a + array_b)
print("Subtraction:", array_a - array_b)
print("Multiplication:", array_a * array_b)
print("Division:", array_a / array_b)


# ============================================================
# 7. AGGREGATION FUNCTIONS
# ============================================================

print("\n=== AGGREGATION FUNCTIONS ===")

data = np.array([1, 2, 3, 4, 5])

print("Sum:", np.sum(data))
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Standard Deviation:", np.std(data))
print("Variance:", np.var(data))
print("Minimum:", np.min(data))
print("Maximum:", np.max(data))


# ============================================================
# 8. AXIS OPERATIONS
# ============================================================

print("\n=== AXIS OPERATIONS ===")

matrix_data = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Column-wise Sum:")
print(np.sum(matrix_data, axis=0))

print("Row-wise Sum:")
print(np.sum(matrix_data, axis=1))


# ============================================================
# 9. BROADCASTING
# ============================================================

print("\n=== BROADCASTING ===")

print(matrix_data + 5)


# ============================================================
# 10. BOOLEAN MASKING
# ============================================================

print("\n=== BOOLEAN MASKING ===")

print("Values Greater Than 5:")
print(matrix_data > 5)

print("Values Less Than 5:")
print(matrix_data[matrix_data < 5])


# ============================================================
# 11. RANDOM NUMBER GENERATION
# ============================================================

print("\n=== RANDOM NUMBER GENERATION ===")

np.random.seed(42)

print("Random Floats:")
print(np.random.rand(5))

print("\nRandom Integers:")
print(np.random.randint(1, 100, 10))

print("\nRandom Normal Distribution:")
print(np.random.randn(5))


# ============================================================
# 12. SORTING AND UNIQUE VALUES
# ============================================================

print("\n=== SORTING AND UNIQUE VALUES ===")

unsorted_array = np.array([87, 45, 134, 77, 2, 133, 55, 9, 43])

print("Sorted Array:")
print(np.sort(unsorted_array))

print("\nUnique Values:")
print(np.unique([4, 5, 5, 6, 6, 5]))


# ============================================================
# 13. COMBINING ARRAYS
# ============================================================

print("\n=== COMBINING ARRAYS ===")

first_array = np.array([1, 2, 3])
second_array = np.array([4, 5, 6])

print("Vertical Stack:")
print(np.vstack((first_array, second_array)))

print("\nHorizontal Stack:")
print(np.hstack((first_array, second_array)))

print("\nConcatenate:")
print(np.concatenate((first_array, second_array)))


# ============================================================
# 14. MATRIX OPERATIONS
# ============================================================

print("\n=== MATRIX OPERATIONS ===")

matrix_a = np.array([
    [1, 2],
    [3, 4]
])

matrix_b = np.array([
    [5, 6],
    [7, 8]
])

print("Matrix Multiplication:")
print(matrix_a @ matrix_b)


# ============================================================
# 15. LINEAR ALGEBRA BASICS
# ============================================================

print("\n=== LINEAR ALGEBRA BASICS ===")

print("Determinant:")
print(np.linalg.det(matrix_a))

print("\nInverse:")
print(np.linalg.inv(matrix_a))


# ============================================================
# 16. HANDLING MISSING VALUES
# ============================================================

print("\n=== HANDLING MISSING VALUES ===")

data_with_nan = np.array([1, np.nan, 22, 22, 3, 3, 56])

print("NaN Positions:")
print(np.isnan(data_with_nan))

print("Total NaN Values:")
print(np.isnan(data_with_nan).sum())

"""
Pandas Fundamentals

Topics Covered:
1. Loading Data
2. Dataset Overview
3. Viewing Data
4. Selecting Columns
5. Selecting Rows
6. Filtering Data
7. Sorting Data
8. Creating Columns
9. Renaming Columns
10. Dropping Columns
11. Handling Missing Values
12. Removing Duplicates
13. Aggregation Functions
14. Value Counts
15. GroupBy Operations
16. Unique Values
17. Applying Functions
18. String Operations
19. Index Operations
20. Exporting Data
21. Merge Operations
22. Concatenation
23. Pivot Tables
"""

import pandas as pd


# ============================================================
# 1. LOADING DATA
# ============================================================

print("\n=== LOADING DATA ===")

df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

print(df.head())


# ============================================================
# 2. DATASET OVERVIEW
# ============================================================

print("\n=== DATASET OVERVIEW ===")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

print("\nStatistics:")
print(df.describe())


# ============================================================
# 3. VIEWING DATA
# ============================================================

print("\n=== VIEWING DATA ===")

print("First 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nRandom Sample:")
print(df.sample(5))


# ============================================================
# 4. SELECTING COLUMNS
# ============================================================

print("\n=== SELECTING COLUMNS ===")

print(df["species"])

print(df[
    ["sepal_length", "petal_length"]
])


# ============================================================
# 5. SELECTING ROWS
# ============================================================

print("\n=== SELECTING ROWS ===")

print("First Row:")
print(df.iloc[0])

print("\nFirst Five Rows:")
print(df.iloc[:5])

print("\nUsing loc:")
print(df.loc[0:4])


# ============================================================
# 6. FILTERING DATA
# ============================================================

print("\n=== FILTERING DATA ===")

setosa = df[
    df["species"] == "setosa"
]

print("Setosa Flowers:")
print(setosa.head())

filtered = df[
    (df["sepal_length"] > 5)
    &
    (df["petal_length"] > 4)
]

print("\nMultiple Conditions:")
print(filtered.head())


# ============================================================
# 7. SORTING DATA
# ============================================================

print("\n=== SORTING DATA ===")

print(
    df.sort_values(
        "sepal_length"
    ).head()
)

print(
    df.sort_values(
        "sepal_length",
        ascending=False
    ).head()
)


# ============================================================
# 8. CREATING NEW COLUMNS
# ============================================================

print("\n=== CREATING NEW COLUMNS ===")

df["sepal_area"] = (
    df["sepal_length"]
    *
    df["sepal_width"]
)

print(df.head())


# ============================================================
# 9. RENAMING COLUMNS
# ============================================================

print("\n=== RENAMING COLUMNS ===")

renamed_df = df.rename(
    columns={
        "sepal_length": "SL",
        "sepal_width": "SW"
    }
)

print(renamed_df.head())


# ============================================================
# 10. DROPPING COLUMNS
# ============================================================

print("\n=== DROPPING COLUMNS ===")

dropped_df = renamed_df.drop(
    "SW",
    axis=1
)

print(dropped_df.head())


# ============================================================
# 11. HANDLING MISSING VALUES
# ============================================================

print("\n=== HANDLING MISSING VALUES ===")

df.loc[0, "sepal_length"] = None
df.loc[5, "sepal_length"] = None

print(df.isnull().sum())

df["sepal_length"] = (
    df["sepal_length"]
    .fillna(
        df["sepal_length"].mean()
    )
)

print("\nAfter Filling:")
print(df.isnull().sum())


# ============================================================
# 12. REMOVING DUPLICATES
# ============================================================

print("\n=== REMOVING DUPLICATES ===")

print(
    "Duplicate Rows:",
    df.duplicated().sum()
)

df = df.drop_duplicates()


# ============================================================
# 13. AGGREGATION FUNCTIONS
# ============================================================

print("\n=== AGGREGATION FUNCTIONS ===")

print(
    "Mean:",
    df["petal_length"].mean()
)

print(
    "Max:",
    df["petal_length"].max()
)

print(
    "Min:",
    df["petal_length"].min()
)

print(
    "Sum:",
    df["petal_length"].sum()
)


# ============================================================
# 14. VALUE COUNTS
# ============================================================

print("\n=== VALUE COUNTS ===")

print(
    df["species"]
    .value_counts()
)


# ============================================================
# 15. GROUPBY OPERATIONS
# ============================================================

print("\n=== GROUPBY OPERATIONS ===")

print(
    df.groupby("species")
      .mean()
)

print(
    df.groupby("species")
      ["petal_length"]
      .agg(
          ["mean", "max", "min"]
      )
)


# ============================================================
# 16. UNIQUE VALUES
# ============================================================

print("\n=== UNIQUE VALUES ===")

print(df["species"].unique())

print(
    "Number of Unique Species:",
    df["species"].nunique()
)


# ============================================================
# 17. APPLYING FUNCTIONS
# ============================================================

print("\n=== APPLYING FUNCTIONS ===")

df["petal_length_mm"] = (
    df["petal_length"]
    .apply(
        lambda x: x * 10
    )
)

print(df.head())


# ============================================================
# 18. STRING OPERATIONS
# ============================================================

print("\n=== STRING OPERATIONS ===")

df["species"] = (
    df["species"]
    .str.upper()
)

print(df.head())


# ============================================================
# 19. INDEX OPERATIONS
# ============================================================

print("\n=== INDEX OPERATIONS ===")

indexed_df = df.set_index(
    "species"
)

print(indexed_df.head())

indexed_df = (
    indexed_df
    .reset_index()
)


# ============================================================
# 20. EXPORTING DATA
# ============================================================

print("\n=== EXPORTING DATA ===")

df.to_csv(
    "iris_cleaned.csv",
    index=False
)

print("File Saved Successfully")


# ============================================================
# 21. MERGE OPERATIONS
# ============================================================

print("\n=== MERGE OPERATIONS ===")

students = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["A", "B", "C"]
})

marks = pd.DataFrame({
    "id": [1, 2, 3],
    "score": [80, 90, 70]
})

merged_df = pd.merge(
    students,
    marks,
    on="id"
)

print(merged_df)


# ============================================================
# 22. CONCATENATION
# ============================================================

print("\n=== CONCATENATION ===")

df1 = pd.DataFrame({
    "A": [1, 2]
})

df2 = pd.DataFrame({
    "A": [3, 4]
})

concat_df = pd.concat(
    [df1, df2]
)

print(concat_df)


# ============================================================
# 23. PIVOT TABLES
# ============================================================

print("\n=== PIVOT TABLES ===")

pivot = pd.pivot_table(
    df,
    values="petal_length",
    index="species",
    aggfunc="mean"
)

print(pivot)