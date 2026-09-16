import pandas as pd
import numpy as np

# Q1. Pandas version
print(pd.__version__)

# Q2. Records count
df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\ML ZOOMCAMP\car_fuel_efficiency_2026.csv")

print(df.info())
print(df.shape) 

# Q3. Fuel types
unique_fuel_types = df["fuel_type"].nunique()
print("Number of fuel types:", unique_fuel_types)

# Q4. Missing values
print(df.isnull().sum())
# Count how many columns have at least one missing value
missing_cols_count = (df.isnull().sum() > 0).sum()
print("Number of columns with missing values:", missing_cols_count)

# Q5. Max fuel efficiency
Max_Fuel_Efficiency = df["fuel_efficiency_mpg"].max()
print("The maximum fuel efficiency is ", Max_Fuel_Efficiency)

# Q6. Median value of horsepower
Median_of_HousePower = df["horsepower"].median()
print("The median value of the HousePower is: ", Median_of_HousePower)

Mode_of_HousePower = df["horsepower"].mode()
print("The Mode of the HousePower is: ", Mode_of_HousePower )

df['fill_horsepower'] = df['horsepower'].fillna(Mode_of_HousePower)

final_median = df["fill_horsepower"].median()
print("The final median value of the HousePower is: ", final_median)

# Q7. Sum of weights

# 1. Filter for cars from Asia
asia_cars = df[df["origin"] == "Asia"]

# 2. Select columns, get the first 7 values, and convert to a NumPy array (X)
X = asia_cars[["vehicle_weight", "model_year"]].head(7).values

# 3. Compute XTX
XTX = X.T.dot(X)

# 4. Invert XTX
XTX_inv = np.linalg.inv(XTX)

# 5. Create array y
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# 6. Multiply the inverse of XTX with the transpose of X, then multiply by y
w = XTX_inv.dot(X.T).dot(y)

# 7. Compute the sum of all the elements of the result
total_sum = w.sum()

print("The sum of all elements of w is:")
print(total_sum)


