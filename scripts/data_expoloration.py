import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df_path = "/home/abraham/tenacademy/Gokada/data/raw"
nb_data = pd.read_csv(f"{df_path}/nb.csv")
driver_location_data = pd.read_csv(f"{df_path}/driver_locations_during_request.csv")

# Check for missing values
print("Missing values in nb.csv:")
print(nb_data.isnull().sum())

print("\nMissing values in driver_locations_during_request.csv:")
print(driver_location_data.isnull().sum())

# Explore data types
print("\nData types in nb.csv:")
print(nb_data.dtypes)

print("\nData types in driver_locations_during_request.csv:")
print(driver_location_data.dtypes)

# Descriptive statistics (summary)
print("\nDescriptive statistics for nb.csv:")
print(nb_data.describe())

print("\nDescriptive statistics for driver_locations_during_request.csv:")
print(driver_location_data.describe(include='all'))  # Include all data types

# Preview the data
print("\nFirst few rows of nb.csv:")
print(nb_data.head())

print("\nFirst few rows of driver_locations_during_request.csv:")
print(driver_location_data.head())

# Explore outliers with boxplots (numerical features)
numerical_features_nb = nb_data.select_dtypes(include=[np.number])  # Assuming numerical data
numerical_features_driver = driver_location_data.select_dtypes(include=[np.number])

if len(numerical_features_nb) > 0:
  numerical_features_nb.plot(kind='box')
  plt.title("Boxplots for numerical features in nb.csv")
  plt.show()

if len(numerical_features_driver) > 0:
  numerical_features_driver.plot(kind='box')
  plt.title("Boxplots for numerical features in driver_locations_during_request.csv")
  plt.show()

# (Optional) Create additional visualizations like histograms for further exploration
