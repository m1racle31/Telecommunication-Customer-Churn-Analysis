import pandas as pd


# Define the function to load data
def read_csv_data(file_path):
    # Load data from CSV file
    df = pd.read_csv(file_path)

    return df


if __name__ == "__main__":
    # Define the path to the dataset
    file_path = "../data/Telco Customer Churn.csv"

    # Load the data
    df = read_csv_data(file_path)

    # Display dataframe information
    print("Dataframe Information:")
    df.info()

    # Display dataframe columns
    print("\nDataframe Columns:")
    print(df.columns)
