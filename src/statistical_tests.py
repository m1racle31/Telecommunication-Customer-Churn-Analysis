import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from load_data import read_csv_data  # Import function from load_data.py
from data_preprocessing import preprocess_data  # Import the preprocessing function


def calculate_correlation_matrix(df):
    # Select all columns
    all_columns = list(df.columns)

    # Calculate Cramér's V correlation coefficient for each pair of variables
    correlation_matrix = pd.DataFrame(index=all_columns, columns=all_columns)

    for i in range(len(all_columns)):
        for j in range(len(all_columns)):
            if i == j:
                correlation_matrix.iloc[i, j] = (
                    1.0  # The correlation between a variable and itself is 1
                )
            else:
                # Create a contingency table for pairs of categorical variables
                contingency_table = pd.crosstab(df.iloc[:, i], df.iloc[:, j])
                # Calculate Cramér's V correlation coefficient
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n = min(contingency_table.shape)
                correlation = (chi2 / (len(df) * (n - 1))) ** 0.5
                correlation_matrix.iloc[i, j] = correlation

    return correlation_matrix


def plot_correlation_matrix(correlation_matrix):
    # Plot a heatmap of the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix.astype(float),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Correlation Matrix", fontweight="bold")
    plt.show()


def chi_square_tests(df):
    # Categorical variables to be tested
    categorical_vars = [
        "Gender",
        "Senior Citizen & Dependents",
        "Partner",
        "Service",
        "Contract",
        "Paperless Billing",
        "Payment Method",
    ]

    # Significance level
    alpha = 0.05

    # Loop to perform chi-square test for each categorical variable
    for var in categorical_vars:
        # Create contingency table
        contingency_table = pd.crosstab(df[var], df["Churn"])

        # Perform chi-square test and get p-value
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        # Separating p-values based on the filter
        if p_value < alpha:
            print(f"P-value ({var}): {p_value} (Significant)")
        else:
            print(f"P-value ({var}): {p_value} (Not Significant)")


if __name__ == "__main__":
    # Load the data using the function from load_data.py
    file_path = "../data/Telco Customer Churn.csv"
    df = read_csv_data(file_path)

    # Preprocess the data
    df = preprocess_data(df)

    # Calculate and plot the correlation matrix
    correlation_matrix = calculate_correlation_matrix(df)
    plot_correlation_matrix(correlation_matrix)

    # Perform chi-square tests
    chi_square_tests(df)
