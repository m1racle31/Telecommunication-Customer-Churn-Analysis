import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from load_data import read_csv_data  # Import function from load_data.py
from data_preprocessing import preprocess_data  # Import the preprocessing function


def handle_imbalance(df):
    # Separate predictor variables (X) and target variable (y)
    X = df[
        [
            "Senior Citizen & Dependents",
            "Partner",
            "Tenure",
            "Service",
            "Contract",
            "Paperless Billing",
            "Payment Method",
            "Monthly Charges",
        ]
    ]
    y = df["Churn"]

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X)

    # Counting the number of churn cases before SMOTE
    churn_counts_before = y.value_counts()

    # Applying SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

    # Counting the number of churn cases after SMOTE
    churn_counts_after = y_resampled.value_counts()

    return churn_counts_before, churn_counts_after


def plot_churn_distribution(churn_counts_before, churn_counts_after):
    # Plotting the comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x=churn_counts_before.index, y=churn_counts_before.values)
    plt.title("Churn Distribution Before SMOTE")
    plt.xlabel("Churn")
    plt.ylabel("Customer Count")

    plt.subplot(1, 2, 2)
    sns.barplot(x=churn_counts_after.index, y=churn_counts_after.values)
    plt.title("Churn Distribution After SMOTE")
    plt.xlabel("Churn")
    plt.ylabel("Customer Count")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the data using the function from load_data.py
    file_path = "../data/Telco Customer Churn.csv"
    df = read_csv_data(file_path)

    # Preprocess the data
    df = preprocess_data(df)

    # Handle imbalance and get churn distribution
    churn_counts_before, churn_counts_after = handle_imbalance(df)

    # Plot churn distribution
    plot_churn_distribution(churn_counts_before, churn_counts_after)
