from load_data import read_csv_data  # Import function from load_data.py


# Define preprocessing functions
def rename_columns(df):
    df.rename(
        columns={
            "customerID": "Customer ID",
            "gender": "Gender",
            "SeniorCitizen": "Senior Citizen",
            "tenure": "Tenure",
            "PhoneService": "Phone Service",
            "MultipleLines": "Multiple Lines",
            "InternetService": "Internet Service",
            "OnlineSecurity": "Online Security",
            "OnlineBackup": "Online Backup",
            "DeviceProtection": "Device Protection",
            "TechSupport": "Tech Support",
            "StreamingTV": "Streaming TV",
            "StreamingMovies": "Streaming Movies",
            "PaperlessBilling": "Paperless Billing",
            "PaymentMethod": "Payment Method",
            "MonthlyCharges": "Monthly Charges",
            "TotalCharges": "Total Charges",
        },
        inplace=True,
    )
    return df


def check_duplicates(df):
    duplicated_id = df.duplicated(subset=["Customer ID"]).sum()
    print(f"Number of duplicated IDs: {duplicated_id}")
    return df


def check_missing_values(df):
    null_value = df.isnull().sum()
    print(f"Missing values in each column:\n{null_value}")
    return df


def merge_senior_dependents(df):
    df["Senior Citizen"] = df["Senior Citizen"].replace({1: "Yes", 0: "No"})
    df["Senior Citizen & Dependents"] = df["Senior Citizen"] + "_" + df["Dependents"]
    new_col = df.pop("Senior Citizen & Dependents")
    df.insert(2, "Senior Citizen & Dependents", new_col)
    mapping = {
        "Yes_Yes": "Senior Citizen with Dependents",
        "No_Yes": "Non-Senior Citizen with Dependents",
        "Yes_No": "Senior Citizen without Dependents",
        "No_No": "Non-Senior Citizen without Dependents",
    }
    df["Senior Citizen & Dependents"] = df["Senior Citizen & Dependents"].map(mapping)
    return df


def merge_services(df):
    df["Service"] = df["Phone Service"] + "_" + df["Internet Service"]
    new_col = df.pop("Service")
    df.insert(6, "Service", new_col)
    mapping = {
        "Yes_DSL": "Phone & DSL Internet",
        "Yes_Fiber optic": "Phone & Fiber Optic Internet",
        "Yes_No": "Phone Only",
        "No_DSL": "DSL (Internet Only)",
        "No_Fiber optic": "Fiber Optic (Internet Only)",
        "No_No": "No Service",
    }
    df["Service"] = df["Service"].map(mapping)
    return df


def drop_irrelevant_columns(df):
    drop_columns = [
        "Customer ID",
        "Senior Citizen",
        "Dependents",
        "Total Charges",
        "Phone Service",
        "Internet Service",
        "Multiple Lines",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]
    df.drop(columns=drop_columns, inplace=True)
    return df


# Function to preprocess data
def preprocess_data(df):
    df = rename_columns(df)
    df = check_duplicates(df)
    df = check_missing_values(df)
    df = merge_senior_dependents(df)
    df = merge_services(df)
    df = drop_irrelevant_columns(df)
    return df


# Test data_preprocessing as standalone
if __name__ == "__main__":
    # Load the data using the function from load_data.py
    file_path = "../data/Telco Customer Churn.csv"
    df = read_csv_data(file_path)

    # Preprocess the data
    df = preprocess_data(df)

    # Display the head of the dataframe after preprocessing
    print("Data after preprocessing:")
    print(df.head())
