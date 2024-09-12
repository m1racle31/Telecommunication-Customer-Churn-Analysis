import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from load_data import read_csv_data  # Import function from load_data.py
from data_preprocessing import preprocess_data  # Import the preprocessing function


def load_model(model_path):
    # Load saved model
    model = joblib.load(model_path)
    return model


def evaluate_model(y_test, y_pred):
    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Define label names
    labels = ["Retention", "Churn"]

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Customer Churn Prediction", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def main():
    # Load the data
    file_path = "../data/Telco Customer Churn.csv"
    df = read_csv_data(file_path)

    # Preprocess the data
    df = preprocess_data(df)

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

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Column transformer to handle numeric and categorical variables
    numeric_features = ["Tenure", "Monthly Charges"]
    categorical_features = [
        "Senior Citizen & Dependents",
        "Partner",
        "Service",
        "Contract",
        "Paperless Billing",
        "Payment Method",
    ]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop=None)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # Fit and transform the preprocessor on training data
    X_train_prep = preprocessor.fit_transform(X_train)

    # Transform test data using preprocessor fitted on training data
    X_test_prep = preprocessor.transform(X_test)

    # Load saved model
    model = load_model("../results/model/logistic_regression_model.joblib")

    # Predict
    y_pred = model.predict(X_test_prep)

    # Evaluate the model
    evaluate_model(y_test, y_pred)

    # Coefficients for predictor variables
    pd.set_option("display.max_rows", None)
    feature_names = numeric_features + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(
            input_features=categorical_features
        )
    )
    coefficients = model.coef_[0]
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
    print("\nCoefficients:")
    print(coef_df.head(21))


if __name__ == "__main__":
    main()
