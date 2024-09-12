import matplotlib.pyplot as plt
import seaborn as sns
from load_data import read_csv_data  # Import function from load_data.py
from data_preprocessing import preprocess_data  # Import the preprocessing function


def visualize_churn_distribution(df):
    # Base visualization
    churn_counts = df["Churn"].value_counts()

    # Plot pie chart for churn distribution
    plt.figure(figsize=(8, 6))
    plt.pie(
        churn_counts,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        colors=["lightcoral", "lightskyblue"],
    )
    plt.title("Customer Churn Distribution", fontsize=15)
    plt.show()


def visualize_tenure_distribution(df):
    # Tenure distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["Tenure"], bins=30, color="gold", edgecolor="black")
    plt.title("Tenure Distribution", fontsize=15)
    plt.xlabel("Tenure", fontsize=12)
    plt.ylabel("Number of Customer", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def visualize_contract_distribution(df):
    # Contract distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="Contract")
    plt.title("Contract Distribution", fontsize=15)
    plt.xlabel("Contract", fontsize=12)
    plt.ylabel("Count of Customer", fontsize=12)
    plt.show()


def visualize_monthly_charges_density(df):
    # Monthly charges density plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Monthly Charges"], kde=True, color="skyblue")
    plt.title("Density Plot of Monthly Charges")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Density")
    plt.show()


def visualize_tenure_by_churn(df):
    # Tenure distribution by Churn Status
    sns.set(style="whitegrid")
    tn = sns.FacetGrid(df, col="Churn", height=6, aspect=1.5)
    tn.map(sns.histplot, "Tenure", bins=20, kde=True, color="skyblue")
    tn.set_axis_labels("Tenure", "Density")
    tn.set_titles("{col_name} Churn Status")

    for ax in tn.axes.flat:
        ax.set_xticks(range(0, df["Tenure"].max() + 1, 5))

    plt.show()


def visualize_churn_rate_per_contract(df):
    # Churn rate per contract type
    sns.set(style="whitegrid")
    ct = sns.catplot(
        x="Contract",
        hue="Churn",
        palette="pastel",
        kind="count",
        data=df,
        aspect=1.5,
        height=5,
    )
    for ax in ct.axes.flat:
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])

    plt.title("Churn Rate per Contract Type")
    plt.ylabel("Customer Count")
    ct.set_xlabels("")
    plt.show()


def visualize_churn_rate_per_payment_method(df):
    # Churn rate per payment method
    sns.set(style="whitegrid")
    pm = sns.catplot(
        x="Payment Method",
        hue="Churn",
        palette="pastel",
        kind="count",
        data=df,
        aspect=1.85,
        height=5,
    )
    for ax in pm.axes.flat:
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])

    plt.title("Churn Rate Across Payment Methods")
    plt.ylabel("Customer Count")
    pm.set_xlabels("")
    plt.show()


def visualize_churn_rate_per_services_package(df):
    # Churn rate per services package
    sns.set(style="whitegrid")
    its = sns.catplot(
        x="Service",
        hue="Churn",
        palette="pastel",
        kind="count",
        data=df,
        aspect=1.75,
        height=6,
    )
    for ax in its.axes.flat:
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])

    plt.title("Churn Rate for Different Services Package")
    plt.ylabel("Customer Count")
    its.set_xlabels("")
    plt.show()


def visualize_churn_rate_per_senior_citizen(df):
    # Churn rate per Senior Citizen and Dependents
    sns.set(style="whitegrid")
    szd = sns.catplot(
        x="Senior Citizen & Dependents",
        hue="Churn",
        palette="pastel",
        kind="count",
        data=df,
        aspect=1.7,
        height=8,
    )
    for ax in szd.axes.flat:
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])

    plt.title("Churn Distribution based on Senior Citizen & Dependents Status")
    plt.ylabel("Customer Count")
    szd.set_xlabels("")
    plt.show()


def visualize_churn_rate_per_gender_and_partner(df):
    # Churn rate per Gender and Partner
    sns.set(style="whitegrid")
    gp = sns.catplot(
        x="Gender",
        hue="Churn",
        col="Partner",
        data=df,
        kind="count",
        palette="pastel",
        height=6,
    )
    for ax in gp.axes.flat:
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])

    plt.subplots_adjust(top=0.85)
    plt.suptitle("Churn Rate per Gender and Partner")
    gp.set_xlabels("")
    plt.show()


if __name__ == "__main__":
    # Load the data using the function from load_data.py
    file_path = "../data/Telco Customer Churn.csv"
    df = read_csv_data(file_path)

    # Preprocess the data
    df = preprocess_data(df)

    # Execute all EDA functions
    visualize_churn_distribution(df)
    visualize_tenure_distribution(df)
    visualize_contract_distribution(df)
    visualize_monthly_charges_density(df)
    visualize_tenure_by_churn(df)
    visualize_churn_rate_per_contract(df)
    visualize_churn_rate_per_payment_method(df)
    visualize_churn_rate_per_services_package(df)
    visualize_churn_rate_per_senior_citizen(df)
    visualize_churn_rate_per_gender_and_partner(df)
