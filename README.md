# Project Title: Telecommunication Customer Churn Analysis

This project focuses on analyzing customer data from a telecommunications company and building a Logistic Regression prediction model to predict customer churn and retention.

## Project Description

This project uses a fictional telecommunications company dataset, containing data on 7043 customers with various attributes such as tenure, contract type, services, payment method, dependents status, etc. The data reveals that 5174 customers chose retention, while 1869 chose to churn, which constitutes 26.5% of the total.

The relatively high churn rate raises the following objectives:
- How to increase customer retention rates?
- What variables have the greatest influence on Telecommunication Customer Churn?
- Can we build a customer churn probability prediction model to reduce churn rates?

## Directory Structure

- `data/`: Contains the dataset used in this project.
- `notebooks/`: Jupyter Notebooks for data exploration.
- `results/`: Contains model results, including evaluation and trained models.
  - `evaluation/`: Model evaluation results like the confusion matrix and classification report.
  - `model/`: The trained model saved in `.joblib` format.
  - `recommendation/`: Recommendations generated from the Telecommunication Customer Churn analysis aimed at improving retention strategies.
- `src/`: Source code for the project.
  - `load_data.py`: Function to load data.
  - `data_preprocessing.py`: Function for data preprocessing.
  - `exploratory_data_analysis.py`: For data exploration.
  - `statistical_tests.py`: For correlation analysis and Chi-Square tests.
  - `imbalance_handling.py`: Demonstrates the comparison of training data before and after SMOTE.
  - `prediction_model.py`: Function to train and save the model.
  - `main.py`: The main script for running prediction and model evaluation.

## Prerequisites

Ensure you have installed all required dependencies. You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

## How to Run

To run the **Telecommunication Customer Churn Analysis** project, follow these steps:

### 1. Clone the Repository
First, clone the repository to your local machine using the following command:
```bash
git clone https://github.com/m1racle31/Telecommunication-Customer-Churn-Analysis
cd Telecommunication-Customer-Churn-Analysis
```

### 2. Install Dependencies
Ensure all required dependencies are installed. This project uses Python, and the dependencies are listed in the `requirements.txt` file.

Install them using:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
Make sure the dataset is located in the `data/` folder. If the dataset is not available in the repository, download the **Telco Customer Churn** dataset and place it inside the `data/` directory.

### 4. Run the Preprocessing and Model Evaluation

You can execute the main script to load the data, preprocess it, load the trained model, and evaluate its performance. Simply run the following command:

```bash
python src/main.py
```

This script will:
- Load the data from the `data/` folder.
- Preprocess the data, applying transformations to both numerical and categorical variables.
- Load the pre-trained Logistic Regression model from `results/model/`.
- Make predictions on the test data.
- Evaluate the model’s performance by calculating accuracy, generating the confusion matrix, and displaying the classification report.

### 5. Additional Exploration with Notebooks (Optional)
If you want to explore the data or run specific analyses interactively, you can use the Jupyter Notebooks provided in the `notebooks/` folder.

To launch Jupyter Notebook:
```bash
jupyter notebook
```
Open any `.ipynb` files from the `notebooks/` directory for further exploration or analysis.

### 6. Output and Recommendations
Once the script completes:
- Model evaluation outputs such as the confusion matrix and classification report will be displayed.
- The trained model will be saved in the results/model/ folder when running the prediction_model.py file.
- You can create and save the recommendations for improving customer retention based on the churn analysis in `results/recommendation/`

## Code Explanation
This section provides a brief overview of the main source code files and their key functionalities.

- `src/load_data.py`: Contains functions for loading and reading the dataset.
  - `read_csv_data(file_path)`: Reads the CSV file from the given file path and returns a pandas DataFrame.

- `src/data_preprocessing.py`: Includes functions for preprocessing the data.
  - `preprocess_data(df)`: Handles missing values, feature engineering, and transforms the dataset to prepare it for modeling.

- `src/exploratory_data_analysis.py`: Functions used for performing exploratory data analysis (EDA) on the dataset.
  - `perform_eda(df)`: Visualizes the dataset and provides insights such as churn rate, distributions of features, etc.

- `src/statistical_tests.py`: Contains code for statistical analysis and correlation testing.
  - `analyze_correlation(df)`: Conducts correlation analysis between features and the target variable to understand relationships.
  - `perform_chi_square(df)`: Performs the Chi-Square test for categorical variables.

- `src/imbalance_handling.py`: Handles class imbalance in the dataset.
  - `handle_imbalance(X_train, y_train)`: Uses SMOTE to resample the training data to balance class distribution.

- `src/prediction_model.py`: Contains the code for building and training the prediction model.
  - `train_model(X_train, y_train)`: Trains a logistic regression model on the resampled dataset.
  - `save_model(model, model_path)`: Saves the trained model in the specified path.

- `src/main.py`: The main script for running the prediction and evaluation process.
  - `main()`: Orchestrates the full workflow including data loading, preprocessing, loading the pre-trained model, predicting on the test set, and evaluating the model.

- `results/`: This folder contains the following subdirectories:
  - `evaluation/`: Stores the evaluation results such as confusion matrix and classification reports.
  - `model/`: Contains the trained model saved in `.joblib` format.
  - `recommendation/`: Contains recommendations from the analysis of customer churn data as a strategy to improve retention rates.

## Contact

For any questions or further assistance, feel free to reach out to me:

- **Email**: [yoshuaaugusta31@gmail.com](mailto:yoshuaaugusta31@gmail.com)
- **LinkedIn**: [Yoshua Augusta](https://www.linkedin.com/in/yoshua-augusta/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.