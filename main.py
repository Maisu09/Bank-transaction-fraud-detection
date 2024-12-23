import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest




# def plot_initial_data(data:pandas.DataFrame):
#     fig, axes = matplotlib.pyplot.subplots(2, 2, figsize=(10, 8))
    
#     # ID vs transactioned ammount
#     axes[0, 0].plot(data["AccountID"], data["TransactionAmount"], 'x', label='ID and TransactionAmount')
#     axes[0, 0].set_title('Transaction per person')
#     axes[0, 0].set_xlabel('ID')
#     axes[0, 0].set_ylabel('TransactionAmount')
    
#     # ID vs Balance
#     axes[0, 1].plot(data["AccountID"], data["AccountBalance"], 'x', label='ID and AccountBalance')
#     axes[0, 1].set_title('Balance per person')
#     axes[0, 1].set_xlabel('ID')
#     axes[0, 1].set_ylabel('AccountBalance')
    
#     # ID vs DateOfTransaction
#     axes[1, 0].plot(data["AccountID"], data["TransactionDate"], 'x', label='ID and TransactionDate')
#     axes[1, 0].set_title('Transaction date per person')
#     axes[1, 0].set_xlabel('ID')
#     axes[1, 0].set_ylabel('TransactionDate')
    
#     matplotlib.pyplot.show()
    
#     matplotlib.pyplot.plot(data["AccountBalance"], data["TransactionAmount"], 'x')
#     matplotlib.pyplot.show()
    

def main():
    data = pd.read_csv(r"C:\Users\Misu\Desktop\Bank-transaction-fraud-detection\bank_transactions_data_2.csv")
    # columns = ['TransactionID', 'AccountID', 'TransactionAmount', 'TransactionDate', 'TransactionType', 'AccountBalance', 'PreviousTransactionDate']
    # process_data = data.drop(['TransactionID'], axis=1)


    # # plot_initial_data(data)    
    
    # train_dataset = data.sample(frac=0.75, random_state=1)
    
    # print(len(train_dataset))
    
    # Define relevant columns for logistic regression
    # Check for missing values
    print("Missing values:\n", data.isnull().sum())

    # Fill or drop missing values (example: fill missing numeric columns with median)
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    # Handle categorical data (example: fill missing with mode)
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Convert date columns to datetime format (if applicable)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Drop columns with too many missing values (optional)
    numeric_data = numeric_data.dropna(axis=1, thresh=len(numeric_data) * 0.5)  # Keep columns with >50% non-NaN values

    # Fill remaining missing values with median (optional)
    numeric_data = numeric_data.fillna(numeric_data.median())
    
    # Preprocessing: Handle datetime columns
    data['TransactionDate'] = pd.to_datetime(data['TransactionDate'],format='%Y-%m-%d %H:%M:%S')
    data['PreviousTransactionDate'] = pd.to_datetime(data['PreviousTransactionDate'],format='%Y-%m-%d %H:%M:%S')
    data['TimeSinceLastTransaction'] = (data['TransactionDate'] - data['PreviousTransactionDate']).dt.total_seconds()

    # Identify numeric and categorical columns
    numeric_cols = ['TransactionAmount', 'TransactionDuration', 'LoginAttempts', 'AccountBalance', 'CustomerAge', 'TimeSinceLastTransaction']
    categorical_cols = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation']

    # Normalize numeric columns
    scaler = StandardScaler()
    numeric_scaled = pd.DataFrame(scaler.fit_transform(data[numeric_cols]), columns=numeric_cols)

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid dummy variable trap
    categorical_encoded = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), 
                                    columns=encoder.get_feature_names_out(categorical_cols))

    # Combine preprocessed data
    processed_data = pd.concat([numeric_scaled, categorical_encoded], axis=1)

    # Descriptive Statistics
    print("Descriptive Statistics:")
    print(data[numeric_cols].describe())

    # Initialize Fraud column
    data['Fraud'] = False
    # warnings.filterwarnings("ignore", category=FutureWarning)

    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=0.02, random_state=42)  # 2% expected anomalies
    iso_forest.fit(numeric_scaled)  # Fit on the scaled numeric data

    # Predict anomalies
    data['IsoForest_Score'] = iso_forest.decision_function(numeric_scaled)
    data['IsoForest_Fraud'] = iso_forest.predict(numeric_scaled) == -1  # Mark anomalies (-1) as fraud

    # Extract fraudulent transactions
    iso_fraud_points = data[data['IsoForest_Fraud']]

    # Summary of Isolation Forest results
    total_iso_fraud_points = data['IsoForest_Fraud'].sum()
    iso_fraud_points_summary = iso_fraud_points[['TransactionAmount', 'TransactionDuration', 'AccountBalance', 'IsoForest_Fraud']]

    # Adding True values to Fraud column
    data['Fraud'] |= data['IsoForest_Fraud']




    
    features = ['TransactionAmount', 'TransactionDuration', 'LoginAttempts', 'AccountBalance','CustomerAge','TimeSinceLastTransaction']
    target = 'Fraud'
    
    # Prepare the dataset
    X = data[features]
    print(X)
    y = data[target]
    print(y)
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
    print(X_test)

    
    # Train Logistic Regression model
    log_reg = LogisticRegression(random_state=42)
    print(y_train.unique())
    print()
    print(y_train.value_counts())
    log_reg.fit(X_train, y_train)

    # Predict fraud on the test set
    y_pred = log_reg.predict(X_test)


    # Evaluate model performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.show()

    # Add predictions to the dataset
    data['LogReg_Fraud'] = log_reg.predict(X_scaled)
    data['Fraud'] |= data['LogReg_Fraud']

    # Visualize fraud vs. non-fraud transactions
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=data['TransactionAmount'],
        y=data['AccountBalance'],
        hue=data['LogReg_Fraud'],
        palette={1: 'red', 0: 'blue'},
        alpha=0.7
    )
    plt.title('Logistic Regression Fraud Detection', fontsize=16)
    plt.xlabel('Transaction Amount', fontsize=14)
    plt.ylabel('Account Balance', fontsize=14)
    plt.legend(title='Fraud', labels=['Non-Fraud', 'Fraud'], fontsize=12)
    plt.grid(True)
    plt.show()

    # Save fraudulent transactions detected by Logistic Regression
    log_reg_fraud_output_path = '/kaggle/working/log_reg_fraud_transactions.csv'
    log_reg_fraud_points = data[data['LogReg_Fraud'] == 1]
    log_reg_fraud_points.to_csv(log_reg_fraud_output_path, index=False)

    # Summary
    total_log_reg_fraud_points = log_reg_fraud_points.shape[0]
    print(f"Total Fraudulent Transactions Detected by Logistic Regression: {total_log_reg_fraud_points}")
    print(f"Fraudulent transactions saved to: {log_reg_fraud_output_path}")
    print(log_reg_fraud_points)

if __name__ == "__main__":
    main()