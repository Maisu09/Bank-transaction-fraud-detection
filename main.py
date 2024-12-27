import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def determine_fraudulent_state():
    global data
    print(data)
    
    # data['FlagNumberOfTransactions'] = False if data['LoginAttempts'] < 3 else True
    data['FlagNumberOfTransactions'] = data['LoginAttempts'].apply(lambda x: False if x < 3 else True)

    for index, transaction in data.iterrows():
        if transaction['TransactionType'] == 'Debit':
            if transaction['TransactionAmount'] > transaction['AccountBalance']:
                data.at[index, 'TransactionTypeFlag'] = True
            else:
                data.at[index, 'TransactionTypeFlag'] = False
        elif transaction['TransactionType'] == 'Credit':
            if transaction['TransactionAmount'] > 5000:
                data.at[index, 'TransactionTypeFlag'] = True
            else:
                data.at[index, 'TransactionTypeFlag'] = False
                
    
    data['Fraud'] = data['FlagNumberOfTransactions'] | data['TransactionTypeFlag']
    x = data['Fraud'].value_counts()
    print(x)    
    
    # print(data['TimeSinceLastTransaction'].max())
    # print(data['TimeSinceLastTransaction'].min())
    


def main():
    global data
    # Load the dataset
    data = pd.read_csv(r"bank_transactions_data_2.csv")

    # Fill missing values
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    # Categorical data
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Datetime columns
    data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
    data['PreviousTransactionDate'] = pd.to_datetime(data['PreviousTransactionDate'])
    data['TimeSinceLastTransaction'] = (data['TransactionDate'] - data['PreviousTransactionDate']).dt.total_seconds()

    features = ['TransactionAmount', 'TransactionDuration', 'LoginAttempts', 'AccountBalance', 'TimeSinceLastTransaction']
    target = 'Fraud'
    
    data['Fraud'] = False
    
    determine_fraudulent_state()
    
    # # Prepare the dataset
    X = data[features]
    y = data[target]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

    # Train Logistic Regression model
    reg = LogisticRegression(random_state=42, solver="saga")
    # print(y_train.unique())
    # print()
    # print(y_train.value_counts())
    reg.fit(X_train, y_train)

    # # Predict fraud on the test set
    y_pred = reg.predict(X_test)

    # Evaluate model performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.show()

    # Add predictions to the dataset
    data['RegFraud'] = reg.predict(X_scaled)
    data['Fraud'] |= data['RegFraud']

    # Visualize fraud vs. non-fraud transactions
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=data['TransactionAmount'],
        y=data['AccountBalance'],
        hue=data['RegFraud'],
        palette={1: 'red', 0: 'blue'},
        alpha=0.7
    )
    plt.title('Fraud Detection', fontsize=16)
    plt.xlabel('Transaction Amount', fontsize=14)
    plt.ylabel('Account Balance', fontsize=14)
    plt.legend(title='Fraud', labels=['Non-Fraud', 'Fraud'], fontsize=12)
    plt.grid(True)
    plt.show()

    # Save fraudulent transactions detected
    fraud_output_path = 'fraud_transactions.csv'
    fraud_points = data[data['LogReg_Fraud'] == 1]
    fraud_points.to_csv(fraud_output_path, index=False)

    # Summary
    total_log_reg_fraud_points = fraud_points.shape[0]
    print(f"Total Fraudulent Transactions Detected by Logistic Regression: {total_log_reg_fraud_points}")
    print(f"Total initial fraudulent transactions: {data['Fraud'].sum()}")
    print(fraud_points)

if __name__ == "__main__":
    main()