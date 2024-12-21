import matplotlib
import matplotlib.pyplot
import pandas
import numpy as np

def main():
    data = pandas.read_csv("bank_transactions_data_2.csv")
    print(data)
    
    fig, axes = matplotlib.pyplot.subplots(2, 2, figsize=(10, 8))
    
    # ID vs transactioned ammount
    axes[0, 0].plot(data["AccountID"], data["TransactionAmount"], 'x', label='ID and TransactionAmount')
    axes[0, 0].set_title('Transaction per person')
    axes[0, 0].set_xlabel('ID')
    axes[0, 0].set_ylabel('TransactionAmount')
    
    # ID vs Balance
    axes[0, 1].plot(data["AccountID"], data["AccountBalance"], 'x', label='ID and AccountBalance')
    axes[0, 1].set_title('Balance per person')
    axes[0, 1].set_xlabel('ID')
    axes[0, 1].set_ylabel('AccountBalance')
    
    # ID vs DateOfTransaction
    axes[1, 0].plot(data["AccountID"], data["TransactionDate"], 'x', label='ID and TransactionDate')
    axes[1, 0].set_title('Transaction date per person')
    axes[1, 0].set_xlabel('ID')
    axes[1, 0].set_ylabel('TransactionDate')
    
    matplotlib.pyplot.show()
    
    matplotlib.pyplot.plot(data["AccountBalance"], data["TransactionAmount"], 'x')
    matplotlib.pyplot.show()


if __name__ == "__main__":
    main()