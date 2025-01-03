input_addr=input("Input the reciever\'s Ethereum Account Address : ")


if input_addr in fraudulent_dataset:
    print("The model predicts that the transaction associated with the provided address is a fraud account.")
else:
    print("The model predicts that the transaction associated with the provided address is a non-fraud account.")