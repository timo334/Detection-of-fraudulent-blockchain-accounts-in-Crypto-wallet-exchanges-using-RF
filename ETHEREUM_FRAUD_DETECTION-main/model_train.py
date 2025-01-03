import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier;
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
import pickle


df = pd.read_csv('ETHEREUM_FRAUD_DETECTION-main\\transaction_dataset.csv\\transaction_dataset.csv', index_col=0)
# print(df.shape)
# df.head()
df = df.iloc[:,2:]

print(df['FLAG'].value_counts())

# SHOW INITIAL GRAPH OF CSV FILE

# pie, ax = plt.subplots(figsize=[15,10])
# labels = ['Non-fraud', 'Fraud']
# colors = ['#f9ae35', '#f64e38']
# plt.pie(x = df['FLAG'].value_counts(), autopct='%.2f%%', explode=[0.02]*2, labels=labels, pctdistance=0.5, textprops={'fontsize': 14}, colors = colors)
# plt.title('Target distribution')
# plt.show()

categories = df.select_dtypes('O').columns.astype('category')
df.drop(df[categories], axis=1, inplace=True)
df.fillna(df.median(), inplace=True)

no_var = df.var() == 0
df.drop(df.var()[no_var].index, axis = 1, inplace = True)

drop = ['total transactions (including tnx to create contract', 'total ether sent contracts', 'max val sent to contract', ' ERC20 avg val rec',
        ' ERC20 avg val rec',' ERC20 max val rec', ' ERC20 min val rec', ' ERC20 uniq rec contract addr', 'max val sent', ' ERC20 avg val sent',
        ' ERC20 min val sent', ' ERC20 max val sent', ' Total ERC20 tnxs', 'avg value sent to contract', 'Unique Sent To Addresses',
        'Unique Received From Addresses', 'total ether received', ' ERC20 uniq sent token name', 'min value received', 'min val sent', ' ERC20 uniq rec addr' ]
df.drop(drop, axis=1, inplace=True)
drops = ['min value sent to contract', ' ERC20 uniq sent addr.1']
df.drop(drops, axis=1, inplace=True)
print(df.shape)


y = df.iloc[:, 0]
X = df.iloc[:, 1:]
# print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

norm = PowerTransformer()
norm_train_f = norm.fit_transform(X_train)
norm_df = pd.DataFrame(norm_train_f, columns=X_train.columns)

oversample = SMOTE()
print(f'Shape of the training before SMOTE: {norm_train_f.shape, y_train.shape}')
x_tr_resample, y_tr_resample = oversample.fit_resample(norm_train_f, y_train)
print(f'Shape of the training after SMOTE: {x_tr_resample.shape, y_tr_resample.shape}')


non_fraud = 0
fraud = 0

for i in y_train:
    if i == 0:
        non_fraud +=1
    else:
        fraud +=1

# Target distribution after SMOTE
no = 0
yes = 1

for j in y_tr_resample:
    if j == 0:
        no +=1
    else:
        yes +=1


print(f'BEFORE OVERSAMPLING \n \tNon-frauds: {non_fraud} \n \tFrauds: {fraud}')
print(f'AFTER OVERSAMPLING \n \tNon-frauds: {no} \n \tFrauds: {yes}')

#LOGISTIC REGRESSION

LR = LogisticRegression(random_state=42)
LR.fit(x_tr_resample, y_tr_resample)
norm_test_f = norm.transform(X_test)
preds = LR.predict(norm_test_f)


#RANDOM FOREST CLASSIFIER

RF = RandomForestClassifier(random_state=42)
RF.fit(x_tr_resample, y_tr_resample)
preds_RF = RF.predict(norm_test_f)

print(classification_report(y_test, preds_RF))
print(confusion_matrix(y_test, preds_RF))
# plot_confusion_matrix(RF, norm_test_f, y_test)



#PREDICTION

original_df = pd.read_csv('ETHEREUM_FRAUD_DETECTION-main\\transaction_dataset.csv\\transaction_dataset.csv', index_col=0)
predictions = RF.predict(norm.transform(X_test))
fraud_indices = np.where(predictions == 1)[0]
non_fraud_indices = np.where(predictions == 0)[0]

fraudulent_addresses = original_df.iloc[X_test.index[fraud_indices]]['Address'] 
fraudulent_dataset = pd.DataFrame({'Address': fraudulent_addresses})

non_fraudulent_addresses = original_df.iloc[X_test.index[non_fraud_indices]]['Address'] 
non_fraudulent_dataset = pd.DataFrame({'Address': non_fraudulent_addresses})

print(fraudulent_dataset)
print(non_fraudulent_dataset)



print(df.columns)

input_addr=input("Input the reciever\'s Ethereum Account Address : ")

if len(input_addr)!=42:
    print("Invalid Address !!")
elif input_addr in fraudulent_dataset['Address'].values:
    print("The model predicts that the transaction associated with the provided address is a fraud account.")
else:
    print("The model predicts that the transaction associated with the provided address is a non-fraud account.")


