
#KNN vs Logistic Regression

#The premise of this is to identify which of these two classification models identifies spam emails


#Logistic regression

#Importing Necessary modules
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np 


#Reading the dataframe in 
df = pd.read_csv('emails.csv')

#Preprocessing data (Checking what data is not numeric) and dropping the columns
# for i in range(df.shape[1]-1):
#     column = df.columns[i]
#     if df.dtypes[column] != np.int64 and np.float64:
#         print(column)


big_columns = []
for col in df.columns:
    if col != 'Email No.':
        big_columns.append(col)

df = df[big_columns]

#Instantiating our classifier
logreg = LogisticRegression()


#Subsetting our X's and Y's
X = df.loc[:, df.columns != 'Prediction']
y = df['Prediction']


#Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=41)

#Fitting the model with the training data
logreg.fit(X_train, y_train)

#Predicting on our test set
y_pred = logreg.predict(X_test)

#Computing test metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

#Spam Test
string = 'Only valid at locations in Texas that are owned and operated by Allied Lube Group.SEE WEBSITE FOR FULL LIST OF LOCATIONS HERE:'

list1 = string.split(" ")
print(len(list1))

lowered_list = []
for string in list1:
    lowered_list.append(string.lower())

new_dict = {}
for i in lowered_list:
    if i not in new_dict.keys():
        new_dict[i] = 0
    else:
        new_dict[i] += 1

print(new_dict)

new_df = pd.DataFrame(new_dict, index=[0])

df = pd.read_csv('emails.csv', nrows=1)


for i in new_df.columns:
    if i not in df.columns:
        del(new_df[i])

#Appending dataframes together
work = df.append(new_df).fillna(0)


#Filtering for Email No. & Prediction PreProcessing
big_columns = []
for col in work.columns:
    if col != 'Email No.':
        big_columns.append(col)

work = work[big_columns]

X_test = work.loc[:, work.columns != 'Prediction']
print(X_test.shape[1])
y_pred_1 = logreg.predict(X_test)
print(y_pred_1)
