import pandas as pd

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

#Final X_Test
X_test = work.loc[:, work.columns != 'Prediction']

#Loading the model and predicting
import pickle
saved_model = "email.pkl"
knn_from_pickle = object = pd.read_pickle(r'email.pkl')
y_pred_1 = knn_from_pickle.predict(X_test)
print(y_pred_1)