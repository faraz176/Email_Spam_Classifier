
# Email Classification using Logistic Regression. 



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
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Prints Confusion Matrix (By default it prints the confusin matrix of the best threshold, however, in order to generalize our model(See how it works with Cross-Validation we want to compute and ROC curve and calculate the area to get an idea of how 'good' our model is))
print(confusion_matrix(y_test, y_pred))

#Prints ROC Curve (Of a cross validation of 1, and getting the area)
y_pred_proba = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

## Plot ROC curve
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='Logistic Regression')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()

#Print the area underneath ROC Curve
print(roc_auc_score(y_test, y_pred_proba))

#Printing different ROC's with cross validation 
# cv_scores = cross_val_score(logreg, X,y, cv=5, scoring='roc_auc')
# print(cv_scores)

# Hyper parameters Tuning for Logistic Regression
# There is 3 main hyper parameters you can adjust for Logistic Regression
#Solver, penalty, C

#Our hyperparameters
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

#Turning into a dictionary 
grid = dict(solver=solvers, penalty=penalty, C=c_values)

grid_search = GridSearchCV(estimator=logreg, param_grid=grid, n_jobs=-1, cv=5, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#Saving the model
import pickle
 
# Save the trained model as a pickle string.
Pkl_Filename = "email.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(grid_result, file)
 




#Spam Test (Processing data and getting input read for prediction)
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


work = df.append(new_df).fillna(0)



big_columns = []
for col in work.columns:
    if col != 'Email No.':
        big_columns.append(col)

work = work[big_columns]

# Test Ready
X_test = work.loc[:, work.columns != 'Prediction']

#Loading the model and predicting on the new data 
import pickle
saved_model = "email.pkl"
knn_from_pickle = object = pd.read_pickle(r'email.pkl')
y_pred_1 = knn_from_pickle.predict(X_test)
print(y_pred_1)