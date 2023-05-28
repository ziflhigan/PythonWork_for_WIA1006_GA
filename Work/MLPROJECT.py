import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import cross_val_score
import pickle

data = pd.read_csv('heart.csv')

# print(data.info)

# Check for missing values
print(data.isnull().sum())
# running result shows no missing value, no need to proceed it

# Counting for the number of how many people have heart diseases
heart_count = data['target'].value_counts()
# 526 has heart disease, 499 do not have heart disease

# Plotting the graphs of age vs. presence of heart disease
# Scatter plot
custom_palette = {0: 'red', 1: 'blue'}

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='age', y='target', hue='target', palette=custom_palette)
plt.xlabel('Age')
plt.ylabel('Presence of Heart Disease')
plt.title('Age vs Presence of Heart Disease')

legend_elements = [Line2D([0], [0], marker='o', color='w', label='No', markerfacecolor='red', markersize=8),
                   Line2D([0], [0], marker='o', color='w', label='Yes', markerfacecolor='blue', markersize=8)]
plt.legend(handles=legend_elements, title='Heart Disease')

plt.show()

# Histogram Plot
custom_palette = {0: 'red', 1: 'blue'}

plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', hue='target', kde=True, palette=custom_palette, element='step', common_norm=False)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution of Patients with and without Heart Disease')

legend_elements = [Line2D([0], [0], color='red', lw=2, label='No'),
                   Line2D([0], [0], color='blue', lw=2, label='Yes')]
plt.legend(handles=legend_elements, title='Heart Disease')

plt.show()

# Split the data into dependent and independent variables
X = data.drop('target', axis=1)
y = data['target']

# For feature selection, let's use Recursive Feature Elimination (RFE) with a Logistic Regression model as the estimator
estimator = LogisticRegression(solver='liblinear')
selector = RFE(estimator, n_features_to_select=12, step=1)
# I have tested several number of features to select manually, and I found that 12 will provide the best score.

selector = selector.fit(X, y)
selected_features = pd.DataFrame({'Feature': list(X.columns),
                                  'Ranking': selector.ranking_})
print(selected_features.sort_values(by='Ranking'))

X_selected = selector.transform(X)

# Split the dataset into training, validation and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y, test_size=0.25, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=44)

# feature scaling using standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.fit_transform(X_val)

# Test different k values and store the cross-validated accuracies
k_values = list(range(200, 500))
cv_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_accuracies.append(cv_scores.mean())

print('\nFor the best KNN model:\n')
# Find the optimal k value and create the best KNN model
optimal_k = k_values[cv_accuracies.index(max(cv_accuracies))]
print("The optimal value of k is : ", optimal_k)
best_knn = KNeighborsClassifier(optimal_k)
best_knn.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = best_knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# store the trained model as pickle file to make a web app that may use this model to predict
with open('trained_model_KNN.pkl', 'wb') as f:
    pickle.dump(best_knn, f)
# store the scaler formula used to be used for standardizing the inputs from the web app
with open('selected_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Calculate the predicted probabilities for the positive class (heart disease presence)
y_pred_proba_knn = best_knn.predict_proba(X_test)[:, 1]

# Calculate the False Positive Rate (FPR) and True Positive Rate (TPR) at various thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_knn)

# Calculate the Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for KNN model')
plt.legend(loc="lower right")

plt.show()

# Try for second model : logistic Regression

# Create a Logistic Regression model instance
logistic_model = LogisticRegression()

# Train the model on the pre-processed training data
logistic_model.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = logistic_model.predict(X_test)

print("\nFor the default logistic Regression model: \n")

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

#

# Let's tune the hyperparameters

# Define the parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'fit_intercept': [True, False],
}

# Create the grid search object
grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Print the best parameters
print("Best parameters: ", best_params)

best_params = {'C': 1, 'fit_intercept': True, 'penalty': 'l2'}

# Create a new Logistic Regression model using the best parameters from the grid search
best_logistic_model = LogisticRegression(C=best_params['C'], fit_intercept=best_params['fit_intercept'], penalty=best_params['penalty'],)

# Train the model on the pre-processed training data
best_logistic_model.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred_logistic = best_logistic_model.predict(X_val)

print("\nFor the best logistic Regression model: \n")

# Calculate the accuracy of the model
accuracy = accuracy_score(y_val, y_pred_logistic)
print("Accuracy:", accuracy)

# Print the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_logistic))

# Print the classification report
print("Classification Report:\n", classification_report(y_val, y_pred_logistic))

# Now let's try to plot the ROC curve
y_pred_proba_logistic = best_logistic_model.predict_proba(X_val)[:, 1]

fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(y_val, y_pred_proba_logistic)

roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

plt.figure(figsize=(10, 6))
plt.plot(fpr_logistic, tpr_logistic, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc_logistic)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression')
plt.legend(loc="lower right")

plt.show()

# Try for third model: SVM

# Define the model
svc = svm.SVC()

# Train the model
svc.fit(X_train, y_train)

# Predict the labels for the test set
y_pred_svm = svc.predict(X_val)

print("For the default SVM model: \n")
# Print the accuracy
print("Accuracy:", accuracy_score(y_val, y_pred_svm), '\n')

# Now we will apply Grid Search for Hyperparameter tuning

# Define the parameter grid
param_grid = {'C': [0.01, 0.1, 5, 10, 50, 100],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)

# Fit the model
grid.fit(X_train, y_train)

# Print best parameters
print("Best parameters: ", grid.best_params_)

# Still, we need to compare the scores the select the parameters after trade-off with
# the possibility of over-fit
best_params_SVM = {'C': 5, 'gamma': 0.1, 'kernel': 'rbf'}

best_SVM = svm.SVC(C=best_params_SVM['C'], gamma=best_params_SVM['gamma'], kernel=best_params_SVM['kernel'],
                   probability=True)

best_SVM.fit(X_train, y_train)

y_pred_SVM = best_SVM.predict(X_test)

print("\nFor the best SVM model: \n")

# Calculate the accuracy of the model
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
print("Accuracy:", accuracy_SVM)

# Print the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_SVM))

# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred_SVM))

# Now let's try to plot the ROC curve
y_pred_proba_SVM = best_SVM.predict_proba(X_test)[:, 1]

fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, y_pred_proba_SVM)

roc_auc_SVM = auc(fpr_SVM, tpr_SVM)

plt.figure(figsize=(10, 6))
plt.plot(fpr_SVM, tpr_SVM, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc_SVM)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVM')
plt.legend(loc="lower right")

plt.show()

# store this SVM model as pickle file to make a web app that may use it to predict
with open('trained_model_SVM.pkl', 'wb') as f:
    pickle.dump(best_SVM, f)
