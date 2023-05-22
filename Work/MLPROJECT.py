import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
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

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=40)

# feature scaling using standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Test different k values and store the cross-validated accuracies
k_values = list(range(300, 420))
cv_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_accuracies.append(cv_scores.mean())

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

# store the trained model as pickle file to make a web app that uses this model to predict
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(best_knn, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Calculate the predicted probabilities for the positive class (heart disease presence)
y_pred_proba = best_knn.predict_proba(X_test)[:, 1]

# Calculate the False Positive Rate (FPR) and True Positive Rate (TPR) at various thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

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
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.show()
