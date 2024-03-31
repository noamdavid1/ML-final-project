import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the training dataset
train_data = pd.read_csv('train.csv')

# Separate features (X_train) and target variable (y_train) for training
X_train = train_data.drop(columns=['fake'])
y_train = train_data['fake']

# Load the test dataset
test_data = pd.read_csv('test.csv')

# Separate features (X_test) and target variable (y_test) for testing
X_test = test_data.drop(columns=['fake'])
y_test = test_data['fake']

# Define a list of k values and p values to try for KNN
k_values = [1, 3, 5, 7]
p_values = [1, 2, float('inf')]

# Perform Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define a list of C values to try for SVM
C_values = [0.1, 1, 10]

# Define a list of n_estimators values to try for AdaBoost
n_estimators_values = [1, 100, 1000]

# Define a list of regularization parameter values (C) and solvers to try for logistic regression
C_values_lr = [0.1, 1, 10]
solvers = ['lbfgs', 'liblinear', 'saga']

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define models and parameters for Grid Search
models_params = {
    'KNN': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': k_values, 'p': p_values}},
    'SVM': {'model': SVC(kernel='linear'), 'params': {'C': C_values}},
    'AdaBoost': {'model': AdaBoostClassifier(), 'params': {'n_estimators': n_estimators_values, 'algorithm': ['SAMME']}},
    'LogisticRegression': {'model': LogisticRegression(max_iter=10000), 'params': {'C': C_values_lr, 'solver': solvers}},
}

# Lists to store metrics for each model
model_names = []
avg_cv_scores = []
true_errors = []
test_accuracies = []
test_precisions = []
test_recalls = []
test_f1s = []

# Perform Grid Search and Cross-Validation for each model
for name, mp in models_params.items():
    model = mp['model']
    params = mp['params']
    
    print(f'{name} results:')
    
    # Grid Search
    grid_search = GridSearchCV(model, params, cv=kf)
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model from Grid Search
    best_model = grid_search.best_estimator_
    
    # Cross-Validation Score
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=kf)
    avg_cv_score = cv_scores.mean()
    
    # True Error on Test Data
    y_test_pred = best_model.predict(X_test_scaled)
    true_error = mean_squared_error(y_test, y_test_pred)
    
    # Calculate Test Accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Calculate Test Precision
    test_precision = precision_score(y_test, y_test_pred)
    
    # Calculate Test Recall
    test_recall = recall_score(y_test, y_test_pred)
    
    # Calculate Test F1 Score
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Append metrics to lists
    model_names.append(name)
    avg_cv_scores.append(avg_cv_score)
    true_errors.append(true_error)
    test_accuracies.append(test_accuracy)
    test_precisions.append(test_precision)
    test_recalls.append(test_recall)
    test_f1s.append(test_f1)
    
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Cross-Validation Mean Score: {avg_cv_score:.2f}')
    print(f'True Error on Test Data: {true_error:.2f}')
    print(f'Test Accuracy: {test_accuracy:.2f}')
    print(f'Test Precision: {test_precision:.2f}')
    print(f'Test Recall: {test_recall:.2f}')
    print(f'Test F1 Score: {test_f1:.2f}\n')

# Plotting metrics for comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
bars1 = plt.bar(model_names, avg_cv_scores)
plt.title('Cross-Validation Mean Score (Higher is Better)')
plt.xlabel('Model')
plt.ylabel('Mean Score')
best_cv_model_index = avg_cv_scores.index(max(avg_cv_scores))
bars1[best_cv_model_index].set_color('green')
for bar, score in zip(bars1, avg_cv_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{score:.2f}', ha='center', va='bottom')

plt.subplot(2, 2, 2)
bars2 = plt.bar(model_names, true_errors)
plt.title('True Error on Test Data (Lower is Better)')
plt.xlabel('Model')
plt.ylabel('True Error')
best_error_model_index = true_errors.index(min(true_errors))
bars2[best_error_model_index].set_color('green')
for bar, error in zip(bars2, true_errors):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{error:.2f}', ha='center', va='bottom')

plt.subplot(2, 2, 3)
bars3 = plt.bar(model_names, test_accuracies)
plt.title('Test Accuracy (Higher is Better)')
plt.xlabel('Model')
plt.ylabel('Accuracy')
best_acc_model_index = test_accuracies.index(max(test_accuracies))
bars3[best_acc_model_index].set_color('green')
for bar, acc in zip(bars3, test_accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{acc:.2f}', ha='center', va='bottom')

plt.subplot(2, 2, 4)
bars4 = plt.bar(model_names, test_f1s)
plt.title('Test F1 Score (Higher is Better)')
plt.xlabel('Model')
plt.ylabel('F1 Score')
best_f1_model_index = test_f1s.index(max(test_f1s))
bars4[best_f1_model_index].set_color('green')
for bar, f1 in zip(bars4, test_f1s):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{f1:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
