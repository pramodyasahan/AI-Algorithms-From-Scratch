from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from svm import SupportVectorMachine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Load the Breast Cancer dataset for SVM
breast_cancer = load_breast_cancer()
X_bc, y_bc = breast_cancer.data, breast_cancer.target
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)

# Train SVM on the Breast Cancer dataset
svm = SupportVectorMachine(learning_rate=0.1, lambda_param=0.1, n_iters=1000)
svm.fit(X_train_bc, y_train_bc)
bc_predictions = svm.predict(X_test_bc)

# Calculate performance metrics
svm_accuracy = accuracy_score(y_test_bc, bc_predictions)
svm_precision = precision_score(y_test_bc, bc_predictions, average='weighted')
svm_recall = recall_score(y_test_bc, bc_predictions, average='weighted')
svm_f1 = f1_score(y_test_bc, bc_predictions, average='weighted')

# Print the metrics
print(f"Accuracy: {svm_accuracy:.2f}")
print(f"Precision: {svm_precision:.2f}")
print(f"Recall: {svm_recall:.2f}")
print(f"F1-score: {svm_f1:.2f}")
