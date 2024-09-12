# Logistic Regression Implementation from Scratch

This project includes an implementation of the Logistic Regression algorithm from scratch, used for binary
classification tasks. The model is evaluated on the Breast Cancer Wisconsin dataset, a commonly used dataset for
classification problems.

### Key Features

- **Custom Implementation:** The logistic regression model is implemented from scratch, including the sigmoid activation
  function and gradient descent optimization.
- **Evaluation Metric:** The model's performance is assessed using accuracy, a common metric for classification tasks.

### Methodology

#### Data Preparation

- The Breast Cancer Wisconsin dataset is used, which contains features related to breast cancer characteristics and
  binary
  labels (malignant or benign).

#### Model Training

- The LogisticRegression class is implemented with methods to fit the model to the training data and predict binary
  labels
  for new data.
- The model is trained using gradient descent with a specified learning rate and number of iterations.

#### Evaluation

- The model's performance is evaluated using the accuracy score, which measures the proportion of correctly classified
  instances in the test set.

### Results

- **Accuracy:** The logistic regression model achieved an accuracy of **93%** on the test set, indicating strong
  performance in
  classifying breast cancer samples.