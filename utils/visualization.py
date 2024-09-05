import numpy as np
import matplotlib.pyplot as plt


def plot_regression_line(X_test, y_test, y_pred, feature_index=0, feature_name='Feature', target_name='Target'):
    plt.figure(figsize=(10, 6))

    # Extract the specific feature for plotting
    X_feature_test = X_test[:, feature_index]

    # Scatter plot of the feature vs target
    plt.scatter(X_feature_test, y_test, color='blue', label='Actual Data')

    # Plot the regression line (for the chosen feature)
    plt.plot(X_feature_test, y_pred, color='red', label='Regression Line')

    plt.xlabel(f'Normalized {feature_name}')
    plt.ylabel(target_name)
    plt.title(f'Regression Line for {feature_name} vs. {target_name}')
    plt.legend()
    plt.show()


def plot_classification_data(X, y):
    # Scatter plot of the two classes
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0', alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1', alpha=0.6)

    # Add plot title and labels
    plt.title('Generated Classification Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()
