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
