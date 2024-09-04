import matplotlib.pyplot as plt


def plot_regression_line(X_test, y_test, feature_name='Feature', target_name='Target'):
    plt.figure(figsize=(10, 6))

    # Scatter plot of the feature vs target
    plt.scatter(X_test, y_test, color='blue', label='Actual Data')

    # Plot the regression line
    plt.plot(X_test, X_test, color='red', label='Regression Line')

    plt.xlabel(f'Normalized {feature_name}')
    plt.ylabel(target_name)
    plt.title(f'Regression Line for {feature_name} vs. {target_name}')
    plt.legend()
    plt.show()
