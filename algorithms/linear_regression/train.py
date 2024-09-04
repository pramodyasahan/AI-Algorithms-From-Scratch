from sklearn import datasets
from utils.validation_metrics import mean_squared_error, r_squared
from utils.visualization import plot_regression_line
from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split

X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression(learning_rate=0.01)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plot_regression_line(X_test, y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r_squared(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R2: {r2}')
