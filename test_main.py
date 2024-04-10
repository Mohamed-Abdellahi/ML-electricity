# first graphics
plt.figure(figsize=(14, 7))
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.plot(y_test.values, label='Actual', alpha=0.7)  # Ensure y_test is a series, so we access .values
plt.title('Actual vs. Predicted Values Over Time')
plt.xlabel('Time/Index')
plt.ylabel('Value')
plt.legend()
plt.show()

---
def plot_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.reset_index(drop=True), label='Actual', alpha=0.7)
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Electricity Price')
    plt.legend()
    plt.show()
#-----------------------
#--------- @                @----------
from bayes_opt import BayesianOptimization

# Define the function to optimize
def svm_eval(C, gamma, epsilon):
    model = SVR(C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return -1.0 * mse  # We negate MSE because BayesianOptimization seeks to maximize the function

# Define the parameter space
pbounds = {
    'C': (0.1, 1000),
    'gamma': (0.001, 1),
    'epsilon': (0.01, 1)
}

optimizer = BayesianOptimization(
    f=svm_eval,
    pbounds=pbounds,
    random_state=42,
)

# Perform optimization
optimizer.maximize(init_points=10, n_iter=50)