import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('student_scores.csv')
x = df[['Hours']].values
y = df[['Scores']].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

theta_0 = 0
theta_1 = 0
alpha = 0.001
num_iterations = 1000
sse_values = []
mse_values = []

for i in range(num_iterations):
    y_pred = theta_0 + theta_1 * X_train
    error = y_train - y_pred

    d_theta_0 = (-2 / len(X_train)) * np.sum(error)
    d_theta_1 = (-2 / len(X_train)) * np.sum(error * X_train)

    theta_0 -= alpha * d_theta_0
    theta_1 -= alpha * d_theta_1

    sse = np.sum(error ** 2)
    mse = np.mean(error ** 2)

    sse_values.append(sse)
    mse_values.append(mse)

    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}: SSE = {sse:.4f}, MSE = {mse:.4f}")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(num_iterations), sse_values, label="SSE")
plt.plot(range(num_iterations), mse_values, label="MSE", linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('SSE and MSE over Iterations')
plt.legend()

plt.subplot(1,2,2)
plt.scatter(X_train, y_train, color='blue', label='Data points')
plt.plot(X_train, theta_0 + theta_1 * X_train, color='red', label='Regression line')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Linear Regression Fit')
plt.legend()
plt.tight_layout()
plt.show()


y_test_pred = theta_0 + theta_1 * X_test
test_error = y_test - y_test_pred

sse_test = np.sum(test_error ** 2)
mse_test = np.mean(test_error ** 2)

print("\n=== Test Set Evaluation ===")
print(f"SSE (Test): {sse_test:.4f}")
print(f"MSE (Test): {mse_test:.4f}")