import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

lr = LinearRegression()

mse_list = []

# Generate & train data
for _ in range(10):
    rng = np.random.RandomState(_)
    x = 10 * rng.rand(1000).reshape(-1, 1)
    y = 2 * x - 5 + rng.randn(1000).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=50
    )

    lr.fit(X_train, y_train)
    y_preds = lr.predict(X_test)

    test_mse = mean_squared_error(y_test, y_preds)
    mse_list.append(test_mse)

# Proper average
average_mse = np.mean(mse_list)

print("Average MSE:", average_mse)

# Save model
joblib.dump(lr, "model1.pkl")

# ✅ ALWAYS create metrics file
with open("metrics.txt", "w") as f:
    f.write(f"Mean Squared Error = {average_mse}")

# Plot graph
plt.scatter(X_train, y_train, color='blue')
plt.scatter(X_test, y_test, color='red')
plt.scatter(X_test, y_preds, color='green')

plt.savefig("model_results.png")