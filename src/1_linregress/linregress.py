import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

N = 200
slope = 5
intercept = 0
noise_strength = 0.5

x = np.random.random(N)
y = slope * x
noise = np.random.randn(N) * noise_strength

x_fit = np.linspace(x.min(), x.max())
y_fit = slope * x_fit + intercept

A = np.column_stack((x, np.ones_like(x)))
x_hat = np.linalg.inv(A.T @ A) @ A.T @ (y + noise)
slope_hat, intercept_hat = x_hat
y_hat_fit = slope_hat * x_fit + intercept_hat
print(x_hat)

model = LinearRegression()
model.fit(A, y + noise)
print([model.coef_[0], model.intercept_])

plt.plot(x, y + noise, "o")
plt.plot(x_fit, y_fit, "--")
plt.plot(x_fit, y_hat_fit, "-")
plt.show()
