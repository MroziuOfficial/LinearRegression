import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Dane/dane16.txt", sep=" ", header=None)


X = df[[0]]
y = df[1]

# LinearRegression with MSE
lg = LinearRegression()
lg.fit(X, y)
y_pred = lg.predict(X)
mse = mean_squared_error(y, y_pred)
print("Współczynniki modelu:")
print("Wyraz wolny (a):", lg.intercept_)
print("Współczynnik przy zmiennej (b):", lg.coef_[0])
print("Mean Squared Error (MSE):", mse)

# Opcjonalnie: wizualizacja wyników
plt.scatter(X, y, color='blue', label='Dane')
plt.plot(X, y_pred, color='red', label='Regression')
plt.xlabel("X")
plt.ylabel("y")
plt.title("LinearRegression")
plt.legend()
plt.show()

