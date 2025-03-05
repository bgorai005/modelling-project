import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from scipy.interpolate import CubicSpline

# Load and preprocess data
data = pd.read_csv(r"C:\Users\abrcb\Downloads\bitcoin_prices.csv", usecols=[0, 2])
f_points = data.iloc[:, 1].astype(float).tolist()  # Use the price data
x_points = list(range(1, len(f_points) + 1))       # All x points
sample_indices = x_points[::4]                     # Every 4th point for ARIMA
f_points_sampled = f_points[::4]                   # Price data every 4th point

# Fit ARIMA(1,1,0) model on the sampled data
model = ARIMA(f_points_sampled, order=(1, 1, 0))
model_fit = model.fit()

#print(len(f_points))

# Predict over the range of all points to match full length of x_points
arima_predictions_full = model_fit.predict(start=1, end=len(f_points))

# Use Cubic Spline to interpolate the ARIMA predictions over all points
spline = CubicSpline(sample_indices, arima_predictions_full[:len(sample_indices)])
spline_predictions = spline(x_points)

# Calculate RMSE between original data and interpolated predictions
def RMSE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))

rmse_spline = RMSE(f_points, spline_predictions)
#print(f'ARIMA + Cubic Spline RMSE: {rmse_spline:.2f}')

# Plotting with adjustments for visibility
plt.figure(figsize=(10, 6))

# Larger red markers for sampled points
plt.scatter(sample_indices, f_points_sampled, color='red', marker='o', label='Sampled Points for ARIMA', s=60)

# Green markers for the original data
plt.scatter(x_points, f_points, color='green', marker='o', label='Original Data', s=20)

# Blue line for ARIMA + Cubic Spline interpolation
plt.plot(x_points, spline_predictions, 'b-', label='ARIMA + Cubic Spline', linewidth=1.5)

plt.title('ARIMA(1,1,0) with Cubic Spline Interpolation')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.grid(True)
plt.show()

