import pandas as pd
from datetime import datetime
from pmdarima import auto_arima
from pathlib import Path

air_passengers_path = Path(__file__).parent.parent / 'Base_de_dados' / 'AirPassengers.csv'

dataset = pd.read_csv(air_passengers_path)

dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
dataset = pd.read_csv(air_passengers_path, parse_dates=['Month',], index_col='Month', date_format=dateparse)

time_series = dataset['#Passengers']
time_series.index = pd.to_datetime(time_series.index)

print(len(time_series))

training_time_series = time_series[:130]
test_time_series = time_series[130:]

model = auto_arima(training_time_series, suppress_warnings=True)

predictions = pd.DataFrame(model.predict(n_periods=14), index=test_time_series.index)
predictions.columns = ['passengers_predictions']

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.plot(training_time_series, label='Training')
plt.plot(test_time_series, label='Test')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
