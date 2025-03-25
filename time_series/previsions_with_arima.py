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

model = auto_arima(time_series, order=(4, 1, 3))
print(model.order)

predictions = model.predict(n_periods=24)
print(predictions)