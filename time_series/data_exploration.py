import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from pathlib import Path

air_passengers_path = Path(__file__).parent.parent / 'Base_de_dados' / 'AirPassengers.csv'

dataset = pd.read_csv(air_passengers_path)

dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
dataset = pd.read_csv(air_passengers_path, parse_dates=['Month',], index_col='Month', date_format=dateparse)

time_series = dataset['#Passengers']

print(time_series)
print(time_series[1])
print(time_series['1949-02'])
print(time_series['1949-02' : '1950-01'])

print(time_series.index.max())
print(time_series.index.min())

# Plots
import matplotlib.pyplot as plt

time_series.index = pd.to_datetime(time_series.index)
time_series_year = time_series.resample('YE').sum()

plt.plot(time_series_year)
plt.show()

time_series_month = time_series.groupby([lambda x: x.month]).sum()
plt.plot(time_series_month)
plt.show()