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
time_series.index = pd.to_datetime(time_series.index)

# Decomposition
decomposition = seasonal_decompose(time_series)
trend = decomposition.trend
seasonal = decomposition.seasonal
random_resid = decomposition.resid

# Plots
import matplotlib.pyplot as plt

plt.plot(trend)
plt.show()

plt.plot(seasonal)
plt.show()

plt.plot(random_resid)
plt.show()

