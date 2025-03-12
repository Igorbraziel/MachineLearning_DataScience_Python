import pandas as pd
import plotly.express as px
import numpy as np

from pathlib import Path

health_plan_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'plano_saude.csv'

health_plan_base = pd.read_csv(health_plan_path)

X_health_plan = health_plan_base.iloc[:, 0].values
y_health_plan = health_plan_base.iloc[:, 1].values 

X_health_plan = X_health_plan.reshape(-1, 1)

print(X_health_plan.shape, y_health_plan.shape)

# Standardisation
from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_health_plan = scaler_x.fit_transform(X_health_plan)
y_health_plan = scaler_y.fit_transform(y_health_plan.reshape(-1, 1)).ravel()

from sklearn.neural_network import MLPRegressor

neural_network_regressor = MLPRegressor(max_iter=10000, random_state=0, n_iter_no_change=1000, tol=0.0000001)
neural_network_regressor.fit(X_health_plan, y_health_plan)

print('Score:', neural_network_regressor.score(X_health_plan, y_health_plan))

previsions = neural_network_regressor.predict(X_health_plan)

previsions = scaler_y.inverse_transform(previsions.reshape(-1, 1)).ravel()
y_health_plan = scaler_y.inverse_transform(y_health_plan.reshape(-1, 1)).ravel()
X_health_plan = scaler_x.inverse_transform(X_health_plan)

graphic = px.scatter(x=X_health_plan.ravel(), y=y_health_plan)
graphic.add_scatter(x=X_health_plan.ravel(), y=previsions, name='MLPRegressor')
graphic.show()