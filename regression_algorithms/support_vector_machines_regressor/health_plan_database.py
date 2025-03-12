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

# Standardisation (Padronização)
from sklearn.preprocessing import StandardScaler

standard_scaler_y = StandardScaler()
standard_scaler_x = StandardScaler()

X_health_plan = standard_scaler_x.fit_transform(X_health_plan)
y_health_plan = standard_scaler_y.fit_transform(y_health_plan.reshape(-1, 1)).ravel()

print(X_health_plan.shape, y_health_plan.shape)

from sklearn.svm import SVR

support_vector_machines_regressor = SVR(kernel='rbf')
support_vector_machines_regressor.fit(X_health_plan, y_health_plan)

print('Score:', support_vector_machines_regressor.score(X_health_plan, y_health_plan))

previsions = support_vector_machines_regressor.predict(X_health_plan)

graphic = px.scatter(x=X_health_plan.ravel(), y=y_health_plan)
graphic.add_scatter(x=X_health_plan.ravel(), y=previsions, name='SVR')
graphic.show()

