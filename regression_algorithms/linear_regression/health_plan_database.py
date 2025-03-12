import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path

health_plan_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'plano_saude.csv'

health_plan_base = pd.read_csv(health_plan_path)

X_health_plan = health_plan_base.iloc[:, 0].values
y_health_plan = health_plan_base.iloc[:, 1].values 

print(X_health_plan.shape, y_health_plan.shape)

correlation_coefficient = np.corrcoef(X_health_plan, y_health_plan)
print(correlation_coefficient)

# Reshape for use sklearn
X_health_plan = X_health_plan.reshape(-1, 1)
print(X_health_plan.shape)

# Linear Regression
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X_health_plan, y_health_plan)

b0 = linear_regression.intercept_
b1 = linear_regression.coef_

print(b0, b1)

previsions = linear_regression.predict(X_health_plan)

print(previsions)

print(X_health_plan.ravel())

graphic = px.scatter(x=X_health_plan.ravel(), y=y_health_plan)
graphic.add_scatter(x=X_health_plan.ravel(), y=previsions, name='Linear Regression')
graphic.show()

print(linear_regression.score(X_health_plan, y_health_plan))

from yellowbrick.regressor import ResidualsPlot

viewer = ResidualsPlot(linear_regression)
viewer.fit(X_health_plan, y_health_plan)
viewer.show()




