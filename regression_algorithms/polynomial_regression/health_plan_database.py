import pandas as pd
import plotly.express as px

from pathlib import Path


health_plan_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'plano_saude.csv'

health_plan_base = pd.read_csv(health_plan_path)

print(health_plan_base)

X_health_plan = health_plan_base.iloc[:, 0:1].values
y_health_plan = health_plan_base.iloc[:, 1].values

print(X_health_plan.shape, y_health_plan.shape)

from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=4)

X_health_plan_poly = polynomial.fit_transform(X_health_plan)

print(X_health_plan_poly)

from sklearn.linear_model import LinearRegression

polynomial_regression = LinearRegression()
polynomial_regression.fit(X_health_plan_poly, y_health_plan)

previsions = polynomial_regression.predict(X_health_plan_poly)

graphic = px.scatter(x=X_health_plan.ravel(), y=y_health_plan)
graphic.add_scatter(x=X_health_plan.ravel(), y=previsions, name='Regression')
graphic.show()