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

from sklearn.ensemble import RandomForestRegressor

random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(X_health_plan, y_health_plan)

print('Score:', random_forest_regressor.score(X_health_plan, y_health_plan))

X_decision_tree_test = np.arange(min(X_health_plan), max(X_health_plan), step=0.1)

previsions = random_forest_regressor.predict(X_decision_tree_test.reshape(-1, 1))

graphic = px.scatter(x=X_health_plan.ravel(), y=y_health_plan)
graphic.add_scatter(x=X_decision_tree_test.ravel(), y=previsions, name='Random Forest Regressor')
graphic.show()