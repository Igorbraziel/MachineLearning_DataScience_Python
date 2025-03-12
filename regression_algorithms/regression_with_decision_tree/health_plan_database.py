import pandas as pd
import plotly.express as px

from pathlib import Path

health_plan_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'plano_saude.csv'

health_plan_base = pd.read_csv(health_plan_path)

X_health_plan = health_plan_base.iloc[:, 0].values
y_health_plan = health_plan_base.iloc[:, 1].values 

X_health_plan = X_health_plan.reshape(-1, 1)

print(X_health_plan.shape, y_health_plan.shape)

from sklearn.tree import DecisionTreeRegressor

decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(X_health_plan, y_health_plan)

previsions = decision_tree_regressor.predict(X_health_plan)

print(decision_tree_regressor.score(X_health_plan, y_health_plan))

graphic = px.scatter(x=X_health_plan.ravel(), y=y_health_plan)
graphic.add_scatter(x=X_health_plan.ravel(), y=previsions, name='Decision Tree Regressor')
graphic.show()

import numpy as np

X_tree_test = np.arange(min(X_health_plan), max(X_health_plan), step=0.1)

print(X_tree_test)

graphic1 = px.scatter(x=X_health_plan.ravel(), y=y_health_plan)
graphic1.add_scatter(x=X_tree_test.ravel(), y=decision_tree_regressor.predict(X_tree_test.reshape(-1, 1)), name='Regressor')
graphic1.show()