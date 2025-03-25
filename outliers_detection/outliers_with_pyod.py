from pathlib import Path
import pandas as pd
from pyod.models.knn import KNN 
import numpy as np

credit_data_path = Path(__file__).parent.parent / 'Base_de_dados' / 'credit_data.csv'

credit_data_base = pd.read_csv(credit_data_path)
    
credit_data_base.dropna(inplace=True)

X_credit = credit_data_base.iloc[:, 1:4].values

outlier_detector = KNN()
outlier_detector.fit(X_credit)

previsions = outlier_detector.labels_

outlier_ids = []
for i in range(len(previsions)):
    if previsions[i] == 1:
        outlier_ids.append(i)
        
outlier_list = credit_data_base.iloc[outlier_ids, :]


if __name__ == '__main__':
    print(np.unique(previsions, return_counts=True))
    print('Previsions Confidence:', outlier_detector.decision_scores_)
    print(outlier_list)