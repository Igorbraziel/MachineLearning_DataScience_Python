import pandas as pd

from pathlib import Path
import os

database_file_name = 'census.csv'

root_path = Path(__file__).parent.parent

database_path = Path(
    os.path.join(
        root_path, 'Base_de_dados', database_file_name
    )
)

base_census = pd.read_csv(database_path)

if __name__ == '__main__':
    print(base_census)
    print(base_census.describe())
    print(base_census.isnull().sum())
    