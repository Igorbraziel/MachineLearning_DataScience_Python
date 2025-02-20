from census_database import base_census

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

if __name__ == '__main__':
    print(np.unique(base_census['income'], return_counts=True))
    
    sns.countplot(x = base_census['income'])
    plt.show()
    
    plt.hist(x = base_census['age'])
    plt.show()
    
    plt.hist(x = base_census['education-num'])
    plt.show()
    
    plt.hist(x = base_census['hour-per-week'])
    plt.show()
    
    treemap_graphic = px.treemap(base_census, path=['workclass', 'age'])
    treemap_graphic.show()
    
    treemap_graphic = px.treemap(base_census, path=['occupation', 'relationship'])
    treemap_graphic.show()
    
    parallel_graphic = px.parallel_categories(base_census, dimensions=['workclass', 'occupation', 'income'])
    parallel_graphic.show()