from credit_database import base_credit
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print(np.unique(base_credit['default']))
    
    sns.countplot(x = base_credit['default'])
    plt.show()
    
    plt.hist(x = base_credit['age'])
    plt.show()
    
    plt.hist(x = base_credit['income'])
    plt.show()
    
    plt.hist(x = base_credit['loan'])
    plt.show()
    
    graphic = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
    graphic.show()
    
    
    