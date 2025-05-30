import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # image visualizations
from IPython.display import Image,display
import seaborn as sns
#from wordcloud import WordCloud, STOPWORDS
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

articles = pd.read_csv(r'C:\Users\NIMISHA\PycharmProjects\MinorProject1\data\articles.csv')
customers=pd.read_csv(r'C:\Users\NIMISHA\PycharmProjects\MinorProject1\data\customers.csv')
transactions=pd.read_csv(r"C:\Users\NIMISHA\PycharmProjects\MinorProject1\data\transactions_train.csv")

# pd.set_option('display.max_columns',None) #to display all columns without ...
#print(transactions.head())
#print(customers.head())
#print(articles.head())

#finding missing data
def missing_data(data):
    total=data.isnull().sum().sort_values(ascending=False)
    percent=(data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)
    return pd.concat([total,percent],axis=1,keys=['Total','Percent'])

def unique_values(data):
    total=data.count() # counts total of all cols in data
    tt=pd.DataFrame(total)
    tt.columns=['Total']
    uniques=[]
    for col in data.columns:
        unique=data[col].nunique()
        uniques.append(unique)

    tt['Uniques']=uniques
    return tt

print(articles.info())
print(missing_data(articles))
print(missing_data(customers))
print(missing_data(transactions))
print(unique_values(articles))
print(unique_values(customers))
print(unique_values(transactions))

#analysing articles data summary stats
#temp=articles.groupby(["product_group_name"])["product_type_name"].nunique()
#df=pd.DataFrame({})
