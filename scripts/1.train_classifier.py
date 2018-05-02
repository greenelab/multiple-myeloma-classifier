
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

from dask_searchcv import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import SGDClassifier


# In[29]:


file = os.path.join('data', 'mad_genes.tsv')
genes_df = pd.read_table(file).query('use_in_classifier == 1')
print(genes_df.shape)
genes_df.tail(2)


# In[32]:


file = os.path.join('data', 'CoMMpass_train_set.csv')
train_df = (
    pd.read_csv(file)
    .drop("Location", axis='columns')
)
print(train_df.shape)
train_df.head(2)


# In[31]:


len(genes_df.GENE_ID)


# In[34]:


a = train_df.query("GENE_ID in @genes_df.GENE_ID")


# In[36]:


a.GENE_ID.duplicated().value_counts()

