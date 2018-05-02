
# coding: utf-8

# # Determine the distribution of gene expression variability
# 
# In the following notebook, we calculate and visualize the per gene variability in the CoMMpass dataset.
# We use Median Absolute Deviation ([MAD](https://en.wikipedia.org/wiki/Median_absolute_deviation)) to measure gene expression variability.
# 
# We output this measurement to a file and recommend subsetting gene expression values before input to machine learning models.
# Subsetting gene expression matrices to between 5,000 and 10,000 genes captures the majority of variation in the data.
# 
# 
# We select 8,000 genes for downstream analyses, which is what we used in other experiments (see [Way et al. 2018](https://doi.org/10.1016/j.celrep.2018.03.046 "Machine Learning Detects Pan-cancer Ras Pathway Activation in The Cancer Genome Atlas") and [this discussion](https://github.com/cognoma/machine-learning/pull/18#issuecomment-236265506))

# In[1]:


import os
import pandas as pd

from statsmodels.robust.scale import mad
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


num_genes = 8000


# In[4]:


file = os.path.join('data', 'raw', 'CoMMpass_train_set.csv')
train_df = pd.read_csv(file, index_col=0).drop("Location", axis='columns')
print(train_df.shape)


# In[5]:


# Reorder genes based on MAD and retain higher variable genes
train_df = (
    train_df
    .assign(mad_genes = mad(train_df, axis=1))
    .sort_values(by='mad_genes', ascending=False)
    .drop('mad_genes')
)

train_df = train_df[~train_df.index.duplicated(keep='first')]

print(train_df.shape)
train_df.head(2)


# In[6]:


mad_genes = mad(train_df, axis=1)

mad_gene_df = (
    pd.DataFrame(mad_genes, index=train_df.index, columns=['mad_genes'])
    .sort_values(by='mad_genes', ascending=False)
)


# In[7]:


# How many genes have no variance
(mad_gene_df['mad_genes'] == 0).value_counts()


# In[8]:


# Remove genes lacking variance
mad_gene_df = mad_gene_df.query("mad_genes > 0")
print(mad_gene_df.shape)


# It looks like the highest variable gene is a large outlier
# The gene is [B2M](http://useast.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000166710;r=15:44711477-44718877).

# In[9]:


# Distribution of gene expression variability after removing zeros
sns.distplot(mad_gene_df['mad_genes']);


# In[10]:


# Distribution of gene expression variability after removing zeros
sns.distplot(mad_gene_df.query("mad_genes > 100")['mad_genes']);


# In[11]:


total_mad = mad_gene_df['mad_genes'].sum()
mad_gene_df = mad_gene_df.assign(variance_proportion = mad_gene_df['mad_genes'].cumsum() / total_mad)


# In[12]:


# Zoom into elbow of plot
sns.regplot(x='index', y='variance_proportion', ci=None, fit_reg=False,
            data=mad_gene_df.reset_index().reset_index())
plt.xlabel('Number of Genes')
plt.ylabel('Proportion of Variance')
plt.axvline(x=5000, color='r', linestyle='--')
plt.axvline(x=10000, color='r', linestyle='--')
plt.axvline(x=num_genes, color='g', linestyle='--');


# In[13]:


mad_gene_df = mad_gene_df.assign(use_in_classifier = 0)
mad_gene_df['use_in_classifier'].iloc[range(0, num_genes)] = 1
mad_gene_df.head()


# In[14]:


file = os.path.join('data', 'mad_genes.tsv')
mad_gene_df.to_csv(file, sep='\t')


# ## Process gene expression data (Training and Testing)
# 
# 1. Zero-one normalization within dataset
# 2. Also subset to MAD genes derived from training set
# 
# This is done for easier processing in downstream analyses.

# In[15]:


use_genes = mad_gene_df.query('use_in_classifier == 1')


# ### Training Data

# In[16]:


train_df = train_df.loc[use_genes.index, :]
fitted_scaler = MinMaxScaler().fit(train_df)
train_df = pd.DataFrame(fitted_scaler.transform(train_df),
                        columns=train_df.columns,
                        index=train_df.index)


# In[17]:


file = os.path.join('data', 'compass_x_train.tsv.gz')
train_df.to_csv(file, compression='gzip', sep='\t')


# ### Testing Data

# In[18]:


# Load and process test data
file = os.path.join('data', 'raw', 'CoMMpass_test_set.csv')
test_df = pd.read_csv(file, index_col=0).drop("Location", axis='columns')

# Reorder genes based on MAD and retain higher variable genes
test_df = (
    test_df
    .assign(mad_genes = mad(test_df, axis=1))
    .sort_values(by='mad_genes', ascending=False)
    .drop('mad_genes')
)

test_df = test_df[~test_df.index.duplicated(keep='first')]
print(test_df.shape)


# In[19]:


test_df = test_df.loc[use_genes.index, :]
fitted_scaler = MinMaxScaler().fit(test_df)
test_df = pd.DataFrame(fitted_scaler.transform(test_df),
                       columns=test_df.columns,
                       index=test_df.index)


# In[20]:


file = os.path.join('data', 'compass_x_test.tsv.gz')
test_df.to_csv(file, compression='gzip', sep='\t')

