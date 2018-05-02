
# coding: utf-8

# # Process CoMMpass Data
# 
# ## Determine the distribution of gene expression variability
# 
# In the following notebook, we calculate and visualize the per gene variability in the CoMMpass dataset.
# We use Median Absolute Deviation ([MAD](https://en.wikipedia.org/wiki/Median_absolute_deviation)) to measure gene expression variability.
# 
# We output this measurement to a file and recommend subsetting gene expression values before input to machine learning models.
# Subsetting gene expression matrices to between 5,000 and 10,000 genes captures the majority of variation in the data.
# 
# We select 8,000 genes for downstream analyses, which is what we used in other experiments (see [Way et al. 2018](https://doi.org/10.1016/j.celrep.2018.03.046 "Machine Learning Detects Pan-cancer Ras Pathway Activation in The Cancer Genome Atlas") and [this discussion](https://github.com/cognoma/machine-learning/pull/18#issuecomment-236265506))
# 
# ## Subset and Process X and Y matrices
# 
# Input the sklearn classifiers requires the data to be a bit different format than what was provided.
# The script details and performs the processing steps for both.

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
    .drop('mad_genes', axis='columns')
)

train_df = train_df[~train_df.index.duplicated(keep='first')]

print(train_df.shape)
train_df.head(2)


# In[6]:


# Get MAD genes again and sort
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


# In[9]:


mad_gene_df.head()


# It looks like the highest variable gene is a large outlier
# The gene is [B2M](http://useast.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000166710;r=15:44711477-44718877).

# In[10]:


# Distribution of gene expression variability after removing zeros
sns.distplot(mad_gene_df['mad_genes']);


# In[11]:


# Distribution of genes with high gene expression variability
sns.distplot(mad_gene_df.query("mad_genes > 100")['mad_genes']);


# In[12]:


# Get the proportion of total MAD variance for each gene
total_mad = mad_gene_df['mad_genes'].sum()
mad_gene_df = mad_gene_df.assign(variance_proportion = mad_gene_df['mad_genes'].cumsum() / total_mad)


# In[13]:


# Visualize the proportion of MAD variance against all non-zero genes
sns.regplot(x='index', y='variance_proportion', ci=None, fit_reg=False,
            data=mad_gene_df.reset_index().reset_index())
plt.xlabel('Number of Genes')
plt.ylabel('Proportion of Variance')
plt.axvline(x=5000, color='r', linestyle='--')
plt.axvline(x=10000, color='r', linestyle='--')
plt.axvline(x=num_genes, color='g', linestyle='--');


# In[14]:


# Use only the top `num_genes` in the classifier
mad_gene_df = mad_gene_df.assign(use_in_classifier = 0)
mad_gene_df['use_in_classifier'].iloc[range(0, num_genes)] = 1
mad_gene_df.head()


# In[15]:


# Write to file
file = os.path.join('data', 'mad_genes.tsv')
mad_gene_df.to_csv(file, sep='\t')


# ## Process gene expression data (Training and Testing)
# 
# 1. Zero-one normalization within dataset
# 2. Also subset to MAD genes derived from training set
# 
# This is done for easier processing in downstream analyses.

# In[16]:


use_genes = mad_gene_df.query('use_in_classifier == 1')


# ### Training X Matrix

# In[17]:


train_df = train_df.loc[use_genes.index, :]
train_df = train_df.sort_index(axis='columns')
train_df = train_df.sort_index(axis='rows')

fitted_scaler = MinMaxScaler().fit(train_df)
train_df = pd.DataFrame(fitted_scaler.transform(train_df),
                        columns=train_df.columns,
                        index=train_df.index)


# In[18]:


file = os.path.join('data', 'compass_x_train.tsv.gz')
train_df.to_csv(file, compression='gzip', sep='\t')


# ### Testing X Matrix
# 
# **Note that the testing matrix includes samples with both _KRAS_ and _NRAS_ mutations**
# 
# Remove these samples from the testing matrix and set aside for separate test phase.
# The data is written to file _after_ processing the Y matrix in order to separate dual _KRAS_/_NRAS_ samples.

# In[19]:


# Load and process test data
file = os.path.join('data', 'raw', 'CoMMpass_test_set.csv')
test_df = pd.read_csv(file, index_col=0).drop("Location", axis='columns')

# Reorder genes based on MAD and retain higher variable genes
test_df = (
    test_df
    .assign(mad_genes = mad(test_df, axis=1))
    .sort_values(by='mad_genes', ascending=False)
    .drop('mad_genes', axis='columns')
)

test_df = test_df[~test_df.index.duplicated(keep='first')]
print(test_df.shape)


# In[20]:


test_df = test_df.loc[use_genes.index, :]
test_df = test_df.sort_index(axis='columns')
test_df = test_df.sort_index(axis='rows')

fitted_scaler = MinMaxScaler().fit(test_df)
test_df = pd.DataFrame(fitted_scaler.transform(test_df),
                       columns=test_df.columns,
                       index=test_df.index)


# ## Process Y Matrices (Training and Testing)
# 
# This Y represents mutation status for all samples. 
# 
# Note that there are 26 samples (3.2%) that have dual _KRAS_ and _NRAS_ mutations. Split these samples into a different X and Y matrices.
# 
# Also, sklearn expects a single array of values for multiclass classifiers. Set the following assignments:
# 
# | Mutation | Assignment |
# | -------- | ---------- |
# | Wild-type | 0 |
# | _KRAS_ | 1 |
# | _NRAS_ | 2 |

# ### Training Y Matrix

# In[21]:


file = os.path.join('data', 'raw', 'CoMMpass_train_set_labels.csv')
y_train_df = pd.read_csv(file, index_col=0)


# In[22]:


y_train_df = y_train_df.sort_index(axis='rows')
y_train_df = y_train_df.reindex(train_df.columns)
y_train_df = y_train_df.astype(int)


# In[23]:


y_train_df.sum()


# In[24]:


y_train_df.sum() / y_train_df.shape[0]


# In[25]:


y_train_multi_df = y_train_df.drop(['dual_RAS_mut'], axis='columns')
y_train_multi_df.columns = ['KRAS_status', 'NRAS_status']

file = os.path.join('data', 'compass_y_train_multiclass.tsv')
y_train_multi_df.to_csv(file, sep='\t')


# In[26]:


# sklearn expects a single column with classes separate 0, 1, 2
# Set NRAS mutations equal to 2
y_train_df.loc[y_train_df['NRAS_mut'] == 1, 'KRAS_mut'] = 2

y_train_df = y_train_df.drop(['NRAS_mut', 'dual_RAS_mut'], axis='columns')
y_train_df.columns = ['ras_status']


# In[27]:


file = os.path.join('data', 'compass_y_train.tsv')
y_train_df.to_csv(file, sep='\t')


# ### Testing Y Matrix

# In[28]:


file = os.path.join('data', 'raw', 'CoMMpass_test_set_labels.csv')
y_test_df = pd.read_csv(file, index_col=0)


# In[29]:


y_test_df = y_test_df.sort_index(axis='rows')
y_test_df = y_test_df.reindex(test_df.columns)
y_test_df = y_test_df.astype(int)
y_test_df.head(3)


# In[30]:


# Split off dual Ras from normal testing
y_dual_df = y_test_df.query('dual_RAS_mut == 1')
y_test_df = y_test_df.query('dual_RAS_mut == 0')
print(y_dual_df.shape)
print(y_test_df.shape)


# In[31]:


y_test_df.sum()


# In[32]:


y_test_df.sum() / y_test_df.shape[0]


# In[33]:


file = os.path.join('data', 'compass_y_test.tsv')
y_test_df.drop('dual_RAS_mut', axis='columns').to_csv(file, sep='\t')


# ### Process and Output both X and Y matrices for Dual Ras mutated samples

# In[34]:


percent_dual = y_dual_df.shape[0] / (train_df.shape[1] + test_df.shape[1])
print('{0:.1f}% of the samples have mutations in both KRAS and NRAS'.format(percent_dual * 100))


# In[35]:


y_dual_df.head()


# In[36]:


file = os.path.join('data', 'compass_y_dual.tsv')
y_dual_df.drop('dual_RAS_mut', axis='columns').to_csv(file, sep='\t')


# #### Split Testing X Matrix based on Dual Y information

# In[37]:


test_dual_df = test_df.reindex(y_dual_df.index, axis='columns')
test_df = test_df.reindex(y_test_df.index, axis='columns')


# In[38]:


file = os.path.join('data', 'compass_x_test.tsv.gz')
test_df.to_csv(file, compression='gzip', sep='\t')


# In[39]:


file = os.path.join('data', 'compass_x_dual.tsv.gz')
test_dual_df.to_csv(file, compression='gzip', sep='\t')

