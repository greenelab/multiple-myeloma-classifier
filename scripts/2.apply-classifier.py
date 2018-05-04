
# coding: utf-8

# # Apply Classifier to Other Data
# 
# In this script we apply the classifier to obtain continuous scores that can be used in downstream applications.
# 
# We apply the multiple myeloma multiclass classifier trained in `1.train-classifier.py` to a cell line RNAseq data set. This data was also provided by Arun Wiita and Tony Lin (UCSF).
# 
# **Note: The classifier can be applied to other datasets by following the steps outlined in this notebook**

# In[1]:


import os
import random
import numpy as np
import pandas as pd

from statsmodels.robust.scale import mad
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve

import seaborn as sns
import matplotlib.pyplot as plt

from utils import shuffle_columns, apply_classifier


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


random.seed(1234)


# In[4]:


file = os.path.join('results', 'classifier', 'classifier_coefficients.tsv')
coef_df = pd.read_table(file, index_col=0)

coef_df.head()


# In[5]:


file = os.path.join('results', 'classifier', 'classifier_intercept.tsv')
intercept_df = pd.read_table(file, index_col=0)

intercept_df


# ## Load and Process Cell Line X Matrix

# In[6]:


file = os.path.join('data', 'raw', 'MMCL_RNAseq.csv')
mmcl_df = pd.read_csv(file, index_col=0).drop('GENE_NAME', axis='columns')

# Reorder genes based on MAD and retain higher variable genes
mmcl_df = (
    mmcl_df
    .assign(mad_genes = mad(mmcl_df, axis=1))
    .sort_values(by='mad_genes', ascending=False)
    .drop('mad_genes', axis='columns')
)

mmcl_df = mmcl_df[~mmcl_df.index.duplicated(keep='first')]

# Note that some genes were not measured in the MMCL data
mmcl_df = mmcl_df.reindex(coef_df.index).dropna().transpose()

print(mmcl_df.shape)
mmcl_df.head(2)


# In[7]:


# How many classifier genes are filtered?
nonzero_genes = set(coef_df.loc[(coef_df.sum(axis='columns').abs() > 0), :].index)

numcommon = len(set(mmcl_df.columns).intersection(nonzero_genes))
print("of the {} non zero genes, {} are present in MMCL ({} %)".format(len(nonzero_genes),
                                                                       numcommon,
                                                                       (numcommon / len(nonzero_genes)) * 100))


# In[9]:


# How many genes are missing
missing_genes = nonzero_genes.difference(set(mmcl_df.columns))
print("So, {} genes are missing.".format(len(missing_genes)))
missing_coef = coef_df.loc[missing_genes, :]
missing_coef = missing_coef.assign(abs_sum = missing_coef.abs().sum(axis='columns'))
missing_coef = missing_coef.sort_values(by='abs_sum', ascending=False)
print("Many of which are somewhat influential - must be careful in downstream interpretation")
missing_coef.head(5)


# ## Process Cell Line Y Matrix

# In[11]:


file = os.path.join('data', 'raw', 'MMCL_RNAseq_labels.csv')
y_df = (
    pd.read_csv(file, index_col=0)
    .reindex(mmcl_df.index)
    .astype(int)
)
y_df.head(3)


# In[12]:


# Number of Ras mutations in the cell line dataset
y_df.sum()


# In[13]:


# Proportion of Ras mutations in the cell line dataset
y_df.sum() / y_df.shape[0]


# In[14]:


# Recode the Y matrix
# sklearn expects a single column with classes separate 0, 1, 2
# Set NRAS mutations equal to 2
y_df.loc[y_df['NRAS_mut'] == 1, 'KRAS_mut'] = 2

y_df = y_df.drop(['NRAS_mut', 'dual_RAS_mut'], axis='columns')
y_df.columns = ['ras_status']
y_df.head()


# In[15]:


# Recode y matrix for metric eval
y_onehot_df = OneHotEncoder(sparse=False).fit_transform(y_df)


# ## Apply Classifier to Cell Line Data

# In[16]:


# Zero one normalize the cell line data
scaled_fit = MinMaxScaler().fit(mmcl_df)
mmcl_processed_df = pd.DataFrame(scaled_fit.transform(mmcl_df),
                                 index=mmcl_df.index,
                                 columns=mmcl_df.columns)
mmcl_processed_df.head()


# In[17]:


# Confirm that the samples are the same between training and testing
assert (y_df.index == mmcl_processed_df.index).all(), 'The samples between X and Y cell line matrices are not aligned!'


# In[18]:


# Use the `apply_classifier` custom function (found in `utils.py`)
mmcl_scores = apply_classifier(x=mmcl_processed_df,
                               w=coef_df,
                               b=intercept_df,
                               proba=True,
                               dropna=True)

file = os.path.join('results', 'mmcl_scores_cellline_set.tsv')
mmcl_scores.to_csv(file, sep='\t')

print(mmcl_scores.shape)
mmcl_scores.head(3)


# ### Apply classifier to a shuffled cell line X matrix

# In[19]:


# Shuffle training X matrix to observe potential metric inflation
# as a result of class imbalance
mmcl_shuffled_df = mmcl_processed_df.apply(shuffle_columns, axis=1)


# In[20]:


mmcl_shuffle_scores = apply_classifier(mmcl_shuffled_df,
                                       coef_df,
                                       intercept_df,
                                       proba=True, dropna=True)
mmcl_shuffle_scores.head()


# ## Obtain classification metrics for the Cell Line Data

# In[21]:


n_classes = 3

fpr_cell = {}
tpr_cell = {}
precision_cell = {}
recall_cell = {}
auroc_cell = {}
aupr_cell = {}

fpr_shuff = {}
tpr_shuff = {}
precision_shuff = {}
recall_shuff = {}
auroc_shuff = {}
aupr_shuff = {}

for i in range(n_classes):
    # Obtain Training Metrics
    train_onehot_class = y_onehot_df[:, i]
    train_score_class = mmcl_scores.iloc[:, i]
    
    fpr_cell[i], tpr_cell[i], _ = roc_curve(train_onehot_class, train_score_class)
    precision_cell[i], recall_cell[i], _ = precision_recall_curve(train_onehot_class, train_score_class)
    auroc_cell[i] = roc_auc_score(train_onehot_class, train_score_class)
    aupr_cell[i] = average_precision_score(train_onehot_class, train_score_class)
    
    # Obtain Shuffled Metrics
    shuff_score_class = mmcl_shuffle_scores.iloc[:, i]
    
    fpr_shuff[i], tpr_shuff[i], _ = roc_curve(train_onehot_class, shuff_score_class)
    precision_shuff[i], recall_shuff[i], _ = precision_recall_curve(train_onehot_class, shuff_score_class)
    auroc_shuff[i] = roc_auc_score(train_onehot_class, shuff_score_class)
    aupr_shuff[i] = average_precision_score(train_onehot_class, shuff_score_class)


# In[22]:


# Visualize ROC curves
plt.subplots(figsize=(4, 4))

labels = ['Wildtype Cell', 'KRAS Cell', 'NRAS Cell']
colors = ['#1b9e77', '#d95f02', '#7570b3']
for i in range(n_classes):
    plt.plot(fpr_cell[i], tpr_cell[i],
             label='{} (AUROC = {})'.format(labels[i], round(auroc_cell[i], 2)),
             linestyle='solid',
             color=colors[i])
    
     # Shuffled Data
    plt.plot(fpr_shuff[i], tpr_shuff[i],
             label='{} Shuffle (AUROC = {})'.format(labels[i], round(auroc_shuff[i], 2)),
             linestyle='dotted',
             color=colors[i])

    
plt.axis('equal')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)

plt.tick_params(labelsize=10)

lgd = plt.legend(bbox_to_anchor=(1.03, 0.85),
                 loc=2,
                 borderaxespad=0.,
                 fontsize=10)

file = os.path.join('figures', 'cellline_roc_curve.pdf')
plt.savefig(file, bbox_extra_artists=(lgd,), bbox_inches='tight')


# In[23]:


# Visualize PR curves
plt.subplots(figsize=(4, 4))

for i in range(n_classes):
    plt.plot(recall_cell[i], precision_cell[i],
             label='{} (AUPR = {})'.format(labels[i], round(aupr_cell[i], 2)),
             linestyle='solid',
             color=colors[i])
    
     # Shuffled Data
    plt.plot(recall_shuff[i], precision_shuff[i],
             label='{} Shuffle (AUPR = {})'.format(labels[i], round(aupr_shuff[i], 2)),
             linestyle='dotted',
             color=colors[i])
    
plt.axis('equal')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)

plt.tick_params(labelsize=10)

lgd = plt.legend(bbox_to_anchor=(1.03, 0.85),
                 loc=2,
                 borderaxespad=0.,
                 fontsize=10)

file = os.path.join('figures', 'cellline_pr_curve.pdf')
plt.savefig(file, bbox_extra_artists=(lgd,), bbox_inches='tight')


# ## Plot Classifier Score Distributions
# 
# This plot represents multiclass probability (`x axis`) scores (`y axis`) for all samples (`points`) and their corresponding ground truth status (`colors`).

# In[24]:


class_dist_df = (
    mmcl_scores.reset_index()
    .merge(y_df, left_on='index', right_index=True)
    .melt(id_vars=['ras_status', 'index'], var_name='proba', value_name='class_proba')
    .sort_values(by='index')
    .reset_index(drop=True)
)
class_dist_df.head(6)


# In[25]:


plt.subplots(figsize=(5.5, 3.5))
ax = sns.boxplot(x="proba",
                 y="class_proba",
                 data=class_dist_df,
                 hue='ras_status',
                 palette = {0: "whitesmoke", 1: 'gainsboro', 2: 'grey'},
                 fliersize=0)
ax = sns.stripplot(x="proba",
                   y="class_proba",
                   data=class_dist_df,
                   hue='ras_status', 
                   dodge=True,
                   edgecolor='black',
                   palette = {1: "seagreen", 0: 'goldenrod', 2: 'blue'},
                   jitter=0.25,
                   size=4,
                   alpha=0.65)

ax.set_ylabel('Classifier Score', fontsize=12)
ax.set_xlabel('Class Predictions', fontsize=12)

handles, labels = ax.get_legend_handles_labels()
lgd = plt.legend(handles[3:6], ['Wild-Type', 'KRAS', 'NRAS'],
               bbox_to_anchor=(1.03, 0.8),
                 loc=2,
                 borderaxespad=0.,
                 fontsize=10)
lgd.set_title("Cell Line Class")

file = os.path.join('figures', 'cellline_predictions_boxscatter.pdf')
plt.savefig(file, bbox_extra_artists=(lgd,), bbox_inches='tight')

