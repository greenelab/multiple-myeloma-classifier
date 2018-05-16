
# coding: utf-8

# # Train a Multiclass Logistic Regression Classifier to Predict _KRAS_, _NRAS_ mutations in Multiple Myeloma
# 
# **Gregory Way 2018**
# 
# The following notebook contains code to train a multiclass logistic regression classifier to predict _KRAS_ and _NRAS_ mutations in multiple myeloma RNAseq data from the [MMRF CoMMpass Study](https://www.themmrf.org/research-partners/mmrf-data-bank/the-mmrf-commpass-study/).
# Previously, we trained a Ras activation classifier ([Way et al. 2018](https://doi.org/10.1016/j.celrep.2018.03.046 "Machine Learning Detects Pan-cancer Ras Pathway Activation in The Cancer Genome Atlas")) and an abberant TP53 classifier ([Knijnenburg et al. 2018](https://doi.org/10.1016/j.celrep.2018.03.076 "Genomic and Molecular Landscape of DNA Damage Repair Deficiency across The Cancer Genome Atlas")) using TCGA PanCanAtlas RNAseq data and an NF1 inactivation classifier in glioblastoma ([Way et al. 2017](https://doi.org/10.1186/s12864-017-3519-7 "A machine learning classifier trained on cancer transcriptomes detects NF1 inactivation signal in glioblastoma")). We have also implemented classifiers on a larger scale and made the models accessible to non-computational biologists (see [Project Cognoma](http://cognoma.org)).
# 
# Here, we implemented a one vs. rest (ovr) [multiclass classifier](https://en.wikipedia.org/wiki/Multiclass_classification) to predict _KRAS_ and _NRAS_ mutations separately in multiple myeloma.
# The hypothesis was that the two mutations result in different downstream biology that machine learning classifiers can detect. Detecting the two mutations separately can potentially inform different treatment options.
# 
# Our collaborators at UCSF (Arun Wiita and Tony Lin) randomly partitioned 10% (mutation-balanced) of the input gene expression data into training (n = 706) and testing (n = 80) sets. We used the training set with the [sklearn.linear_model.LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) class to train models. We selected optimal regularization (`C`) and penalty (`l1` vs. `l2` loss) terms following 5 fold cross validation. We observed optimal hyperparameters of `C=0.45` and `penalty=l1`. We applied the model to the held out test set to assess performance. We also apply the model to a series of 26 multiple myeloma samples with dual _KRAS_/_NRAS_ mutations.
# 
# ## Notebook Outputs
# 
# The output of this notebook are:
# 
# 1. Predictions for each class for training, testing, and dual mutation sets.
# 2. Gene coefficients (importance scores) for each ovr multiclass classifier.
# 3. Receiver operating characteristic (ROC) and precision-recall (PR) curves for training, testing, and randomly shuffled predictions for each ovr classifier

# In[1]:


import os
import random
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from dask_searchcv import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from utils import shuffle_columns, get_confusion_matrix


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


random.seed(1234)


# In[4]:


# hyperparameters to loop over
cs = [0.001, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 1, 10]
penalties = ['l1', 'l2']


# In[5]:


# Read in training RNAseq data (X matrix)
# RNAseq data was FPKM normalized and then MinMax scaled by gene to a range of (0, 1)
file = os.path.join('data', 'compass_x_train.tsv.gz')
x_df = pd.read_table(file, index_col=0)

print(x_df.shape)
x_df.head(2)


# In[6]:


# Read in training status data (Y matrix)
# Wildtype = 0, KRAS mut = 1, NRAS mut = 2
file = os.path.join('data', 'compass_y_train.tsv')
y_df = pd.read_table(file, index_col=0)

print(y_df.shape)
print(y_df.ras_status.value_counts())
y_df.head(2)


# ## 5-Fold Cross Validation
# 
# ### Initialize Pipeline

# In[7]:


# Build the 5-fold cross validation architecture
clf_parameters = {'classify__C': cs,
                  'classify__penalty': penalties}

estimator = Pipeline(
    steps=[
        ('classify',
         LogisticRegression(random_state=123,
                            class_weight='balanced',
                            multi_class='ovr',
                            max_iter=100,
                            solver='saga')
        )
    ]
)

# Custom scorer that optimizes f1 score weighted by class proportion
weighted_f1_scorer = make_scorer(f1_score, average='weighted')

# Cross validation pipeline
cv_pipeline = GridSearchCV(estimator=estimator,
                           param_grid=clf_parameters,
                           n_jobs=-1,
                           cv=5,
                           return_train_score=True,
                           scoring=weighted_f1_scorer)


# ### Fit Model
# 
# This takes a couple minutes to train. For many model parameters, sklearn will throw convergence warnings. This means that after 100 iterations, an optimal solution is not found.  Prevent the warnings from being printed redundantly.

# In[8]:


get_ipython().run_cell_magic('time', '', 'with warnings.catch_warnings():\n    warnings.simplefilter("ignore")\n    cv_pipeline.fit(X=x_df, y=y_df.ras_status)')


# In[9]:


# Compile cross validation results
cv_results = (
    pd.concat([
        pd.DataFrame(cv_pipeline.cv_results_).drop('params', axis=1),
        pd.DataFrame.from_records(cv_pipeline.cv_results_['params'])
    ], axis=1)
)

cv_results.sort_values(by='rank_test_score').head(5)


# In[10]:


# Cross-validated performance heatmap
cv_score_mat = pd.pivot_table(cv_results,
                              values='mean_test_score',
                              index='classify__penalty',
                              columns='classify__C')

plt.subplots(figsize=(8,4))
ax = sns.heatmap(cv_score_mat, annot=True, fmt='.1%')
ax.set_xlabel('Regularization strength multiplier (C)')
ax.set_ylabel('Penalty Term')
plt.tight_layout()


# ## Output Results from Optimal Model
# 
# ### Extract and Save Classifier Coefficients

# In[11]:


path = os.path.join('results', 'classifier')
if not os.path.exists(path):
    os.makedirs(path)


# In[12]:


coef_df = pd.DataFrame(
    cv_pipeline
    .best_estimator_
    .named_steps['classify']
    .coef_).T

coef_df.columns = ['wildtype', 'KRAS', 'NRAS']
coef_df.index = x_df.columns
coef_df.index.name = 'GENE_ID'
coef_df = coef_df.sort_values(by='KRAS', ascending=False)

file = os.path.join('results', 'classifier', 'classifier_coefficients.tsv')
coef_df.to_csv(file, sep='\t')

coef_df.head()


# In[13]:


# How many classifier genes are nonzero?
num_nonzero = (coef_df.sum(axis='columns').abs() > 0).sum()
print('{} genes are nonzero ({} %)'.format(num_nonzero,
                                           (num_nonzero / coef_df.shape[0]) * 100))


# In[14]:


# Save the intercept of the multiclass classifier
intercept_df = pd.DataFrame(cv_pipeline.best_estimator_.named_steps['classify'].intercept_).T
intercept_df.columns = ['wildtype', 'KRAS', 'NRAS']
intercept_df.index = ['intercept']

file = os.path.join('results', 'classifier', 'classifier_intercept.tsv')
intercept_df.to_csv(file, sep='\t')

intercept_df


# ### Obtain probability estimates for all training samples

# In[15]:


scores_df = cv_pipeline.best_estimator_.predict_proba(x_df)

scores_df = pd.DataFrame(scores_df,
                         index=x_df.index,
                         columns=['wildtype', 'KRAS', 'NRAS'])

file = os.path.join('results', 'sample_scores_training_set.tsv')
scores_df.to_csv(file, sep='\t')

print(scores_df.shape)
scores_df.head()


# In[16]:


# Visualize probability estimates against ground truth status
# Compress Y to between 0 and 1 (in other words, 0 = wildtype, 0.5 = KRAS, 1 = NRAS)
# Do this for visualization purposes only
y_compress = pd.DataFrame(MinMaxScaler().fit_transform(y_df),
                          index=y_df.index,
                          columns=y_df.columns)

score_heatmap = (
    (1 - scores_df)
    .join(y_compress)
    .sort_values(by=['ras_status', 'wildtype', 'KRAS', 'NRAS'])
    .T
)

plt.subplots(figsize=(18,2))
sns.heatmap(score_heatmap);


# In[18]:


# Get Confusion Matrix for Training Data
y_train_pred = cv_pipeline.best_estimator_.predict(x_df)
y_train_true = np.array(y_df.ras_status)

plt.subplots(figsize=(5,5))

train_confusion, ax = get_confusion_matrix(y_true=y_train_true,
                                           y_pred=y_train_pred)

ax.set_title('Training Confusion Matrix')
file = os.path.join('figures', 'train_confusion_matrix.png')
plt.savefig(file, bbox_inches='tight')

file = os.path.join('results', 'training_confusion_matrix.tsv')
train_confusion.to_csv(file, sep='\t')
train_confusion


# ## Apply Classifier to Testing and Shuffled Sets
# 
# A shuffled matrix will determine how the model would have performed withough any signal. This tests if class imbalance induces biased metrics.
# 
# ### Load and process testing set
# 
# Scores for this set are output as well.

# In[19]:


# Read in testing RNAseq data (X matrix)
file = os.path.join('data', 'compass_x_test.tsv.gz')
x_test_df = pd.read_table(file, index_col=0)

print(x_test_df.shape)
x_test_df.head(2)


# In[20]:


# Read in testing status data (Y matrix)
# Wildtype = 0, KRAS mut = 1, NRAS mut = 2
file = os.path.join('data', 'compass_y_test.tsv')
y_test_df = pd.read_table(file, index_col=0)

print(y_test_df.shape)
print(y_test_df.ras_status.value_counts())
y_test_df.head(2)


# In[21]:


# Apply classifier to testing data and get scores
test_scores_df = cv_pipeline.best_estimator_.predict_proba(x_test_df)
test_scores_df = pd.DataFrame(test_scores_df,
                              index=x_test_df.index,
                              columns=['wildtype', 'KRAS', 'NRAS'])

file = os.path.join('results', 'sample_scores_testing_set.tsv')
test_scores_df.to_csv(file, sep='\t')

print(test_scores_df.shape)
test_scores_df.head(2)


# In[22]:


# Visualize probability estimates against ground truth status for the testing set
# Compress Y to between 0 and 1 (in other words, 0 = wildtype, 0.5 = KRAS, 1 = NRAS)
# Do this for visualization purposes only
y_test_compress = pd.DataFrame(MinMaxScaler().fit_transform(y_test_df),
                               index=y_test_df.index,
                               columns=y_test_df.columns)

score_heatmap = (
    (1 - test_scores_df)
    .join(y_test_compress)
    .sort_values(by=['ras_status', 'wildtype', 'KRAS', 'NRAS'])
    .T
)

plt.subplots(figsize=(18,2))
sns.heatmap(score_heatmap);


# In[23]:


# Get Confusion Matrix for Testing Data
y_test_pred = cv_pipeline.best_estimator_.predict(x_test_df)
y_test_true = np.array(y_test_df.ras_status)

plt.subplots(figsize=(5,5))

test_confusion, ax = get_confusion_matrix(y_true=y_test_true,
                                          y_pred=y_test_pred)

ax.set_title('Test Set Confusion Matrix')
file = os.path.join('figures', 'test_confusion_matrix.png')
plt.savefig(file, bbox_inches='tight')

file = os.path.join('results', 'test_confusion_matrix.tsv')
test_confusion.to_csv(file, sep='\t')
test_confusion


# ### Get a Shuffled Training X Matrix

# In[24]:


# Shuffle training X matrix to observe potential metric inflation
# as a result of class imbalance
x_shuffled_df = x_df.apply(shuffle_columns, axis=1)


# In[25]:


shuffle_scores_df = cv_pipeline.best_estimator_.predict_proba(x_shuffled_df)
shuffle_scores_df = pd.DataFrame(shuffle_scores_df,
                                 index=x_df.index,
                                 columns=['wildtype', 'KRAS', 'NRAS'])

print(shuffle_scores_df.shape)
shuffle_scores_df.head()


# In[26]:


# Get Confusion Matrix for Shuffled Data
y_shuff_pred = cv_pipeline.best_estimator_.predict(x_shuffled_df)
y_shuff_true = np.array(y_df.ras_status)

plt.subplots(figsize=(5,5))

shuff_confusion, ax = get_confusion_matrix(y_true=y_shuff_true,
                                           y_pred=y_shuff_pred)

ax.set_title('Shuffled Data Confusion Matrix')
file = os.path.join('figures', 'shuffled_confusion_matrix.png')
plt.savefig(file, bbox_inches='tight')

file = os.path.join('results', 'shuffled_confusion_matrix.tsv')
shuff_confusion.to_csv(file, sep='\t')
shuff_confusion


# ## Collect Classification Metrics for Training, Testing, and Shuffled Data
# 
# 1. False Positive Rate
# 2. True Positive Rate
# 3. Precision
# 4. Recall
# 5. Area under the ROC (AUROC) Curve
# 6. Area under the PR (AUPR) Curve
# 
# Also output ROC and PR Curves to the `figures` folder.

# In[27]:


path = os.path.join('figures')
if not os.path.exists(path):
    os.makedirs(path)


# In[28]:


# Convert the y vector into one hot encoded matrices
# Note that the shuffled matrix will use the same training onehot vector
onehot = OneHotEncoder(sparse=False).fit(y_df)

train_onehot_df = onehot.transform(y_df)
test_onehot_df = onehot.transform(y_test_df)


# In[29]:


n_classes = 3

fpr = {}
tpr = {}
precision = {}
recall = {}
auroc = {}
aupr = {}

fpr_test = {}
tpr_test = {}
precision_test = {}
recall_test = {}
auroc_test = {}
aupr_test = {}

fpr_shuff = {}
tpr_shuff = {}
precision_shuff = {}
recall_shuff = {}
auroc_shuff = {}
aupr_shuff = {}

for i in range(n_classes):
    # Obtain Training Metrics
    train_onehot_class = train_onehot_df[:, i]
    train_score_class = scores_df.iloc[:, i]
    
    fpr[i], tpr[i], _ = roc_curve(train_onehot_class, train_score_class)
    precision[i], recall[i], _ = precision_recall_curve(train_onehot_class, train_score_class)
    auroc[i] = roc_auc_score(train_onehot_class, train_score_class)
    aupr[i] = average_precision_score(train_onehot_class, train_score_class)
    
    # Obtain Testing Metrics
    test_onehot_class = test_onehot_df[:, i]
    test_score_class = test_scores_df.iloc[:, i]
    
    fpr_test[i], tpr_test[i], _ = roc_curve(test_onehot_class, test_score_class)
    precision_test[i], recall_test[i], _ = precision_recall_curve(test_onehot_class, test_score_class)
    auroc_test[i] = roc_auc_score(test_onehot_class, test_score_class)
    aupr_test[i] = average_precision_score(test_onehot_class, test_score_class)
    
    # Obtain Shuffled Metrics
    shuff_score_class = shuffle_scores_df.iloc[:, i]
    
    fpr_shuff[i], tpr_shuff[i], _ = roc_curve(train_onehot_class, shuff_score_class)
    precision_shuff[i], recall_shuff[i], _ = precision_recall_curve(train_onehot_class, shuff_score_class)
    auroc_shuff[i] = roc_auc_score(train_onehot_class, shuff_score_class)
    aupr_shuff[i] = average_precision_score(train_onehot_class, shuff_score_class)


# In[30]:


plt.subplots(figsize=(4, 4))

labels = ['Wildtype', 'KRAS', 'NRAS']
colors = ['dodgerblue', 'salmon', 'goldenrod']
for i in range(n_classes):
    # Training Data
    plt.plot(fpr[i], tpr[i],
             label='{} Train (AUROC = {})'.format(labels[i], round(auroc[i], 2)),
             linestyle='solid',
             color=colors[i])

    # Testing Data
    plt.plot(fpr_test[i], tpr_test[i],
             label='{} Test (AUROC = {})'.format(labels[i], round(auroc_test[i], 2)),
             linestyle='dashed',
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

file = os.path.join('figures', 'roc_curve.pdf')
plt.savefig(file, bbox_extra_artists=(lgd,), bbox_inches='tight')


# In[31]:


plt.subplots(figsize=(4, 4))

for i in range(n_classes):
    # Training Data
    plt.plot(recall[i], precision[i],
             label='{} Train (AUPR = {})'.format(labels[i], round(aupr[i], 2)),
             linestyle='solid',
             color=colors[i])

    # Testing Data
    plt.plot(recall_test[i], precision_test[i],
             label='{} Test (AUPR = {})'.format(labels[i], round(aupr_test[i], 2)),
             linestyle='dashed',
             color=colors[i])

    # Shuffled Data
    plt.plot(recall_shuff[i], precision_shuff[i],
             label='{} Shuffle (AUPR = {})'.format(labels[i], round(aupr_shuff[i], 2)),
             linestyle='dotted',
             color=colors[i])

plt.axis('equal')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)

plt.tick_params(labelsize=10)

lgd = plt.legend(bbox_to_anchor=(1.03, 0.85),
                 loc=2,
                 borderaxespad=0.,
                 fontsize=10)

file = os.path.join('figures', 'pr_curve.pdf')
plt.savefig(file, bbox_extra_artists=(lgd,), bbox_inches='tight')


# ## Load Dual Ras Samples and Apply Classifier
# 
# Output classifier scores per Dual Ras mutated sample

# In[32]:


# Read in dual RNAseq data (X matrix)
file = os.path.join('data', 'compass_x_dual.tsv.gz')
x_dual_df = pd.read_table(file, index_col=0)

print(x_dual_df.shape)
x_dual_df.head(2)


# In[33]:


# Apply classifier to dual data and save scores
dual_scores_df = cv_pipeline.best_estimator_.predict_proba(x_dual_df)
dual_scores_df = pd.DataFrame(dual_scores_df,
                              index=x_dual_df.index,
                              columns=['wildtype', 'KRAS', 'NRAS'])

file = os.path.join('results', 'sample_scores_dual_set.tsv')
dual_scores_df = dual_scores_df.sort_values(by='wildtype')
dual_scores_df.to_csv(file, sep='\t')

print(dual_scores_df.shape)
dual_scores_df.head()


# In[34]:


# Visualize dual Ras sample score heatmap
dual_score_heatmap = (
    (1 -scores_df)
    .sort_values(by=['wildtype', 'KRAS', 'NRAS'])
    .T
)

plt.subplots(figsize=(18,1.5))
g = sns.heatmap(dual_score_heatmap);


# In[35]:


# Observe the predictions on the dual mutated samples
y_dual_pred = cv_pipeline.best_estimator_.predict(x_dual_df)
pd.Series(y_dual_pred).value_counts()

