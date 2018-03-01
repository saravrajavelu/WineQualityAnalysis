
# coding: utf-8

# In[1]:


# Loading libraries
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import time
import collections
import numpy as np
import math
from IPython.display import Markdown, display
from sklearn.metrics import pairwise

COUNT = 1
DISTANCE_WEIGHTED = 2
EUCLID = 3
COSINE = 4

get_ipython().run_line_magic('matplotlib', 'inline')


def printmd(string):
    display(Markdown(string))


# In[67]:


# Import data
wine = pd.read_csv('winequality-red.csv', sep = ';')


# In[68]:


# Creating the class attribute
wine['class'] = ['Low' if i <= 5 else "High" for i in wine.quality]
wine['class'].value_counts().plot(kind = 'bar', title = 'Distribution of classes')

# Dropping quality attribute
wine.drop(columns = ['quality'], inplace = True)

# Paritioning dependent attributes from the independent attributes
wine_data = wine[wine.columns[wine.columns != 'class']].copy()
wine_label = wine['class'].copy()
print(pd.DataFrame(wine_label.value_counts()))


# In[69]:


wine.head()


# In[70]:


wine_data.describe()


# In[71]:


printmd('## Correlation Matrix')
wine_data.corr(method = 'pearson')


# In[72]:


printmd('## Correlated Pairs')
correlated_pairs = wine_data.corr()
top_correlated_pairs = correlated_pairs.unstack().sort_values(kind="quicksort")
pd.DataFrame(top_correlated_pairs[((top_correlated_pairs < -0.5) & (top_correlated_pairs > -1)) | ((top_correlated_pairs>0.5) & (top_correlated_pairs <1))], columns = ['Coefficient'])


# In[73]:


# Dropping fixed acidity attribute
wine_data.drop(columns = ['fixed acidity','total sulfur dioxide'], inplace = True)


# In[74]:


printmd('## Boxplot')
wine_data.plot(kind = 'box', figsize = (20,10), title = 'Boxplot')


# In[75]:


printmd('## Outlier Analysis')
# Removing outliers
# Keeping records where the column values are within +3/-3 standard deviations from the mean

outlier_filter = (np.abs(wine_data - wine_data.mean()) <= (3*wine_data.std())).all(1)
wine_data = wine_data[outlier_filter] 
wine_label = wine_label[outlier_filter]

printmd('### Boxplot')
wine_data.plot(kind = 'box', figsize = (20,10), title = 'Boxplot')


# In[76]:


# Splitting data into train/test
X = np.array(wine_data)
y = np.array(wine_label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[92]:


def euclidian_distance(A, B):
    if not (A.ndim == 1 and B.ndim == 1):
        raise ValueError("Both numpy arrays should be single rows (1 dimensional).")
    return np.sqrt(np.sum(np.square(A - B)))

def cosine_similarity(A, B):
    if not (A.ndim == 1 and B.ndim == 1):
        raise ValueError("Both numpy arrays should be single rows (1 dimensional).")
    return np.sum(A*B)/(np.sqrt(np.sum(A*A)) * np.sqrt(np.sum(B*B)))


def predict(train_X, labels, test, K, metric, measure):
    distances = []
    for i, sample in enumerate(train_X):
        if measure == EUCLID:
            distance = euclidian_distance(sample, test)
        elif measure == COSINE:
            distance = cosine_similarity(sample, test)
        distances.append((distance, i))

    distances.sort()

    return predict_label(distances, labels, K, metric, measure)

def predict_label(distances, labels, K, metric, measure):
    if metric == COUNT:
        k_closest = [labels[x[1]] for x in distances[:K]]
        counts = collections.Counter(k_closest)
        return counts.most_common()[0][0], counts.most_common()[0][1] / K
    if metric == DISTANCE_WEIGHTED:
        label_sum = {}
        max_sum = -1
        count = 0
        for distance, index in distances:
            if distance == 0: continue
            if count == K: break
            count += 1
            label_sum[labels[index]] = label_sum.get(labels[index], 0) + (1 / distance)
            if label_sum[labels[index]] > max_sum:
                max_sum = label_sum[labels[index]]
                predicted_label = labels[index]
        return predicted_label, max_sum / sum(label_sum.values())
    
def kNN(X_train, y_train, X_test, y_test, K, metric, measure):
    predicted = []
    for i, test in enumerate(X_test):
        predicted_class, posterior = predict(X_train, y_train, test, K, metric, measure)
        actual_class = y_test[i]
        predicted.append((actual_class, predicted_class, posterior))
    
    prediction = pd.DataFrame.from_records(predicted, columns=["Actual", "Predicted", "Posterior"])
    
    return prediction

def get_accuracy(prediction_df):
    return prediction_df[prediction_df["Actual"] == prediction_df["Predicted"]].shape[0] / prediction_df.shape[0]


# In[78]:


# Maximum accuracy is found at K = 38 for the euclidian distance measure.
def optimal_K(metric, measure):
    K_candidates = range(1,50,2)
    errors = []
    for K in K_candidates:
        accuracy = get_accuracy(kNN(X_train, y_train, X_test, y_test, K, metric, measure))
        errors.append(1 - accuracy)
    
    plt.plot(K_candidates, errors)
    plt.show()
    return np.where(errors==np.min(errors))


# In[14]:


def confusion_matrix(prediction_df):
    y = list(prediction_df.Actual)
    x = list(prediction_df.Predicted)
    if len(x) != len(y):
        return 'Length do not match'
    TP, TN, FP, FN = 0,0,0,0
    for i in range(len(x)):
        if y[i] == 'High':
            if x[i] == 'High':
                TP += 1
            else:
                FN += 1
        else:
            if x[i] == 'High':
                FP += 1
            else:
                TN += 1
    accuracy = ((TP+TN))/(TP+FN+FP+TN)
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
    f_measure = (2*recall*precision)/(recall+precision)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    error_rate = 1 - accuracy
    
    return accuracy, precision, recall, f_measure, sensitivity, specificity, error_rate

def optimal_K_confusion(metric, measure):
    K_candidates = range(1,50,2)
    out = {}
    for K in K_candidates:
        accuracy, precision, recall, f_measure, sensitivity, specificity, error_rate = confusion_matrix(kNN(X_train, y_train, X_test, y_test, K, metric, measure))
        out['K'] = out.get('K',[]) + [K]
        out['f_measure'] = out.get('f_measure',[]) + [f_measure]
        out['sensitivity'] = out.get('sensitivity',[]) + [sensitivity]
        out['specificity'] = out.get('specificity',[]) + [specificity]
        out['error_rate'] = out.get('error_rate',[]) + [error_rate]
        out['accuracy'] = out.get('accuracy',[]) + [accuracy]
        out['precision'] = out.get('precision',[]) + [precision]
        out['recall'] = out.get('recall',[]) + [recall]
        
    return pd.DataFrame(out)



# In[15]:


printmd('# Model Evaluation')


# In[16]:


# kNN using COUNT method and Euclidean distance as proximity measure
base_count_euclid = optimal_K_confusion(COUNT, EUCLID)
base_count_euclid.sort_values(by = ['f_measure', 'accuracy'], ascending = False)


# In[17]:


base_count_cosine = optimal_K_confusion(COUNT, COSINE)
base_count_cosine.sort_values(by = ['f_measure', 'accuracy'], ascending = False)


# In[18]:


base_DW_euclid = optimal_K_confusion(DISTANCE_WEIGHTED, EUCLID)
base_DW_euclid.sort_values(by = ['f_measure', 'accuracy'], ascending = False)


# In[19]:


base_DW_cosine = optimal_K_confusion(DISTANCE_WEIGHTED, COSINE)
base_DW_cosine.sort_values(by = ['f_measure', 'accuracy'], ascending = False)


# In[20]:


# Normalizing the data using mix-max method
X_train, X_test, Y_train, Y_test = train_test_split(wine_data, wine_label, test_size=0.25, random_state=42)
X_train_norm = (X_train - X_train.min()) / (X_train.max() -  X_train.min())
X_test_norm = (X_test - X_train.min()) / (X_train.max() -  X_train.min())
X_train = np.array(X_train_norm)
X_test = np.array(X_test_norm)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)


# In[21]:


norm_count_euclid = optimal_K_confusion(COUNT, EUCLID)
norm_count_euclid.sort_values(by = ['f_measure', 'accuracy'], ascending = False)


# In[23]:


norm_count_cosine = optimal_K_confusion(COUNT, COSINE)
norm_count_cosine.sort_values(by = ['f_measure', 'accuracy'], ascending = False)


# In[22]:


norm_DW_euclid = optimal_K_confusion(DISTANCE_WEIGHTED, EUCLID)
norm_DW_euclid.sort_values(by = ['f_measure', 'accuracy'], ascending = False)


# In[ ]:


norm_DW_cosine = optimal_K_confusion(DISTANCE_WEIGHTED, COSINE)
norm_DW_cosine.sort_values(by = ['f_measure', 'accuracy'], ascending = False)


# In[60]:


printmd('# Comparing with 50% data')

wine2, wine2_n = train_test_split(wine, test_size = 0.5)
wine_data = wine2[wine2.columns[wine.columns != 'class']].copy()
wine_label = wine2['class'].copy()
X_train, X_test, Y_train, Y_test = train_test_split(wine_data, wine_label, test_size=0.25, random_state=42)
X_train_norm = (X_train - X_train.min()) / (X_train.max() -  X_train.min())
X_test_norm = (X_test - X_train.min()) / (X_train.max() -  X_train.min())
X_train = np.array(X_train_norm)
X_test = np.array(X_test_norm)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

f_base_count_euclid = optimal_K_confusion(COUNT, EUCLID)
f_base_count_euclid.sort_values(by = ['f_measure', 'accuracy'], ascending = False)


# In[53]:


printmd('# Comparing with off the shelf KNN classifiers')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

wine_data = wine[wine.columns[wine.columns != 'class']].copy()
wine_label = wine[wine.columns[wine.columns == 'class']].copy()
wine_data.drop(columns = ['fixed acidity','total sulfur dioxide'], inplace = True)
out = {}
train_data, test_data, train_label, test_label = train_test_split(wine_data, wine_label, train_size = 0.75, test_size = 0.25, random_state = 42) 

K_candidates = range(1,50,2)
out = {}
for K in K_candidates:
    knn = KNeighborsClassifier(algorithm='auto',  metric='minkowski', 
                               metric_params=None, n_jobs=-1, n_neighbors=K, p=2,
                           weights='uniform')
    knn.fit(train_data, train_label['class']) 
    accuracy, precision, recall, f_measure, sensitivity, specificity, error_rate = confusion_matrix(pd.DataFrame({'Actual' : test_label['class'],'Predicted' : knn.predict(test_data)}))
    out['K'] = out.get('K',[]) + [K]
    out['f_measure'] = out.get('f_measure',[]) + [f_measure]
    out['sensitivity'] = out.get('sensitivity',[]) + [sensitivity]
    out['specificity'] = out.get('specificity',[]) + [specificity]
    out['error_rate'] = out.get('error_rate',[]) + [error_rate]
    out['accuracy'] = out.get('accuracy',[]) + [accuracy]
    out['precision'] = out.get('precision',[]) + [precision]
    out['recall'] = out.get('recall',[]) + [recall]
pd.DataFrame(out).sort_values(by = ['f_measure', 'accuracy'], ascending = False)


# In[93]:


# KNN output
kNN(X_train, y_train, X_test, y_test, K, COUNT, EUCLID)

