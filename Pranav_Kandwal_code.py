#!/usr/bin/env python
# coding: utf-8

# #PRE-PROCESSING OF DATA

# Importing of data set and required python packages
# 
# Extracting information from our data set(for eg: no. of features and total no. of data points or feature vectors)
# 
# Removal of duplicates in our data set
# 
# Getting our training set(x) and training class label(y)

# In[1]:


import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import linear_model

X=pd.read_csv("music30s_test.csv")
print(X.describe())
df=pd.read_csv("merged_Data.csv")
df.drop(df.columns[0],axis=1,inplace=True)
print("***************ORIGINAL DATA SET***************\n")
print(df)
print("\n**********DATA SET AFTER REMOVAL OF DUPLICATE ENTERIES**********\n")
df.drop_duplicates(inplace=True)
print(df)


# **Class** **Labels**
# 

# In[2]:


y=df.label
print(y)
print(y.describe())


# **Training** **Data** **Set**

# In[3]:


x=df.drop('label',axis=1)
print(x)
print(x.describe())


# In[3]:





# # VISUALISATION OF DATA

# **Analysing** **the** **distribution** **of** **the** **target** **variables**
# 
# To understand about the evenness and unevenness of class label distribution to check whether there is a class imbalance problem or not.
# 
# Visualising the eveness and uneveness of our training test data through pie chart and bar graph

# In[ ]:


# creating list for all the possible  values of the target variables
class_label_name=np.array(["rock","disco","classical","hiphop","pop","jazz","metal","blues","country","reggae"])

#creating list for storing the frequency of target variables
class_label_count=[]       
count=0
for i in y:
  if i=="rock":
    count+=1
class_label_count.append(count)
count=0
for i in y:
  if i=="disco":
    count+=1
class_label_count.append(count)
count=0
for i in y:
  if i=="classical":
    count+=1
class_label_count.append(count)
count=0
for i in y:
  if i=="hiphop":
    count+=1
class_label_count.append(count)
count=0
for i in y:
  if i=="pop":
    count+=1
class_label_count.append(count)
count=0
for i in y:
  if i=="jazz":
    count+=1
class_label_count.append(count)
count=0
for i in y:
  if i=="metal":
    count+=1
class_label_count.append(count)
count=0
for i in y:
  if i=="blues":
    count+=1
class_label_count.append(count)
count=0
for i in y:
  if i=="country":
    count+=1
class_label_count.append(count)
count=0
for i in y:
  if i=="reggae":
    count+=1
class_label_count.append(count)

plt.pie(class_label_count,labels=class_label_name)
plt.title("CLASS LABEL DISTRIBUTION")
plt.show()
plt.bar(class_label_name,class_label_count)
plt.title("ANALYSING COUNT OF DIFFERENT CLASS LABELS")
plt.show()


# **Distribution** **of** **features** **over** **class** **labels**
# 
# How are features classified over class labels in the training data set 
# 
# We'll use the scatter plot for this 

# In[ ]:



print("\n**********VISUALISATION OF CLASSIFICATION OF FEATURES OVER CLASS LABELS**********\n")

plt.scatter(x.length,y,color="blue")
plt.title("Feature: length")
plt.show()

plt.scatter(x.chroma_stft_mean,y,color="red")
plt.title("Feature: chroma_stft_mean")
plt.show()

plt.scatter(x.chroma_stft_var,y,color="blue")
plt.title("Feature: chroma_stft_var")
plt.show()

plt.scatter(x.rms_mean,y,color="red")
plt.title("Feature: rms_mean")
plt.show()

plt.scatter(x.rms_var,y,color="blue")
plt.title("Feature: rms_var")
plt.show()

plt.scatter(x.spectral_centroid_mean,y,color="red")
plt.title("Feature: spectral_centroid_mean")
plt.show()

plt.scatter(x.spectral_centroid_var,y,color="blue")
plt.title("Feature: spectral_centroid_var")
plt.show()

plt.scatter(x.spectral_bandwidth_mean,y,color="red")
plt.title("Feature: spectral_bandwidth_mean")
plt.show()

plt.scatter(x.spectral_bandwidth_var,y,color="blue")
plt.title("Feature: spectral_bandwidth_var")
plt.show()

plt.scatter(x.rolloff_mean,y,color="red")
plt.title("Feature: rolloff_mean")
plt.show()

plt.scatter(x.rolloff_var,y,color="blue")
plt.title("Feature: rolloff_var")
plt.show()

plt.scatter(x.zero_crossing_rate_mean,y,color="red")
plt.title("Feature: zero_crossing_rate_mean")
plt.show()

plt.scatter(x.zero_crossing_rate_var,y,color="blue")
plt.title("Feature: zero_crossing_rate_var")
plt.show()

plt.scatter(x.harmony_mean,y,color="red")
plt.title("Feature: harmony_mean")
plt.show()

plt.scatter(x.harmony_var,y,color="blue")
plt.title("Feature: harmony_var")
plt.show()

plt.scatter(x.perceptr_mean,y,color="red")
plt.title("Feature: perceptr_mean")
plt.show()

plt.scatter(x.perceptr_var,y,color="blue")
plt.title("Feature: perceptr_var")
plt.show()

plt.scatter(x.tempo,y,color="red")
plt.title("Feature: tempo")
plt.show()

plt.scatter(x.mfcc1_mean,y,color="blue")
plt.title("Feature: mfcc1_mean")
plt.show()

plt.scatter(x.mfcc1_var,y,color="red")
plt.title("Feature: mfcc1_var")
plt.show()

plt.scatter(x.mfcc2_mean,y,color="blue")
plt.title("Feature: mfcc2_mean")
plt.show()

plt.scatter(x.mfcc2_var,y,color="red")
plt.title("Feature: mfcc2_var")
plt.show()

plt.scatter(x.mfcc3_mean,y,color="blue")
plt.title("Feature: mfcc3_mean")
plt.show()

plt.scatter(x.mfcc3_var,y,color="red")
plt.title("Feature: mfcc3_var")
plt.show()

plt.scatter(x.mfcc4_mean,y,color="blue")
plt.title("Feature: mfcc4_mean")
plt.show()

plt.scatter(x.mfcc4_var,y,color="red")
plt.title("Feature: mfcc4_var")
plt.show()

plt.scatter(x.mfcc5_mean,y,color="blue")
plt.title("Feature: mfcc5_mean")
plt.show()


plt.scatter(x.mfcc5_var,y,color="red")
plt.title("Feature: mfcc5_var")
plt.show()

plt.scatter(x.mfcc6_mean,y,color="blue")
plt.title("Feature: mfcc6_mean")
plt.show()

plt.scatter(x.mfcc6_var,y,color="red")
plt.title("Feature: mfcc6_var")
plt.show()

plt.scatter(x.mfcc7_mean,y,color="blue")
plt.title("Feature: mfcc7_mean")
plt.show()

plt.scatter(x.mfcc7_var,y,color="red")
plt.title("Feature: mfcc7_var")
plt.show()

plt.scatter(x.mfcc8_mean,y,color="blue")
plt.title("Feature: mfcc8_mean")
plt.show()

plt.scatter(x.mfcc8_var,y,color="red")
plt.title("Feature: mfcc8_var")
plt.show()

plt.scatter(x.mfcc9_mean,y,color="blue")
plt.title("Feature: mfcc9_mean")
plt.show()

plt.scatter(x.mfcc9_var,y,color="red")
plt.title("Feature: mfcc9_var")
plt.show()

plt.scatter(x.mfcc10_mean,y,color="blue")
plt.title("Feature: mfcc10_mean")
plt.show()

plt.scatter(x.mfcc10_var,y,color="red")
plt.title("Feature: mfcc10_var")
plt.show()

plt.scatter(x.mfcc11_mean,y,color="blue")
plt.title("Feature: mfcc11_mean")
plt.show()

plt.scatter(x.mfcc11_var,y,color="red")
plt.title("Feature: mfcc11_var")
plt.show()

plt.scatter(x.mfcc12_mean,y,color="blue")
plt.title("Feature: mfcc12_mean")
plt.show()

plt.scatter(x.mfcc12_var,y,color="red")
plt.title("Feature: mfcc12_var")
plt.show()

plt.scatter(x.mfcc13_mean,y,color="blue")
plt.title("Feature: mfcc13_mean")
plt.show()

plt.scatter(x.mfcc13_var,y,color="red")
plt.title("Feature: mfcc13_var")
plt.show()

plt.scatter(x.mfcc14_mean,y,color="blue")
plt.title("Feature: mfcc14_mean")
plt.show()

plt.scatter(x.mfcc14_var,y,color="red")
plt.title("Feature: mfcc14_var")
plt.show()

plt.scatter(x.mfcc15_mean,y,color="blue")
plt.title("Feature: mfcc15_mean")
plt.show()

plt.scatter(x.mfcc15_var,y,color="red")
plt.title("Feature: mfcc15_var")
plt.show()

plt.scatter(x.mfcc16_mean,y,color="blue")
plt.title("Feature: mfcc16_mean")
plt.show()

plt.scatter(x.mfcc16_var,y,color="red")
plt.title("Feature: mfcc16_var")
plt.show()

plt.scatter(x.mfcc17_mean,y,color="blue")
plt.title("Feature: mfcc17_mean")
plt.show()

plt.scatter(x.mfcc17_var,y,color="red")
plt.title("Feature: mfcc17_var")
plt.show()

plt.scatter(x.mfcc18_mean,y,color="blue")
plt.title("Feature: mfcc18_mean")
plt.show()

plt.scatter(x.mfcc18_var,y,color="red")
plt.title("Feature: mfcc18_var")
plt.show()

plt.scatter(x.mfcc19_mean,y,color="blue")
plt.title("Feature: mfcc19_mean")
plt.show()

plt.scatter(x.mfcc19_var,y,color="red")
plt.title("Feature: mfcc19_var")
plt.show()

plt.scatter(x.mfcc20_mean,y,color="blue")
plt.title("Feature: mfcc20_mean")
plt.show()

plt.scatter(x.mfcc20_var,y,color="red")
plt.title("Feature: mfcc20_var")
plt.show()


# # FEATURE SELECTION
# 
# Studying the covariance between the features

# In[ ]:


#CORRELATION BETWEEN THE FEATURES
corr_matrix=x.corr()
#print(corr_matrix.to_string())
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(50,50))
sns.heatmap(corr_matrix,annot=True)
#selection of upper triangular matrix from the covariance matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#Find index of feature columns with correlation greater than 0.85
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
print(to_drop)


# # MODEL BUILDING

# In[4]:


from sklearn.model_selection import train_test_split
#splitting our training data set into 80% training data + 20% test data
#stratify parameter will preserve the proportion of target as in original dataset, in the train and test data subsets as well.
# we want to preserve the dataset proportions for better prediction and reproduceability of results
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.2,random_state = 30, stratify = y)

#Removing the redundant features found using techniques in previous part and splitting our new data set into training and validation set
x1=x.drop(['spectral_bandwidth_mean', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'harmony_var', 'mfcc20_var'],axis=1)
train_x1, test_x1, train_y1, test_y1 = train_test_split(x1,y,test_size = 0.2,random_state = 30, stratify = y)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
train_scaled1_x = scaler.fit_transform(train_x)
test_scaled1_x = scaler.fit_transform(test_x)
print("TRAINING AND VALIDATION DATA SET SCALED USING STANDARD SCALAR\n")
print(train_scaled1_x,"\n")
print(test_scaled1_x,"\n")


from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler()
train_scaled2_x = scaling.fit_transform(train_x)
test_scaled2_x = scaling.fit_transform(test_x)
print("TRAINING AND VALIDATION DATA SET SCALED USING MINMAX SCALAR\n")
print(train_scaled2_x,"\n")
print(test_scaled2_x,"\n")

# We'll use standard scalar in our case so as to preserve the -ve feature values as well

train_scaled_x1 = scaler.fit_transform(train_x1)
test_scaled_x1 = scaler.fit_transform(test_x1)

#Evaluation criteria for data classification 
#to evaluate the perfomance of the classifier using the ground truths of the give data set.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.pipeline import make_pipeline

#importing the classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[ ]:


#(1) Decision Tree Classifier
print("\n*************** DECISION TREE CLASSIFIER *****************\n")
print("\n********** WITHOUT USING HYPERPARAMETER TUNING **********\n")

from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier()
DTC.fit(train_x,train_y)
DTC_predict=DTC.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING **********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,DTC_predict))
print("\nClassification Report:\n", classification_report(test_y,DTC_predict))
print("Accuracy Score:\n", accuracy_score(test_y,DTC_predict))


DTC=DecisionTreeClassifier()
DTC.fit(train_scaled1_x,train_y)
DTC_predict=DTC.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING**********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,DTC_predict))
print("\nClassification Report:\n", classification_report(test_y,DTC_predict))
print("Accuracy Score:\n", accuracy_score(test_y,DTC_predict))


DTC=DecisionTreeClassifier()
DTC.fit(train_x1,train_y1)
DTC_predict=DTC.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING**********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,DTC_predict))
print("\nClassification Report:\n", classification_report(test_y1,DTC_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,DTC_predict))


DTC=DecisionTreeClassifier()
DTC.fit(train_scaled_x1,train_y1)
DTC_predict=DTC.predict(test_scaled_x1)
print("\n********** WITH FEATURE SELECTION AND WITH SCALING**********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,DTC_predict))
print("\nClassification Report:\n", classification_report(test_y1,DTC_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,DTC_predict))

print("\n********** USING HYPERPARAMETER TUNING **********\n")
dec_tree=DecisionTreeClassifier()

criterion = ['gini', 'entropy']
max_depth = [int(i) for i in np.linspace(10,10000,num=100)]
scoring=['accuracy','f1_macro']
pipe = Pipeline(steps=[('dec_tree', dec_tree)])
parameters = dict(dec_tree__criterion=criterion,
                  dec_tree__max_depth=max_depth)

print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING **********\n")
clf_GS = GridSearchCV(pipe, parameters,refit=True,scoring='f1_macro')
clf_GS.fit(train_x,train_y)
print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
print(clf_GS.best_estimator_.get_params()['dec_tree'])
dec_tree_predict=clf_GS.predict(test_x)
print("Confusion Matrix:\n", confusion_matrix(test_y,dec_tree_predict))
print("\nClassification Report:\n", classification_report(test_y,dec_tree_predict))
print("Accuracy Score:\n", accuracy_score(test_y,dec_tree_predict))

print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING **********\n")
clf_GS = GridSearchCV(pipe, parameters,refit=True,scoring='f1_macro')
clf_GS.fit(train_scaled1_x,train_y)
print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
print(clf_GS.best_estimator_.get_params()['dec_tree'])
dec_tree_predict=clf_GS.predict(test_scaled1_x)
print("Confusion Matrix:\n", confusion_matrix(test_y,dec_tree_predict))
print("\nClassification Report:\n", classification_report(test_y,dec_tree_predict))
print("Accuracy Score:\n", accuracy_score(test_y,dec_tree_predict))

print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING **********\n")
clf_GS = GridSearchCV(pipe, parameters,refit=True,scoring='f1_macro')
clf_GS.fit(train_x1,train_y1)
print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
print(clf_GS.best_estimator_.get_params()['dec_tree'])
dec_tree_predict=clf_GS.predict(test_x1)
print("Confusion Matrix:\n", confusion_matrix(test_y1,dec_tree_predict))
print("\nClassification Report:\n", classification_report(test_y1,dec_tree_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,dec_tree_predict))

print("\n********** WITH FEATURE SELECTION AND WITH SCALING **********\n")
clf_GS = GridSearchCV(pipe, parameters,refit=True,scoring='f1_macro')
clf_GS.fit(train_scaled_x1,train_y1)
print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
print(clf_GS.best_estimator_.get_params()['dec_tree'])
dec_tree_predict=clf_GS.predict(test_scaled_x1)
print("Confusion Matrix:\n", confusion_matrix(test_y1,dec_tree_predict))
print("\nClassification Report:\n", classification_report(test_y1,dec_tree_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,dec_tree_predict))



#NOTES: 
#2*100 TOTAL POSSIBLE COMBINATIONS 
#i.e., the no. of iterations in the grid search


# In[ ]:


#(2) Random Forest Classifier
print("\n********** RANDOM FOREST CLASSIFIER ***********\n")
print("\n********** WITHOUT HYPERPARAMETER TUNING  ***********\n")

RFC1 = RandomForestClassifier()
RFC1.fit(train_x, train_y) 
RFC1_predict = RFC1.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,RFC1_predict))
print("\nClassification Report:\n", classification_report(test_y,RFC1_predict))
print("Accuracy Score:\n", accuracy_score(test_y,RFC1_predict))

RFC1 = RandomForestClassifier()
RFC1.fit(train_scaled1_x, train_y) 
RFC1_predict = RFC1.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,RFC1_predict))
print("\nClassification Report:\n", classification_report(test_y,RFC1_predict))
print("Accuracy Score:\n", accuracy_score(test_y,RFC1_predict))

RFC1 = RandomForestClassifier()
RFC1.fit(train_x1, train_y1) 
RFC1_predict = RFC1.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,RFC1_predict))
print("\nClassification Report:\n", classification_report(test_y1,RFC1_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,RFC1_predict))

RFC1 = RandomForestClassifier()
RFC1.fit(train_scaled_x1, train_y1) 
RFC1_predict = RFC1.predict(test_scaled_x1)
print("\n********** WITH FEATURE SELECTION AND  WITH SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,RFC1_predict))
print("\nClassification Report:\n", classification_report(test_y1,RFC1_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,RFC1_predict))


print("\n********** WITH HYPERPARAMETER TUNING ***********\n")
RFC = RandomForestClassifier()
#pprint(RFC.get_params())

#Number of trees in random forest
n_estimators = [int(n) for n in range(10,300,10)]

# Number of features to consider at every split
max_features = ['log2', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(n) for n in range(2, 30,1)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [int(n) for n in range(2,50,1)]

# Minimum number of samples required at each leaf node
min_samples_leaf = [int(n) for n in range(1,30,1)]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
#pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
# Random search of parameters, using 5 fold cross validation, 
# search across 300 different combinations, and use all available cores 

rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(), scoring='f1_macro', param_distributions = random_grid,n_iter=60,cv=5,verbose=3,n_jobs=4)
# Fit the random search model
rf_random.fit(train_x, train_y)
print(rf_random.best_params_)

RFC2_predict =rf_random.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUR SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,RFC2_predict))
print("\nClassification Report:\n", classification_report(test_y,RFC2_predict))
print("Accuracy Score:\n", accuracy_score(test_y,RFC2_predict))

rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(),scoring='f1_macro', param_distributions = random_grid,n_iter=60,cv = 5,verbose=3,n_jobs=4)
# Fit the random search model
rf_random.fit(train_scaled1_x, train_y)
print(rf_random.best_params_)

RFC2_predict =rf_random.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,RFC2_predict))
print("\nClassification Report:\n", classification_report(test_y,RFC2_predict))
print("Accuracy Score:\n", accuracy_score(test_y,RFC2_predict))

rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(),scoring='f1_macro', param_distributions = random_grid,n_iter=60,cv = 5,verbose=3,n_jobs=4)
# Fit the random search model
rf_random.fit(train_x1, train_y1)
print(rf_random.best_params_)

RFC2_predict =rf_random.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,RFC2_predict))
print("\nClassification Report:\n", classification_report(test_y1,RFC2_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,RFC2_predict))

rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(),scoring='f1_macro', param_distributions = random_grid,n_iter=60,cv = 5,verbose=3,n_jobs=4)
# Fit the random search model
rf_random.fit(train_scaled_x1, train_y1)
print(rf_random.best_params_)

RFC2_predict =rf_random.predict(test_scaled_x1)
print("\n********** WITH FEATURE SELECTION AND WITH SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,RFC2_predict))
print("\nClassification Report:\n", classification_report(test_y1,RFC2_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,RFC2_predict))


# In[5]:


#3) Support vector machine
from sklearn.svm import SVC
print("\n********** SUPPORT VECTOR MACHINE  ***********\n")
print("\n********** WITHOUT USING HYPERPARAMETER TUNING  ***********\n")

'''SVM = SVC()
SVM.fit(train_x, train_y)
SVM_predict = SVM.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y,SVM_predict ))
print('\nClassification Report:\n', classification_report(test_y, SVM_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y, SVM_predict ))
'''

SVM = SVC()
SVM.fit(train_scaled1_x, train_y)
SVM_predict = SVM.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y,SVM_predict ))
print('\nClassification Report:\n', classification_report(test_y, SVM_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y, SVM_predict ))

'''
SVM = SVC()
SVM.fit(train_x1, train_y1)
SVM_predict = SVM.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y1,SVM_predict ))
print('\nClassification Report:\n', classification_report(test_y1, SVM_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y1, SVM_predict ))
'''

SVM = SVC()
SVM.fit(train_scaled_x1, train_y1)
SVM_predict = SVM.predict(test_scaled_x1)
print("\n********** WITH FEATURE SELECTION AND WITH SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y1,SVM_predict ))
print('\nClassification Report:\n', classification_report(test_y1, SVM_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y1, SVM_predict ))

param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [10,1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','linear','sigmoid','poly']} 

print("\n********** USING HYPERPARAMETER TUNING  ***********\n")
'''svm = RandomizedSearchCV(SVC(), param_distributions=param_grid,scoring='f1_macro',n_iter=12,cv=5)
svm.fit(train_x, train_y)
print(svm.best_params_)
svm_predict = svm.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y,svm_predict ))
print('\nClassification Report:\n', classification_report(test_y, svm_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y, svm_predict ))
'''





svm = GridSearchCV(SVC(), param_grid, refit = True,scoring='f1_macro', verbose = -1)
svm.fit(train_scaled1_x, train_y)
print(svm.best_params_)
svm_predict = svm.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y,svm_predict ))
print('\nClassification Report:\n', classification_report(test_y, svm_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y, svm_predict ))

'''
svm = GridSearchCV(SVC(), param_grid, refit = True,scoring='f1_macro', verbose = -1)
svm.fit(train_x1, train_y1)
print(svm.best_params_)
svm_predict = svm.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y1,svm_predict ))
print('\nClassification Report:\n', classification_report(test_y1, svm_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y1, svm_predict ))
'''


svm = GridSearchCV(SVC(), param_grid, refit = True, verbose = -1,scoring='f1_macro')
svm.fit(train_scaled_x1, train_y1)
print(svm.best_params_)
svm_predict = svm.predict(test_scaled_x1)
print("\n********** WITH FEATURE SELECTION AND WITH SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y1,svm_predict ))
print('\nClassification Report:\n', classification_report(test_y1, svm_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y1, svm_predict ))


# In[ ]:



#(4) Logistic Regression
print("\n********** MULTINOMIAL LOGISTIC REGRESSION CLASSIFIER ***********\n")
from sklearn.linear_model import LogisticRegression
print("\n********** WITHOUT USING HYPERPARAMETER TUNING ***********\n")


LR = LogisticRegression(max_iter=1000,solver='liblinear')
LR.fit(train_x, train_y) 
LR_predict = LR.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,LR_predict))
print("\nClassification Report:\n", classification_report(test_y,LR_predict))
print("Accuracy Score:\n", accuracy_score(test_y,LR_predict))

LR = LogisticRegression(max_iter=100,solver='liblinear')
LR.fit(train_scaled1_x, train_y) 
LR_predict = LR.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,LR_predict))
print("\nClassification Report:\n", classification_report(test_y,LR_predict))
print("Accuracy Score:\n", accuracy_score(test_y,LR_predict))

LR = LogisticRegression(max_iter=1000,solver='liblinear')
LR.fit(train_x1, train_y1) 
LR_predict = LR.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,LR_predict))
print("\nClassification Report:\n", classification_report(test_y1,LR_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,LR_predict))

LR = LogisticRegression(max_iter=100,solver='liblinear')
LR.fit(train_scaled_x1, train_y1) 
LR_predict = LR.predict(test_scaled_x1)
print("\n********** WITH FEATURE SELECTION AND WITH SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,LR_predict))
print("\nClassification Report:\n", classification_report(test_y1,LR_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,LR_predict))




print("\n********** USING HYPERPARAMETER TUNING  ***********\n")
LR1 = LogisticRegression(max_iter=10000)
LR2 = LogisticRegression(max_iter=100)
param_grid = {'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
              'penalty': ['l2'],
              'C' : [100, 10, 1.0, 0.1, 0.01]}

param_grid2 = {'solver' : ['lbfgs', 'liblinear'],
              'penalty': ['l2'],
              'C' : [100, 10, 1.0, 0.1, 0.01]}


lr = GridSearchCV(estimator=LR1, param_grid = param_grid2, n_jobs=4, scoring='f1_macro',error_score=0)
grid_result = lr.fit(train_x, train_y)
# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):

print(lr.best_params_)
lr_predict = lr.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y,lr_predict ))
print('\nClassification Report:\n', classification_report(test_y, lr_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y, lr_predict ))


lr = GridSearchCV(estimator=LR2, param_grid = param_grid, n_jobs=4, scoring='f1_macro',error_score=0)
grid_result = lr.fit(train_scaled1_x, train_y)
# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):

print(lr.best_params_)
lr_predict = lr.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y,lr_predict ))
print('\nClassification Report:\n', classification_report(test_y, lr_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y, lr_predict ))


lr = GridSearchCV(estimator=LR1, param_grid = param_grid2, n_jobs=4, scoring='f1_macro',error_score=0)
grid_result = lr.fit(train_x1, train_y1)
# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):

print(lr.best_params_)
lr_predict = lr.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y1,lr_predict ))
print('\nClassification Report:\n', classification_report(test_y1, lr_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y1, lr_predict ))


lr = GridSearchCV(estimator=LR2, param_grid = param_grid, n_jobs=4, scoring='f1_macro',error_score=0)
grid_result = lr.fit(train_scaled_x1, train_y1)
# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):

print(lr.best_params_)
lr_predict = lr.predict(test_scaled_x1)
print("\n********** WITH FEATURE SELECTION AND WITH SCALING  ***********\n")
print('\nConfusion Matrix:\n', confusion_matrix(test_y1,lr_predict ))
print('\nClassification Report:\n', classification_report(test_y1, lr_predict ))
print('\nAccuracy Score:\n', accuracy_score(test_y1, lr_predict ))



# In[ ]:


#(5) KNN
print("\n********** K-NEAREST NEIGHBOUR CLASSIFIER***********\n")
print("\n********** WITHOUT USING HYPERPARAMETER TUNING ***********\n")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
KNN= KNeighborsClassifier()
KNN.fit(train_x,train_y)
KNN_predict= KNN.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,KNN_predict))
print("\nClassification Report:\n", classification_report(test_y,KNN_predict))
print("Accuracy Score:\n", accuracy_score(test_y,KNN_predict))

KNN= KNeighborsClassifier()
KNN.fit(train_scaled1_x,train_y)
KNN_predict= KNN.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,KNN_predict))
print("\nClassification Report:\n", classification_report(test_y,KNN_predict))
print("Accuracy Score:\n", accuracy_score(test_y,KNN_predict))

KNN= KNeighborsClassifier()
KNN.fit(train_x1,train_y1)
KNN_predict= KNN.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,KNN_predict))
print("\nClassification Report:\n", classification_report(test_y1,KNN_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,KNN_predict))


KNN= KNeighborsClassifier()
KNN.fit(train_scaled_x1,train_y1)
KNN_predict= KNN.predict(test_scaled_x1)
print("\n********** WITH FEATURE SELECTION AND WITH SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,KNN_predict))
print("\nClassification Report:\n", classification_report(test_y1,KNN_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,KNN_predict))

n_neighbors = list(range(1,40))
metric=['euclidean','manhattan','minkowski']
weights=['uniform','distance']

grid=dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2)

print("\n********** USING HYPERPARAMETER TUNING ***********\n")


knn=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=grid,cv=cv,scoring='f1_macro',error_score=0)
knn.fit(train_x,train_y)
print(knn.best_params_)
knn_predict=knn.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,knn_predict))
print("\nClassification Report:\n", classification_report(test_y,knn_predict))
print("Accuracy Score:\n", accuracy_score(test_y,knn_predict))

knn=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=grid,cv=cv,scoring='f1_macro',error_score=0)
knn.fit(train_scaled1_x,train_y)
print(knn.best_params_)
knn_predict=knn.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,knn_predict))
print("\nClassification Report:\n", classification_report(test_y,knn_predict))
print("Accuracy Score:\n", accuracy_score(test_y,knn_predict))

knn=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=grid,cv=cv,scoring='f1_macro',error_score=0)
knn.fit(train_x1,train_y1)
print(knn.best_params_)
knn_predict=knn.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,knn_predict))
print("\nClassification Report:\n", classification_report(test_y1,knn_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,knn_predict))

knn=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=grid,cv=cv,scoring='f1_macro',error_score=0)
knn.fit(train_scaled_x1,train_y1)
print(knn.best_params_)
knn_predict=knn.predict(test_scaled_x1)
print("\n**********WITH FEATURE SELECTION AND WITH SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,knn_predict))
print("\nClassification Report:\n", classification_report(test_y1,knn_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,knn_predict))


# In[ ]:


#(6) AdaBoost Classifier
print("\n********** ADA BOOST CLASSIFIER  ***********\n")
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier
 

random_grid={'n_estimators': [int(n) for n in range(10,100,10)],
              'learning_rate': [0.001,0.01,0.1,0.3,0.6,0.8,1]
}


print("\n********** Using RFC As Base Estimator ***********\n")
rfc=RandomForestClassifier()
ABC=AdaBoostClassifier(estimator=rfc)
ABC.fit(train_x,train_y)
ABC_predict=ABC.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,ABC_predict))
print("\nClassification Report:\n", classification_report(test_y,ABC_predict))
print("Accuracy Score:\n", accuracy_score(test_y,ABC_predict))

ABC=AdaBoostClassifier(estimator=rfc)
ABC.fit(train_scaled1_x,train_y)
ABC_predict=ABC.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,ABC_predict))
print("\nClassification Report:\n", classification_report(test_y,ABC_predict))
print("Accuracy Score:\n", accuracy_score(test_y,ABC_predict))

ABC=AdaBoostClassifier(estimator=rfc)
ABC.fit(train_x1,train_y1)
ABC_predict=ABC.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,ABC_predict))
print("\nClassification Report:\n", classification_report(test_y1,ABC_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,ABC_predict))

ABC=AdaBoostClassifier(estimator=rfc)
ABC.fit(train_scaled_x1,train_y1)
ABC_predict=ABC.predict(test_scaled_x1)
print("\n********** WITH FEATURE SELECTION AND WITH SCALING ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,ABC_predict))
print("\nClassification Report:\n", classification_report(test_y1,ABC_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,ABC_predict))



print("\n********** USING HYPERPARAMETER TUNING  ***********\n")
abc_random=RandomizedSearchCV(estimator=ABC,param_distributions=random_grid,n_iter=2,cv=5,scoring='f1_macro')
abc_random.fit(train_x, train_y)
print(abc_random.best_params_)
abc_predict =abc_random.predict(test_x)
print("\n********** WITHOUT FEATURE SELECTION AND WITHOUT SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,abc_predict))
print("\nClassification Report:\n", classification_report(test_y,abc_predict))
print("Accuracy Score:\n", accuracy_score(test_y,abc_predict))

abc_random=RandomizedSearchCV(estimator=ABC,param_distributions=random_grid,n_iter=2,cv=5,scoring='f1_macro')
abc_random.fit(train_scaled1_x, train_y)
print(abc_random.best_params_)
abc_predict =abc_random.predict(test_scaled1_x)
print("\n********** WITHOUT FEATURE SELECTION BUT WITH SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y,abc_predict))
print("\nClassification Report:\n", classification_report(test_y,abc_predict))
print("Accuracy Score:\n", accuracy_score(test_y,abc_predict))

abc_random=RandomizedSearchCV(estimator=ABC,param_distributions=random_grid,n_iter=2,cv=5,scoring='f1_macro')
abc_random.fit(train_x1, train_y1)
print(abc_random.best_params_)
abc_predict =abc_random.predict(test_x1)
print("\n********** WITH FEATURE SELECTION BUT WITHOUT SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,abc_predict))
print("\nClassification Report:\n", classification_report(test_y1,abc_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,abc_predict))

abc_random=RandomizedSearchCV(estimator=ABC,param_distributions=random_grid,n_iter=2,cv=5,scoring='f1_macro')
abc_random.fit(train_scaled_x1, train_y1)
print(abc_random.best_params_)
abc_predict =abc_random.predict(test_scaled_x1)
print("\n********** WITH FEATURE SELECTION AND WITH SCALING  ***********\n")
print("Confusion Matrix:\n", confusion_matrix(test_y1,abc_predict))
print("\nClassification Report:\n", classification_report(test_y1,abc_predict))
print("Accuracy Score:\n", accuracy_score(test_y1,abc_predict))


# In[ ]:


##Running on test data set

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
test_X = scaler.fit_transform(X)
rfc=RandomForestClassifier()
ABC=AdaBoostClassifier(estimator=rfc,n_estimators=10,learning_rate=0.3)
ABC.fit(train_scaled1_x,train_y)
ABC_predict=ABC.predict(test_X)
print("***PREDICTED CLASSES LABELS***")
print(ABC_predict)

