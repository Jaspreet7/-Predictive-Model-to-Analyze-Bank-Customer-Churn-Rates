#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


#load data
df=pd.read_excel('train_Capstone.xlsx')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


#Drop Feature Id
df.drop(['id','CustomerId','Surname'],axis=1,inplace=True)


# In[6]:


df.info()


# In[7]:


#Change the data type of Age feature
df['Age']=df['Age'].astype(int)


# In[8]:


df.shape[0]


# In[9]:


len(df[df.duplicated()])


# In[10]:


duplicate_rows = df[df.duplicated()]


# In[11]:


df=df.drop(duplicate_rows.index)


# In[12]:


df.shape


# In[13]:


#Creating a copy of data frame 
df_org=df.copy()


# In[14]:


df_org.head()


# In[15]:


df['Gender']=df['Gender'].map({'Female':0,'Male':1})


# In[16]:


df['Geography'].unique()


# In[17]:


df['Geography']=df['Geography'].map({'France':0,'Spain':1,'Germany':2})


# In[18]:


df.isnull().values.any()


# In[19]:


#percentage of customer churn
churned=df['Exited'].value_counts()[1]/df.shape[0]*100
print('Churned rate is {:.2f}%'.format(churned))


# In[20]:


#Show inbalance dataset
df['Exited'].value_counts()


# In[21]:


# Compare different class in the dataset
classes = df['Exited'].value_counts()
classes.plot( kind='bar')
plt.title("Churned class histogram")
plt.xlabel("Exited")
plt.ylabel("Frequency")


# In[22]:


#Correlation
fig = plt.figure(figsize=(15,5))
sns.heatmap(df.corr(),annot=True,cmap="Blues")


# In[23]:


df['Age'].hist(bins=40)
plt.title('Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[25]:


# Compare different classes in the dataset
for feature in df.columns:
    df[feature].hist(bins=25)
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()


# # Box plot to find the outliers in data

# In[26]:


plt.boxplot(df['Age'])
plt.show()


# In[27]:


plt.boxplot(df['CreditScore'])
plt.show()


# In[28]:


plt.boxplot(df['Balance'])
plt.show()


# In[29]:


plt.boxplot(df['EstimatedSalary'])
plt.show()


# # Distribution of data

# In[30]:


sns.kdeplot(data=df,x='EstimatedSalary',fill=True)
plt.xlim(df['EstimatedSalary'].min())


# In[31]:


sns.kdeplot(data=df,x='Age',fill=True)
#plt.xlim(df['EstimatedSalary'].min())


# In[32]:


sns.kdeplot(data=df,x='CreditScore',fill=True)


# In[33]:


sns.kdeplot(data=df,x='Balance',fill=True)


# In[34]:


# List of numeric variables
numeric_variables = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# Set up subplots
fig, axes = plt.subplots(nrows=len(numeric_variables), ncols=1, figsize=(10, 16))

# Color palette for 'Exited' category (0: Not exited, 1: Exited)
colors = {0: 'blue', 1: 'red'}

# Plot histograms for each numeric variable
for i, var in enumerate(numeric_variables):
    sns.histplot(df, x=var, hue='Exited', ax=axes[i], kde=True, bins=30, palette=colors)
    axes[i].set_title(f'Distribution of {var}', fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].legend(title='Exited', labels=['Exited', 'Not_Exited'])

plt.tight_layout()
plt.show()


# In[35]:


#No of customer churned in age between 48 to 68
df[(df['Age']>47) & (df['Age']<59)&(df['Exited']==1)].shape


# In[36]:


#No of customer retained in age between 48 to 68
df[(df['Age']>47) & (df['Age']<59)&(df['Exited']==0)].shape


# In[37]:


#No of customer in age between 48 to 68
df_age=df[(df['Age']>47) & (df['Age']<59)]


# In[38]:


#Correlation among feature of people age between 48 to 58
fig = plt.figure(figsize=(15,5))
sns.heatmap(df_age.corr(),annot=True,cmap="Blues")


# In[39]:


fig=plt.figure(figsize=(10,4))
sns.countplot(data=df_org,x='Gender',hue='Geography')
plt.show()


# In[40]:


plt.figure(figsize=(10, 6))

# Countplot with stacked bars for Gender based on Geography
sns.countplot(data=df, x='Geography', hue='Gender', palette='Set1')

# Adding title and labels
plt.title('Distribution of Gender based on Geography', fontsize=14)
plt.xlabel('Geography', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1, 2], ['France', 'Germany', 'Spain'])

# Adding legend
plt.legend(title='Gender', labels=['Female', 'Male'])

# Show plot
plt.show()


# # Outliers with the help of Z-score

# In[41]:


outliers=[]
def det_otl(data):
    threshold=3
    mean=data.mean()
    std=data.std()
    for i in data:
        z_score=(i-mean)/std
        if np.abs(z_score)>threshold:
            outliers.append(i)
    return outliers


# In[42]:


outl=det_otl(df['Age'])


# In[43]:


min(outl)


# In[44]:


len(outl)


# In[45]:


#Code to fing Q1 and Q3
qun1,qun3=np.percentile(df['Age'],[25,75])


# In[46]:


qun1


# In[47]:


qun3


# In[48]:


df['Age'].mean()


# # Outlier by using IQR

# In[49]:


iqr_value=qun3-qun1
print(iqr_value)


# In[50]:


#Find lower and upper bound
lower_bound=qun1-(1.5*iqr_value)
upper_bound=qun3+(1.5*iqr_value)


# In[51]:


lower_bound


# In[52]:


upper_bound


# In[53]:


#No. of outliers by using IQR
df[df['Age']>57].shape


# In[54]:


#No. of outliers by using Z-score
df[df['Age']>65].shape


# # One Hot-encoding

# In[55]:


#Finding Categorical features in dataframe
crt_ft=[ft for ft in df_org.columns if df_org[ft].dtypes=='O']


# In[56]:


crt_ft


# In[57]:


for ft in crt_ft:
    # Perform one-hot encoding
    one_hot_encoded = pd.get_dummies(df_org[ft], prefix=ft)
    one_hot_encoded = one_hot_encoded.astype(int)

    # Concatenate one-hot encoded dataframe with original dataframe
    df_org = pd.concat([df_org, one_hot_encoded], axis=1)
    


# In[58]:


df_org.head()


# In[59]:


df_org.drop(['Geography','Gender'],axis=1,inplace=True)


# In[71]:


df_org.isnull().values.any()


# In[60]:


#Define x and y variables
x = df_org.drop('Exited',axis=1).to_numpy()
y = df_org['Exited'].to_numpy()

#Create Train and Test Datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=100)

#Fix the imbalanced Classes
from imblearn.over_sampling import SMOTE
smt=SMOTE(random_state=100)
x_train_smt,y_train_smt = smt.fit_resample(x_train,y_train)


# In[61]:


#Scale the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train_smt)
x_test2 = sc.transform(x_test)


# In[62]:


#Class Balance - Test Data
print('Train Data - Class Split')
num_zeros = (y_train_smt == 0).sum()
num_ones = (y_train_smt == 1).sum()
print('Class 0 -',  num_zeros)
print('Class 1 -',  num_ones)


# In[63]:


#Logistic Base Model
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.linear_model import LogisticRegression

method = LogisticRegression(solver='lbfgs',
                            class_weight='balanced',
                            random_state=100)

method.fit(x_train2, y_train_smt)
predict = method.predict(x_test2)
print('\nEstimator: Logistic Regression') 
print(confusion_matrix(y_test, predict))  
print(classification_report(y_test, predict))


# In[64]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.linear_model import LogisticRegression

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'penalty': ['l1', 'l2']  # Regularization penalty
}

# Initialize Logistic Regression
method = LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=100)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=method, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform GridSearchCV
grid_search.fit(x_train2, y_train_smt)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the model with the best parameters
best_method = LogisticRegression(**best_params, solver='lbfgs', class_weight='balanced', random_state=100)
best_method.fit(x_train2, y_train_smt)

# Make predictions
predict = best_method.predict(x_test2)

# Print confusion matrix and classification report
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predict))  
print('\nClassification Report:')
print(classification_report(y_test, predict))


# In[65]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=100)

# Train the base model
rf_model.fit(x_train2, y_train_smt)

# Make predictions
predict_base = rf_model.predict(x_test2)

# Print confusion matrix and classification report for the base model
print("Base Random Forest Model:")
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predict_base))  
print('\nClassification Report:')
print(classification_report(y_test, predict_base))


# In[ ]:


# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt'],  # Number of features to consider when looking for the best split
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform GridSearchCV
grid_search.fit(x_train2, y_train_smt)

# Get the best parameters
best_params = grid_search.best_params_
print("\nBest Parameters:", best_params)

# Train the model with the best parameters
best_rf_model = RandomForestClassifier(**best_params, random_state=100)
best_rf_model.fit(x_train2, y_train_smt)

# Make predictions using the optimized model
predict_optimized = best_rf_model.predict(x_test2)

# Print confusion matrix and classification report for the optimized model
print("\nOptimized Random Forest Model:")
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predict_optimized))  
print('\nClassification Report:')
print(classification_report(y_test, predict_optimized))


# In[68]:


# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50],  # Number of trees in the forest
    'max_depth': [None, 10],  # Maximum depth of the tree
    'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt'],  # Number of features to consider when looking for the best split
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform GridSearchCV
grid_search.fit(x_train2, y_train_smt)

# Get the best parameters
best_params = grid_search.best_params_
print("\nBest Parameters:", best_params)

# Train the model with the best parameters
best_rf_model = RandomForestClassifier(**best_params, random_state=100)
best_rf_model.fit(x_train2, y_train_smt)

# Make predictions using the optimized model
predict_optimized = best_rf_model.predict(x_test2)

# Print confusion matrix and classification report for the optimized model
print("\nOptimized Random Forest Model:")
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predict_optimized))  
print('\nClassification Report:')
print(classification_report(y_test, predict_optimized))


# In[ ]:





# In[ ]:




