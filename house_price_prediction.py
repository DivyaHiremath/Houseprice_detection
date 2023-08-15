# -*- coding: utf-8 -*-

from IPython.display import Image
Image(url="https://miro.medium.com/max/503/1*2foyXif7hwkO8wWB5T9KtQ.png",width = 800, height = 300,)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statistics as st

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression


import statsmodels.api as sn
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

"""# Reading the Data"""

df_house = pd.read_csv('/content/train.csv')
df_house.head()

"""# Preliminary Investigation"""

df_house.shape

"""We see the dataframe has 81 columns and 1460 observations."""

df_house.info()

"""The above observation gives the count of Non-Null values and their respective Datatypes of each variable"""

df_house['GarageYrBlt'] = df_house['GarageYrBlt'].astype('object')
df_house['YearBuilt'] = df_house['YearBuilt'].astype('str')
df_house['YearRemodAdd'] = df_house['YearRemodAdd'].astype('str')
df_house['MoSold'] = df_house['MoSold'].astype('str')
df_house['YrSold'] = df_house['YrSold'].astype('str')

"""The above columns consists of years, which must be Categorical for our Analysis, hence we convert these variables from Integer to String."""

df_house.drop('Id', axis=1, inplace=True)
df_house_copy = df_house.copy()

"""Data has a variety of data types. The main types stored in pandas dataframes are object, float, int64, bool and datetime64. In order to learn about each attribute, it is always good for us to know the data type of each column."""

df_house_copy.describe()

def missing_values(df):
  missing_val_count_by_column = (df.isnull().sum()/df.shape[0])*100
  return missing_val_count_by_column[missing_val_count_by_column > 0]

missing_values(df_house_copy)



plt.figure(figsize=(20, 5))
sns.heatmap(df_house_copy.isnull(), cbar=False)
plt.show()


df_house_copy.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1, inplace=True)
missing_values(df_house_copy)


missing_value_df = missing_values(df_house_copy)

for col in missing_value_df.index:
  if df_house_copy[col].dtype == 'float64':
    df_house_copy[col].fillna(df_house_copy[col].mean(), inplace=True)
  if df_house_copy[col].dtype == 'object':
    df_house_copy[col].fillna(st.mode(df_house_copy[col]), inplace=True)

missing_values(df_house_copy)

cat_cols = [df_house_copy.columns[i] for i in range(0, df_house_copy.shape[1])  if df_house_copy.iloc[:,i].dtype=='O']

encoded = df_house_copy.loc[:,cat_cols]

label_encoder = LabelEncoder()
for col in encoded:
    encoded[col] = label_encoder.fit_transform(encoded[col])

df_house_copy.drop(cat_cols, axis=1, inplace=True)
pd.concat([df_house_copy, encoded])
df_house_copy.head()


def data_split(df):
  X = df.drop('SalePrice', axis=1)
  y = df['SalePrice']

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size = 0.2)

  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = data_split(df_house_copy)



linreg = LinearRegression()
MLR_model = linreg.fit(X_train, y_train)
MLR_model.score(X_train, y_train)

train_pred = MLR_model.predict(X_train)
mse = mean_squared_error(y_train, train_pred)
round(np.sqrt(mse), 4)

"""<b>Inference :</b> The Base Model is built on Regression Analysis without handling Outliers, Scaling and Normalization. From above, the train dataset gives an accuracy of <b>80.11%</b> """

linreg = LinearRegression()
MLR_model = linreg.fit(X_train, y_train)
MLR_model.score(X_test, y_test)

train_pred = MLR_model.predict(X_train)
mse = mean_squared_error(y_train, train_pred)
round(np.sqrt(mse), 4)

test_pred = MLR_model.predict(X_test)
mse = mean_squared_error(y_test, test_pred)
round(np.sqrt(mse), 4)

"""<b>Inference :</b> Similarly we are building a Regression Base Model without handling anything and the accuracy in Test is <b>81.88%</b>."""

X_train = sn.add_constant(X_train)
X_test = sn.add_constant(X_test)

X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values

mlr_model = sn.OLS(y_train, X_train).fit()
print(mlr_model.summary())


cat_cols = [df_house.columns[i] for i in range(0, df_house.shape[1])  if df_house.iloc[:,i].dtype=='O']

"""Let us fetch the Categorical variables from the dataset"""

fig, axes = plt.subplots(22, 2, figsize=(22,50))
axes = [ax for axes_rows in axes for ax in axes_rows]

plot_cat_cols = cat_cols.copy()
plot_cat_cols.remove('Neighborhood')
plot_cat_cols.remove('YearBuilt')
plot_cat_cols.remove('YearRemodAdd')
plot_cat_cols.remove('GarageYrBlt')

for i, c in enumerate(df_house[plot_cat_cols]):
    df_house[c].value_counts()[::-1].plot(kind='pie',
                                          ax=axes[i],
                                          title=c,
                                          autopct='%.0f%%',
                                          fontsize=10)
    axes[i].set_ylabel('')

fig, axes = plt.subplots(22, 2, figsize=(18,80))
axes = [ax for axes_rows in axes for ax in axes_rows]

for i, c in enumerate(df_house[plot_cat_cols]):
    df_house[c].value_counts()[::-1].plot(kind='barh',
                                          ax=axes[i],
                                          title=c,
                                          fontsize=12)

"""## Univariate analysis of Numerical features"""

num_cols = [c for c in df_house.columns if c not in cat_cols]

fig, axes = plt.subplots(16, 2, figsize=(20,80))
y =0
for i,c in enumerate(df_house[num_cols]):
    pd.DataFrame(df_house[[c]]).boxplot(ax=axes.flatten()[i], vert=False)

fig, axes = plt.subplots(16, 2, figsize=(20,80))
y =0
for i,c in enumerate(df_house[num_cols]):
    pd.DataFrame(df_house[[c]]).plot(kind='kde', ax=axes.flatten()[i])

"""Above are the plots on Numercial columns to check which columns are Normally DIstributed

## Imputing missing values in numerical variables
"""

missing_values(df_house[num_cols])

sns.boxplot(x = df_house['LotFrontage'])

df_house['LotFrontage'].fillna(df_house['LotFrontage'].median(), inplace=True)

"""<b>Inference :</b> From above analysis, LotFrontage has many outliers present towards the rightend of the Whiskers. Hence we impute NULL Values with its median."""

sns.boxplot(x = df_house['MasVnrArea'])

df_house['MasVnrArea'].fillna(df_house['MasVnrArea'].median(), inplace=True)

"""<b>Inference :</b> From above analysis, MasVnrArea has huge amount of Outliers, hence we impute NULL Values with Median."""

df_house[num_cols].isnull().sum()

missing_cat = missing_values(df_house[cat_cols])
missing_cat

for i in missing_cat[missing_cat>0].index:
    if i == 'MasVnrType':
        df_house['MasVnrType'].fillna(st.mode(df_house['MasVnrType']), inplace=True)
    if i == 'GarageYrBlt':
        df_house['GarageYrBlt'].fillna(st.mode(df_house['GarageYrBlt']), inplace=True)
    else:
        df_house[i] = df_house[i].replace(np.nan, 'Na')
df_house['GarageYrBlt'] = df_house['GarageYrBlt'].astype('str')



df_house[cat_cols].isnull().sum()



object_nunique = list(map(lambda col: df_house[col].nunique(), cat_cols))
d = dict(zip(cat_cols, object_nunique))

sorted(d.items(), key=lambda x: x[1])

"""The above lists explains the number of Unique values in each variables."""

low_cardinality_cols = [col for col in cat_cols if df_house[col].nunique() < 10]
high_cardinality_cols = list(set(cat_cols)-set(low_cardinality_cols))

"""## One-Hot encoding low cardinality categorical variables"""

OH_low_cardinal = pd.DataFrame()

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

for col in low_cardinality_cols:
    columns=[]
    for val in df_house[col].unique():
        name = ''+col+'_'+val
        columns.append(name)
    temp_df = pd.DataFrame(OH_encoder.fit_transform(df_house[[col]]), columns = columns)
    OH_low_cardinal = pd.concat([OH_low_cardinal, temp_df], axis=1)

OH_low_cardinal

OH_high_cardinal = df_house.loc[:,high_cardinality_cols]

label_encoder = LabelEncoder()
for col in high_cardinality_cols:
    OH_high_cardinal[col] = label_encoder.fit_transform(OH_high_cardinal[col])

df_house.drop(cat_cols, axis=1, inplace=True)
df_house.head()

"""The above datset is Label Encoded with <b>High Cardinality</b> by replacing those NULL Values which are greater than 10 Unique values."""

# merging one hot encoded, label encoded and numerical data
df_house = pd.concat([df_house, OH_low_cardinal], axis=1)
df_house = pd.concat([df_house, OH_high_cardinal], axis=1)

print(df_house.shape)
df_house.head()

min_max = MinMaxScaler()

new_num_col = num_cols.copy()
new_num_col.remove('SalePrice')

for col in new_num_col:
    df_house[[col]] = min_max.fit_transform(df_house[[col]])

"""### Train-test split"""

X_train, X_test, y_train, y_test = data_split(df_house)
print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)

"""### Standardization"""

def standardize(X_train, X_test):
  scaler = StandardScaler()

  X_train_scalar = scaler.fit_transform(X_train)
  X_train = pd.DataFrame(X_train_scalar, columns = X_train.columns)

  X_test_scalar = scaler.fit_transform(X_test)
  X_test = pd.DataFrame(X_test_scalar, columns = X_test.columns)

  return X_train, X_test

X_train, X_test = standardize(X_train, X_test)

X_train


linreg = LinearRegression()

linreg_forward = sfs(estimator=linreg, k_features = 100, forward=True,
                     verbose=2, scoring='r2')
sfs_forward = linreg_forward.fit(X_train, y_train)


linreg = LinearRegression()

linreg_forward = sfs(estimator=linreg, k_features = 51, forward=True,
                     verbose=2, scoring='r2')
sfs_forward = linreg_forward.fit(X_train, y_train)

sfs_forward.k_feature_names_

X_train, X_test, y_train, y_test = data_split(df_house)

X_train = X_train[list(sfs_forward.k_feature_names_)]
X_test = X_test[list(sfs_forward.k_feature_names_)]

X_train, X_test = standardize(X_train, X_test)

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)

X_train

linreg = LinearRegression()
MLR_model = linreg.fit(X_train, y_train)
MLR_model.score(X_train, y_train)

"""<b>Inference :</b> Implementing the Linear Regression Model on Train set, the accuracy score is 88.05%."""

linreg = LinearRegression()
MLR_model = linreg.fit(X_train, y_train)
MLR_model.score(X_test, y_test)

"""<b>Inference :</b> Implementing the Regression Model on Test set, the accuracy score is 87.5%"""

train_pred = MLR_model.predict(X_train)
mse = mean_squared_error(y_train, train_pred)
print('RMSE for train data',round(np.sqrt(mse), 4))

test_pred = MLR_model.predict(X_test)
mse = mean_squared_error(y_test, test_pred)
print('RMSE for test data',round(np.sqrt(mse), 4))



"""<b>Inference :</b> The Root Mean Squared Error from the y_Actual and y_Predicted obtained is <b>26971.57</b> as the units of RMSE is based on the Target Variable <b>SalePrice</b>."""

X_train = sn.add_constant(X_train)
X_test = sn.add_constant(X_test)

X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values

mlr_model = sn.OLS(y_train, X_train).fit()
print(mlr_model.summary())


X_train, X_test, y_train, y_test = data_split(df_house)

X_train = X_train[list(sfs_forward.k_feature_names_)]
X_test = X_test[list(sfs_forward.k_feature_names_)]

X_train, X_test = standardize(X_train, X_test)

X_train

scores = cross_val_score(estimator = LinearRegression(), X = X_train, y = y_train, cv = 10, scoring = 'r2')

print('All scores: ', scores)
print("\nMinimum score obtained: ", round(min(scores), 4))
print("Maximum score obtained: ", round(max(scores), 4))
print("Average score obtained: ", round(np.mean(scores), 4))



loocv_rmse = []

loocv = LeaveOneOut()


for train_index, test_index in loocv.split(X_train):
    X_train_l, X_test_l, y_train_l, y_test_l = X_train.iloc[train_index], X_train.iloc[test_index], \
                                               y_train.iloc[train_index], y_train.iloc[test_index]
    
    # instantiate the regression model
    linreg = LinearRegression()
    
    # fit the model on training dataset
    linreg.fit(X_train_l, y_train_l)
    
    # calculate MSE using test dataset
    # use predict() to predict the values of target variable 
    #pred = linreg.predict(X_train_l)
    mse = linreg.score(X_train_l, y_train_l)
    
    # calculate the RMSE
    rmse = np.sqrt(mse)
    
    # use append() to add each RMSE to the list 'loocv_rmse'
    loocv_rmse.append(rmse)

print(min(loocv_rmse))
print(max(loocv_rmse))
print(np.mean(loocv_rmse))

"""<b>Inference :</b> 

Upon implementing LOOCV, we obtain a Mean range of <b>93.84%</b>, indicates that the model can have high level of variance or estimates from each level of Folds are Highly Correlated.

**Regularization**
"""

ridge = Ridge(alpha = 1, max_iter = 500)
ridge.fit(X_train, y_train)

ridge.score(X_train,y_train)

"""<b>Inference :</b> 

After applying the ridge regression with alpha equal to one, we get <b>88.05%</b> as the RMSE value on Train set.
"""

ridge.score(X_test,y_test)

ridge.fit(X_train, y_train)
train_pred = ridge.predict(X_train)
mse = mean_squared_error(y_train, train_pred)
print('RMSE for train data',round(np.sqrt(mse), 4))

y_pred = ridge.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
print('RMSE for test data',round(np.sqrt(mse), 4))

"""**Ridge Regression using Grid Search**"""

tuned_paramaters = [{'alpha':[1e-15, 1e-10, 1e-8, 1e-4,1e-3, 1e-2, 0.1, 1, 5, 10, 20, 40, 60, 80, 100,150,200,250]}]
ridge = Ridge()
ridge_grid = GridSearchCV(estimator = ridge, 
                          param_grid = tuned_paramaters,
                          cv = 10)
ridge_grid.fit(X_train, y_train)

ridge_grid.score(X_train,y_train)

ridge_grid.score(X_test,y_test)

"""<b>Inference :</b> From above we can conclude that by implementing Grid Regularization, we see that there is not much variations between the Test and Train Model, in which case the error's are balanced."""

ridge.fit(X_train, y_train)
train_pred = ridge.predict(X_train)
mse = mean_squared_error(y_train, train_pred)
print('RMSE for train data',round(np.sqrt(mse), 4))

y_pred = ridge.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
print('RMSE for test data',round(np.sqrt(mse), 4))

"""## Summary Conclusion"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize=(20,10))
img=mpimg.imread('/content/Conclusion_Table_Final.PNG')
plt.xticks([]), plt.yticks([])
plt.imshow(img)

