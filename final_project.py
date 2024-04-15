# -*- coding: utf-8 -*-
"""Final_Project.ipynb


#Implementation of Regression Analysis and Prediction of Real Estate Prices with Machine Learning Algorithm

# Loading data
"""

# Commented out IPython magic to ensure Python compatibility.
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm # package for many different statistical models
from functools import reduce

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

#Package for inbuilt dataset in Python [scikit-learn]
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#from google.colab import drive
#drive.mount('/content/drive')

#above code to be used in colab

#Loading datasets
df1 = pd.read_csv('../data_folder/2017.csv',encoding='unicode_escape')
df2 = pd.read_csv('../data_folder/2018.csv',encoding='unicode_escape')
df3 = pd.read_csv('../data_folder/2019.csv',encoding='unicode_escape')
df4 = pd.read_csv('../data_folder/2020.csv',encoding='unicode_escape')
df5 = pd.read_csv('../data_folder/2021.csv',encoding='unicode_escape')
#Additional datasets after recognising risk - details in the RAID Log
df6 = pd.read_csv('../data_folder/2012.csv',encoding='unicode_escape')
df7 = pd.read_csv('../data_folder/2013.csv',encoding='unicode_escape')
df8 = pd.read_csv('../data_folder/2014.csv',encoding='unicode_escape')
df9 = pd.read_csv('../data_folder/2015.csv',encoding='unicode_escape')
df10 = pd.read_csv('../data_folder/2016.csv',encoding='unicode_escape')

df1.head()

df1.info()

"""#Data integration"""

#Merging 9 datasets together by rows
#https://datacarpentry.org/python-socialsci/11-joins/index.html
full_df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10], axis = 0)

#Displaying merged dataset details plus identigying datatypes
full_df.info()

#Displaying dataset's shape
print(full_df.shape)

#Checking unique values for each variable
full_df.nunique()

#identifying datatypes
full_df.info()

#Checking descriptive statsitics of the dataframe
full_df.describe()

"""#Pre-processing data

Missing values
"""

#Identifying missing values
nan_rows  = full_df.loc[full_df.isna().any(axis=1)]
print(nan_rows)

#Displaying missing values heatmap
full_df.isnull().sum()
sns.heatmap(full_df.isnull(), cbar = False).set_title("Missing values heatmap")

#Replacing blank fields with NaN
#https://stackoverflow.com/questions/13445241/replacing-blank-values-white-space-with-nan-in-pandas
print(full_df.replace(r'^\s*$', np.nan, regex=True))

"""Variables





"""

#Data reduction
transformed_df = full_df.drop(['Address', 'Eircode', 'Not Full Market Price','VAT Exclusive', 'Not Full Market Price','Property Size Description'], axis=1)
transformed_df.head()

#Displaying details of reduced dataset
transformed_df.info()

#Renaming fields
final_df = transformed_df.rename({'Date of Sale (dd/mm/yyyy)':'DATE_OF_SALE',
                                       'County':'COUNTY',
                                       'Price ()':'PRICE',
                                       'Description of Property':'PROPERTY_DESC'
                                       }, axis=1)
final_df.head()

# Replacing '' in cells with series.str methods
#https://stackoverflow.com/questions/36218971/pandas-remove-the-first-and-last-element-from-values-in-columns-in-a-df
for col in final_df:
    if final_df[col].dtype == 'object':
        final_df[col] = final_df[col].str.replace('', '')

final_df.head()

"""#Datatypes


"""

#Printing datatypes
print(final_df.dtypes)

#Displaying final rows
final_df.head()

"""#Changing datatype of the PRICE variable to float"""

#Replacing a sign for col in final_df:
final_df['PRICE'] = final_df['PRICE'].str.replace(',','')
final_df.head()

#Converting datatypes accordingly
#https://www.delftstack.com/howto/python-pandas/pandas-convert-object-to-float/

final_df['PRICE'] = final_df['PRICE'].astype(float, errors = 'raise')
final_df['PRICE'] = final_df['PRICE'].astype(int, errors = 'raise')
final_df['COUNTY'] = final_df['COUNTY'].astype(str, errors = 'raise')
final_df.head()

#Displaying new datatypes
print(final_df.dtypes)

"""#Outliers"""

#Checking descriptive statistics of the dataframe
final_df.describe()

#Identified outliers visualisation by boxplot
numeric_col = ['PRICE']
final_df.boxplot(numeric_col)

#Numer of outliers identification
#https://www.askpython.com/python/examples/detection-removal-outliers-in-python 
for x in ['PRICE']:
  q75,q25 = np.percentile(final_df.loc[:,x],[75,25])
  intr_qr = q75-q25


  max = q75+(1.5*intr_qr)
  min = q25-(1.5*intr_qr)
 
  final_df.loc[final_df[x] < min,x] = np.nan
  final_df.loc[final_df[x] > max,x] = np.nan

  print(final_df.isnull().sum())

#Removing the outliers
final_df = final_df.dropna(axis = 0)
#Checking null entries in the dataset
print(final_df.isnull().sum())

#Outliers removal confirmation by boxplot
final_df.boxplot(numeric_col)

"""#Analysing data

#Forecasting
"""

#Checking number of variables and records for the analysis
print('The final shape of the real estate prices dataset : ' + str(final_df.shape))

#Exporting DataFrame to xlsx
#final_df.to_excel(r'/content/drive/MyDrive/Datasets/forecast_df.xlsx', index=False, encoding='unicode_escape')

"""#Statistical Analysis"""

#Splitting one columns into three and adding these to the dataframe
#https://www.geeksforgeeks.org/split-a-text-column-into-two-columns-in-pandas-dataframe/
final_df[['DAY','MONTH','YEAR']] = final_df.DATE_OF_SALE.apply(
   lambda x: pd.Series(str(x).split("/")))
   
final_df.head()

#Data reduction
final_df = final_df.drop(['DATE_OF_SALE'], axis=1)
final_df.head()

#Exporting DataFrame to xlsx
#final_df.to_excel(r'/content/drive/MyDrive/Datasets/stats_df.xlsx', index=False, encoding='unicode_escape')

#Converting YEAR datatype to integer
final_df['YEAR'] = final_df['YEAR'].astype(int, errors = 'raise')
#Displaying datatypes of all the variables
print(final_df.dtypes)

"""#County Dublin"""

#Filtering rows by conditions
#https://sparkbyexamples.com/pandas/pandas-filter-rows-by-conditions/
dublin_df = final_df.loc[final_df['COUNTY'] == "Dublin"]
print('The shape of the new dataset called Dublin_df is ' + str(dublin_df.shape))

#Grouping average price per each of 10 years
#https://stackoverflow.com/questions/53287976/getting-the-average-value-for-each-group-of-a-pandas-dataframe
avg_price = dublin_df.groupby('YEAR')['PRICE'].agg(np.mean)
print(avg_price)

#Converting list to pandas dataframe colum
#https://stackoverflow.com/questions/42049147/convert-list-to-pandas-dataframe-column
dub_forecast_df = pd.DataFrame({'YEAR':[2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]})
print(dub_forecast_df)

#Adding another column to existing column
dub_forecast_df = pd.DataFrame({'PRICE': [221565,226857,256004,269078,291535,308654,322159,327730,336367,344812]})

#add column to existing df 
dub_forecast_df['YEAR'] = [2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
print (dub_forecast_df)

#Presenting details of new DataFrame called dub_forecast_df
dub_forecast_df.info()

"""#Linear Regression"""

#Linear Regression chart
slope, intercept, r_value, p_value, std_error = stats.linregress(dub_forecast_df['YEAR'],dub_forecast_df['PRICE'])

def regression(x):
  return slope*x + intercept

  model = list(map(regression,dub_forecast_df['PRICE']))

plt.scatter(dub_forecast_df['YEAR'],dub_forecast_df['PRICE'])
plt.title('Real estate prices over 10 years in Dublin')
plt.xlabel('Year')
plt.ylabel('Price (€)')
plt.plot(dub_forecast_df['YEAR'],dub_forecast_df['PRICE'])
plt.show()

#Calculating the Correlation Coefficient
slope, intercept, r_value, p_value, std_error = stats.linregress(dub_forecast_df['YEAR'],dub_forecast_df['PRICE'])

print('Slope: ' + str(slope))
print('Intercept: ' + str(intercept))
print('R value(The Correlation Coefficient is): ' + str(r_value))
print('P value: ' + str(p_value))
plt.show()

#Displaying Scatter Plot
dub_forecast_df.plot(kind='scatter',x='YEAR', y='PRICE',
        title='Real estate prices in Dublin over 10 years',logx=True,
        xlabel= 'Year',
        ylabel= 'Price (€)',
        fontsize=10,
        grid=True)

plt.show()

#Displaying scatter plot - Linear Regression
sns.regplot('YEAR', 'PRICE', dub_forecast_df,
         line_kws = {"color":"b"}, ci=None)

"""#Building model"""

#Linear Regression model
reg = linear_model.LinearRegression()

#Reshaping data from 1D array to 2D array
#https://stackoverflow.com/questions/58663739/reshape-your-data-either-using-array-reshape-1-1-if-your-data-has-a-single-fe
reg.fit(np.array(dub_forecast_df['YEAR']).reshape(-1, 1), np.array(dub_forecast_df['PRICE']))

"""Creating the Linear Regression model"""

lm = sm.OLS.from_formula('YEAR ~ PRICE', dub_forecast_df)
model_result = lm.fit()

#Displaying model details
print(model_result.summary())

model_result.pvalues

"""Model Predictions"""

#Predicting future housing prices
reg.fit(np.array(dub_forecast_df['YEAR']).reshape(-1, 1), np.array(dub_forecast_df['PRICE']))
print('Average real estate price in County Dublin in 2022 is predicted to be equal to '
 + str(reg.predict([[2022]])))
print('Average real estate price in County Dublin in 2023 is predicted to be equal to '
 + str(reg.predict([[2023]])))
print('Average real estate price in County Dublin in 2024 is predicted to be equal to '
 + str(reg.predict([[2024]])))
print('Average real estate price in County Dublin in 2025 is predicted to be equal to '
 + str(reg.predict([[2025]])))
print('Average real estate price in County Dublin in 2026 is predicted to be equal to '
 + str(reg.predict([[2026]])))
print('Average real estate price in County Dublin in 2027 is predicted to be equal to '
 + str(reg.predict([[2027]])))
print('Average real estate price in County Dublin in 2028 is predicted to be equal to '
 + str(reg.predict([[2028]])))
print('Average real estate price in County Dublin in 2029 is predicted to be equal to '
 + str(reg.predict([[2029]])))
print('Average real estate price in County Dublin in 2030 is predicted to be equal to '
 + str(reg.predict([[2030]])))

"""#County Cork"""

#Filtering rows by conditions
#https://sparkbyexamples.com/pandas/pandas-filter-rows-by-conditions/
cork_df = final_df.loc[final_df['COUNTY'] == "Cork"]
print('The shape of the new dataset called Cork_df is ' + str(cork_df.shape))

#Grouping average price per each of 10 years
#https://stackoverflow.com/questions/53287976/getting-the-average-value-for-each-group-of-a-pandas-dataframe
avg_price_2 = cork_df.groupby('YEAR')['PRICE'].agg(np.mean)
print(avg_price_2)

#Converting list to pandas dataframe colum
#https://stackoverflow.com/questions/42049147/convert-list-to-pandas-dataframe-column
cork_forecast_df = pd.DataFrame({'YEAR':[2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]})
print(cork_forecast_df)

#Adding another column to existing column
cork_forecast_df = pd.DataFrame({'PRICE': [166401,152794,161300,176102,192041,207679,224215,232507,241586,256389]})

#add column to existing df 
cork_forecast_df['YEAR'] = [2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
print (cork_forecast_df)

#Presenting details of new DataFrame called dub_forecast_df
cork_forecast_df.info()

"""#Linear Regression"""

#Linear Regression chart
slope, intercept, r_value, p_value, std_error = stats.linregress(cork_forecast_df['YEAR'],cork_forecast_df['PRICE'])

def regression(x):
  return slope*x + intercept

  model = list(map(regression,cork_forecast_df['PRICE']))

plt.scatter(cork_forecast_df['YEAR'],cork_forecast_df['PRICE'])
plt.title('Real estate prices over 10 years in Cork')
plt.xlabel('Year')
plt.ylabel('Price (€)')
plt.plot(cork_forecast_df['YEAR'],cork_forecast_df['PRICE'])
plt.show()

#Calculating the Correlation Coefficient
slope, intercept, r_value, p_value, std_error = stats.linregress(cork_forecast_df['YEAR'],cork_forecast_df['PRICE'])

print('Slope: ' + str(slope))
print('Intercept: ' + str(intercept))
print('R value(The Correlation Coefficient is): ' + str(r_value))
print('P value: ' + str(p_value))
plt.show()

#Displaying Scatter Plot
cork_forecast_df.plot(kind='scatter',x='YEAR', y='PRICE',
        title='Real estate prices in Cork over 10 years',logx=True,
        xlabel= 'Year',
        ylabel= 'Price (€)',
        fontsize=10,
        grid=True)

plt.show()

#Displaying scatter plot - Linear Regression
sns.regplot('YEAR', 'PRICE', cork_forecast_df,
         line_kws = {"color":"b"}, ci=None)

"""#Building model"""

#Reshaping data from 1D array to 2D array
#https://stackoverflow.com/questions/58663739/reshape-your-data-either-using-array-reshape-1-1-if-your-data-has-a-single-fe
reg.fit(np.array(cork_forecast_df['YEAR']).reshape(-1, 1), np.array(cork_forecast_df['PRICE']))
#Linear Regression model
reg = linear_model.LinearRegression()

"""Creating the Linear Regression model"""

lm = sm.OLS.from_formula('YEAR ~ PRICE', cork_forecast_df)
model_result = lm.fit()

#Displaying model details
print(model_result.summary())

model_result.pvalues

"""Model Predictions"""

#Predicting future housing prices
reg.fit(np.array(cork_forecast_df['YEAR']).reshape(-1, 1), np.array(cork_forecast_df['PRICE']))
print('Average real estate price in County Cork in 2022 is predicted to be equal to '
 + str(reg.predict([[2022]])))
print('Average real estate price in County Cork in 2023 is predicted to be equal to '
 + str(reg.predict([[2023]])))
print('Average real estate price in County Cork in 2024 is predicted to be equal to '
 + str(reg.predict([[2024]])))
print('Average real estate price in County Cork in 2025 is predicted to be equal to '
 + str(reg.predict([[2025]])))
print('Average real estate price in County Cork in 2026 is predicted to be equal to '
 + str(reg.predict([[2026]])))
print('Average real estate price in County Cork in 2027 is predicted to be equal to '
 + str(reg.predict([[2027]])))
print('Average real estate price in County Cork in 2028 is predicted to be equal to '
 + str(reg.predict([[2028]])))
print('Average real estate price in County Cork in 2029 is predicted to be equal to '
 + str(reg.predict([[2029]])))
print('Average real estate price in County Cork in 2030 is predicted to be equal to '
 + str(reg.predict([[2030]])))