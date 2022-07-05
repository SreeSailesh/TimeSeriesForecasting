import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
color = sns.color_palette()
sns.set_style('darkgrid')
from subprocess import check_output

train = pd.read_csv("train.csv")
train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d")
print(train.head())
# per 1 store, 1 item
train_df = train[train['store']==1]
train_df = train_df[train['item']==1]
# train_df = train_df.set_index('date')
train_df['year'] = train['date'].dt.year
train_df['month'] = train['date'].dt.month
train_df['day'] = train['date'].dt.dayofyear
train_df['weekday'] = train['date'].dt.weekday

print(train_df.head())

#identify the best d ranges in ARIMA

def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    #print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag=20)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    pvalue = dftest[1]
    if pvalue < 0.03:
        #print('p-value = %.4f. The series is likely stationary.' % pvalue)
        return pvalue
    else:
        #print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
        return -1

    print(dfoutput)


target_cloumn = "sales"
d_range = [*range(0,11,1)]
d_list = []
print(d_range)
test_df = train_df
for d in d_range:
    if d > 0:
        test_diff = test_df[target_cloumn] -test_df[target_cloumn].shift(d)
    else:
        test_diff = test_df[target_cloumn]
    test_diff = test_diff.dropna(inplace = False)
    b = test_stationarity(test_diff)
    if b > 0:
        #print(str(d)+" "+str(b))
        d_list.append(d)

print(d_list)


#Determining the best P value for ARIMA


