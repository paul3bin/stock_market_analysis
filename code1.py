from __future__ import division
import pandas as pd 
from pandas import Series, DataFrame
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('whitegrid')
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import DataReader
from datetime import datetime 

tech_list = ['AAPL','GOOG','MSFT','AMZN']
end = datetime.now() #getting the date og today
start = datetime(end.year-1,end.month,end.day)
for stock in tech_list:
    globals()[stock] = DataReader(stock,'yahoo',start,end)
#globals uses the values from the right hand side and convert it into a global variable 
print(AAPL.head())
#the string name is converted to a dataframe using globals in line 16
print(AAPL.describe())
print(AAPL.info())

AAPL['Adj Close'].plot(legend=True,figsize=(10,4))
plt.show()

AAPL['Volume'].plot(legend=True,figsize=(10,4))
plt.show()

"""
Links for moving average:
http://www.investopedia.com/terms/movingaverage.asp
http://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp
"""
ma_day = [10,20,50]
for ma in ma_day:
    column_name = "MA for "+str(ma)+" days"
    AAPL[column_name] = AAPL['Adj Close'].rolling(ma).mean()#new way to find the rolling mean
AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(10,4))
plt.show()

#daily returns and risk in the stock
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(10,4), legend=True,linestyle="--",marker='o')
plt.show()

#Average data return can be done using histogram
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
plt.show()
#the above plot can be done in a short way given below
AAPL['Daily Return'].hist(bins=100)
plt.show()

#To analyse all the stocks returned
closing_df = DataReader(tech_list,'yahoo',start,end)['Adj Close']
tech_rets = closing_df.pct_change()
print(tech_rets.head())
#calculating google's daily return to itself
sns.jointplot(GOOG,GOOG,tech_rets,kind='scatter',color='seagreen')
plt.show()

sns.jointplot(GOOG,MSFT,tech_rets,kind='scatter')
plt.show()

#part 3

#pairplots
sns.pairplot(tech_rets.dropna())
plt.show()
#manually creating the pairplot
returns_fig = sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
plt.show()
#manually creating the pairplot for closing prices
returns_fig = sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
plt.show()
#corelation plot is deprecated from seaborn and is replaced by heatmap
sns.heatmap(tech_rets,annot=True)
plt.show()
sns.heatmap(closing_df,annot=True)
plt.show()

#part 4

#stock and its risks
rets = tech_rets.dropna()
area = np.pi*20
plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')
#http://matplotlib.org/users/annotations_guide.html
for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
        label,
        xy=(x,y),xytext=(50,50),
        textcoords = 'offset points', ha='right',va='bottom',
        arrowprops = dict(arrowstyle='-',connectionstyle='arc3,rad=-0.3')
    )
#zip allows to call all the things inside at once
plt.show()

#part 5

#Value at risks
#look at the wikipedia page of the quantile
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
plt.show()
print(rets['AAPL'].quantile(0.05))