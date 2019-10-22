#!/usr/bin/env python
# coding: utf-8

# Rishabh Shah 
# Bitcoin Performance Analysis 
# 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import norm


# Part 1: Construct monthly return series for Bitcoin and S&P500 index.

# In[2]:


#1
#Read in daily Bitcoin prices (BTC_DailyPrice_USD.csv) and S&P500 indices 
#(SP500_DailyIndex.csv) and compute daily returns for both.

bitcoinprices = pd.read_csv('BTC_DailyPrice_USD.csv')
bitcoinprices.set_index('Date', inplace=True)
bitcoinprices.index = pd.to_datetime(bitcoinprices.index)
bitcoinprices_close = bitcoinprices['Close'].rename(columns={'Close': 'Bitcoin'})
bitcoinprices_returns = bitcoinprices_close.pct_change()
print(bitcoinprices_returns.head())

sp500indices = pd.read_csv('SP500_DailyIndex.csv')
sp500indices.set_index('Date', inplace=True)
sp500indices.index = pd.to_datetime(sp500indices.index)
sp500indices_close = sp500indices['Close'].rename(columns={'Close': 'SP500'})
sp500indices_returns = sp500indices_close.pct_change()
print(sp500indices_returns.head())


# In[3]:


#2
#Compute monthly returns series for Bitcoin and S&P500 based on daily returns.
monthbitcoinreturns = ((bitcoinprices_returns+1).resample('M').prod()-1)
print(monthbitcoinreturns.head())
monthsp500returns    = ((sp500indices_returns +1).resample('M').prod()-1)
print(monthsp500returns.head())


# In[4]:


#3
#Read in the monthly returns for Fama-French three factors and risk-free rates from
#“FF3Factors_Monthly.csv”. Make sure to convert all percentage values into decimal values
#(dividing by 100).
ff3factors = pd.read_csv('FF3Factors_Monthly.csv', parse_dates = True, index_col = "Date")/100
# ff3factors = ff3factors/100
# ff3factors.set_index('Date', inplace=True)
# ff3factors.index = pd.to_datetime(ff3factors.index)
ff3factors.tail()


# In[5]:


#4
#Merge the monthly returns for Bitcoin, S&P500, Fama-French three factors and risk-free rates into one Dataframe
totaldf = pd.concat([monthbitcoinreturns,monthsp500returns,ff3factors], axis =1)
totaldf.rename(columns={totaldf.columns[0]: "Bitcoin", totaldf.columns[1]: "SP500"}, inplace = True)
totaldf.tail()


# In[6]:


#5
#Keep monthly returns for the period from August 2010 to December 2018.
monthbitcoinreturns = monthbitcoinreturns[(monthbitcoinreturns.index >= '2010-08-01') & (monthbitcoinreturns.index <= '2019-01-01')]
print(monthbitcoinreturns.head())

monthsp500returns    = monthsp500returns[(monthsp500returns.index >= '2010-08-01') & (monthsp500returns.index <= '2019-01-01')]
print(monthsp500returns.head())

new_totaldf = pd.concat([monthbitcoinreturns,monthsp500returns,ff3factors], axis =1)
new_totaldf =  new_totaldf[(new_totaldf.index >= '2010-08-01') & (new_totaldf.index <= '2019-01-01')]
new_totaldf.rename(columns={new_totaldf.columns[0]: "Bitcoin", new_totaldf.columns[1]: "SP500"}, inplace = True)
new_totaldf.head()


# Part 2: Statistics, plotting, and CAPM analysis 

# In[7]:


#1
#Compute the following summary statistics for excess Bitcoin and excess S&P 500 returns:
#mean (annualized), standard deviation (annualized), Sharpe ratio, correlation coefficient, skew, and kurtosis

bitcoin_excess_returns = pd.DataFrame()
#for Bitcoin in new_totaldf.columns:
bitcoin_excess_returns['Bitcoin ER'] = new_totaldf['Bitcoin'] - new_totaldf['RF']
    
bitcoin_excess_returns.tail()

snp_excess_returns = pd.DataFrame()
#for SP500 in new_totaldf.columns:
snp_excess_returns['SP500 ER'] = new_totaldf['SP500'] - new_totaldf['RF']
    
snp_excess_returns.tail()

excess_returns_df = pd.concat([bitcoin_excess_returns,snp_excess_returns], axis =1)

print(excess_returns_df.tail())

annualized_means = excess_returns_df.mean()*12
print("Bitcoin and S&P500 Annualized Means")
print(annualized_means.tail())

annualized_std = excess_returns_df.std()*(12**0.5)
print("Bitcoin and S&P500 Annualized Standard Deviations")
print(annualized_std.tail())

er_sharperatio = annualized_means / annualized_std
print('Bitcoin and SP500 Sharpe Ratio')
print(er_sharperatio.tail())

excess_returns_corr_coeff = excess_returns_df.corr()
print('Correlation Coefficent Bitcoin and SP500 Excess Returns')
print(excess_returns_corr_coeff)

skew_er = excess_returns_df.skew()
print("Excess Returns Skew")
print(skew_er)

kurt_er = excess_returns_df.kurt()
print("Excess Returns Kurtosis")
print(kurt_er)


# In[8]:


#2 
#Plot the cumulative returns for Bitcoin and S&P 500 over the sample period

snp_cumm_return = sp500indices_returns + 1
# use cumprod and subtract 1 again to get the value for the snp cummulative return 
snp_cumm_return = snp_cumm_return.cumprod() - 1
snp_cumm_return.tail()
print("S&P 500 Cummulative Return")
snp_cumm_return.plot()


# In[9]:


bit_cumm_return = bitcoinprices_returns + 1
# use cumprod and subtract 1 again to get the value for the snp cummulative return 
bit_cumm_return = bit_cumm_return.cumprod() - 1
bit_cumm_return.tail()
print("Bitcoin Cummulative Return ")
bit_cumm_return.plot()


# In[10]:


#3 Plot the empirical distribution of Bitcoin excess returns using histogram

bitcoin_excess_returns.hist()
plt.ylabel('Frequency')
plt.xlabel('Bitcoin Excess Return')
plt.title('Histogram')


# In[11]:


#4 Plot the scatter plot with excess Bitcoin returns on the y-axis and excess S&P 500 returns on the x-axis

excess_returns_df.plot(kind='scatter',x= 'SP500 ER' ,y='Bitcoin ER')
plt.show()


# In[12]:


#5 CAPM analysis: regress the excess Bitcoin returns on the excess S&P 500 returns and report the regression output (alpha, beta, R-square, T-stat, etc.)
CAPM_reg = smf.ols("bitcoin_excess_returns ~ snp_excess_returns ", excess_returns_df).fit()
print(CAPM_reg.summary())


# In[13]:


new_ff3factors = ff3factors[(ff3factors.index >= '2010-08-01') & (ff3factors.index <= '2019-01-01')]
new_ff3factors.drop(new_ff3factors.columns[len(new_ff3factors.columns)-1], axis=1, inplace=True)
new_ff3factors.tail()


# In[14]:


#6 Fama-French analysis: regress the excess Bitcoin returns on Fama-French three factors and report the regression output (alpha, beta, R-square, T-stat, etc.)
Fama_french_reg = smf.ols ("bitcoin_excess_returns ~ new_ff3factors ", excess_returns_df).fit()
print(Fama_french_reg.summary())


# Part 3  Performance of actively managed crypto hedge funds

# In[15]:


#1. Read in the monthly returns for HFR Cryptocurrency Index from “HFR_CrpytoCurrency_IndexReturns.csv”. 
mr_hfr = pd.read_csv('HFR_CrpytoCurrency_IndexReturns.csv')
mr_hfr.set_index('Date', inplace=True)
mr_hfr.index = pd.to_datetime(mr_hfr.index)
mr_hfr.tail()


# In[16]:


#2 Merge the HFR Cryptocurrency Index returns with other monthly return series you have computed so far: Bitcoin, S&P 500, Fama-French three factors, and the risk-free rates.
rev_totaldf = pd.concat([mr_hfr,monthbitcoinreturns,monthsp500returns,ff3factors], axis =1)
rev_totaldf =  rev_totaldf[(rev_totaldf.index >= '2015-01-01') & (rev_totaldf.index <= '2019-01-01')]
rev_totaldf.rename(columns={rev_totaldf.columns[1]: "Bitcoin", rev_totaldf.columns[2]: "SP500"}, inplace = True)
rev_totaldf.head()


# In[17]:


#3 Compute the following statistics for the excess HFR index returns and compare them with
#the statistics for the excess S&P 500 and excess Bitcoin returns for the period from January
#2015 to December 2018: mean, standard deviation, Sharpe ratio, skew, kurtosis, and valueat-risk (VaR) at the 5% level.
ret_hf_excess_returns = pd.DataFrame()
for ret_hf in rev_totaldf.columns:
    ret_hf_excess_returns['ret_hf ER'] = rev_totaldf['ret_hf'] - rev_totaldf['RF']
print('HFR Index Excess Returns')
print(ret_hf_excess_returns.tail())

print('HFR Index excess return mean')
hfr_mean = print(ret_hf_excess_returns.mean())

print('HFR Index excess return std dev')
hfr_std  = print(ret_hf_excess_returns.std())

print('HFR Index Excess Return Sharpe Ratio')
hfr_sr = (ret_hf_excess_returns.mean()) / ret_hf_excess_returns.std()
print(hfr_sr)

print('HFR Index Excess Return Skew')
print(ret_hf_excess_returns.skew())

print('HFR Index Excess Return Kurtosis')
print(ret_hf_excess_returns.kurt())

print('Value-at-Risk (VaR) 5% Level')
print(norm.ppf(1-0.95, ret_hf_excess_returns.mean(), ret_hf_excess_returns.std()))
    


# In[18]:


# 3 (Cont.) Comparing HFR index excess returns to excess SNP and excess Bitcoin returns for Jan 2015 - Dec 2018
new_snp_bit_returns = excess_returns_df[(excess_returns_df.index >= '2015-01-01') & (excess_returns_df.index <= '2019-01-01')]

print(new_snp_bit_returns.head())

new_bitcoin_excess_returns = bitcoin_excess_returns[(bitcoin_excess_returns.index >= '2015-01-01') & (bitcoin_excess_returns.index <= '2019-01-01')]
print(new_bitcoin_excess_returns.head())

print('Revised SNP and Bit excess return mean')
hfr_mean = print(new_snp_bit_returns.mean()*12)

print('Revised SNP and Bit excess return std dev')
hfr_std  = print(new_snp_bit_returns.std()*(12**0.5))

print('Revised SNP and Bit Excess Return Sharpe Ratio')
hfr_sr = (new_snp_bit_returns.mean() / new_snp_bit_returns.std())
print(hfr_sr)

print('Revised SNP and Bit Excess Return Skew')
print(new_snp_bit_returns.skew())

print('Revised SNP and Bit Excess Return Kurtosis')
print(new_snp_bit_returns.kurt())

print('Value-at-Risk (VaR) 5% Level')
print(norm.ppf(1-0.95, new_snp_bit_returns.mean(), new_snp_bit_returns.std()))


# In[19]:


#4Benchmark performance analysis (I): Now use the monthly Bitcoin returns as the benchmark to evaluate whether the actively managed crypto hedge funds add any additional
#value to investors. Regress the excess HFR index returns on the excess Bitcoin returns:
#𝑟𝐻𝐹 − 𝑟𝑓 = 𝛼 + 𝛽[𝑟𝐵𝑇𝑈 − 𝑟𝑓] + 𝜀.
#Report the coefficient estimates and t-statistics for α and β. Also compute the information
#ratio (IR) as the intercept (α) divided by the standard deviation of the residuals (ԑ).
HFR_reg = smf.ols("ret_hf_excess_returns ~ new_bitcoin_excess_returns ", new_snp_bit_returns).fit()
print(HFR_reg.summary())


# In[20]:


Info_Ratio = HFR_reg.params[0]/np.sqrt(HFR_reg.scale)
Info_Ratio


# In[21]:


#5 Benchmark performance analysis (II): Expand the benchmark performance analysis in (4) and include an additional term to test whether actively managed crypto hedge funds have any
#timing ability: 𝑟𝐻𝐹 − 𝑟𝑓 = 𝛼 + 𝛽1[𝑟𝐵𝑇𝑈 − 𝑟𝑓] + 𝛽2[𝑟𝐵𝑇𝑈 − 𝑟𝑓] + + 𝜀 , where [𝑟𝐵𝑇𝑈 − 𝑟𝑓] + = max (0, 𝑟𝐵𝑇𝑈 − 𝑟𝑓). In this specification, a positive and statistically
#significant 𝛽2 would indicate that the hedge fund managers have timing ability.
rev_totaldf['Bitcoin_ret'] = np.where(rev_totaldf['Bitcoin'] > 0, rev_totaldf['Bitcoin'],0)
rev_totaldf.head()


# In[22]:


HFR_reg_2 = smf.ols("ret_hf_excess_returns ~ new_bitcoin_excess_returns + Bitcoin_ret ", rev_totaldf).fit()
print(HFR_reg_2.summary())


# In[23]:


Info_Ratio_2 = HFR_reg_2.params[0]/np.sqrt(HFR_reg_2.scale)
Info_Ratio_2


# In[ ]:




