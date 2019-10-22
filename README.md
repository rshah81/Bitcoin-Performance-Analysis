# Bitcoin-Performance-Analysis
Analyze the risk-return characteristics of Bitcoin and Actively managed crypto hedge funds
Part I: Construct monthly return series for Bitcoin and S&P500 index.
 (1)Read in daily Bitcoin prices (BTC_DailyPrice_USD.csv) and S&P500 indices (SP500_DailyIndex.csv) and compute daily returns for both.
 (2)Compute monthly returns series for Bitcoin and S&P500 based on daily returns.
 (3)Read in the monthly returns for Fama-French three factors and risk-free rates from “FF3Factors_Monthly.csv”. Make sure to convert all percentage values into decimal values(dividing by 100).
 (4)Merge the monthly returns for Bitcoin, S&P500, Fama-French three factors and risk-free rates into one Dataframe.
 (5)Keep monthly returns for the period from August 2010 to December 2018.
Part II: Statistics, plotting, and CAPM analysis
  (1)Compute the following summary statistics for excessBitcoin and excess S&P 500 returns: mean (annualized), standard deviation (annualized), Sharpe ratio, correlation coefficient, skew, and kurtosis;
  (2)Plot the cumulative returns for Bitcoin and S&P 500 over the sample period;
  (3)Plot the empirical distribution of Bitcoin excessreturnsusing histogram;
  (4)Plot the scatter plot with excessBitcoin returns on the y-axis and excessS&P 500 returns on the x-axis;
  (5)CAPM analysis: regress the excessBitcoin returns on the excessS&P 500 returnsand report the regression output (alpha, beta, R-square, T-stat, etc.)
  (6)Fama-French analysis: regress the excessBitcoin returns on Fama-Frenh three factors and report the regression output (alpha, beta, R-square, T-stat, etc.)
 Part III: Performance of actively managed crypto hedge funds
  (1)Read in the monthly returns for HFR Cryptocurrency Index from “HFR_CrpytoCurrency_IndexReturns.csv”. The index returns are compiled by Hedge Fund Research based on the reported returns of hedge funds actively investing in cryptocurrencies. The sample period is from January 2015 to December 2018.
  (2)Merge the HFR Cryptocurrency Index returns with other monthly return series you have computed so far: Bitcoin, S&P 500, Fama-French three factors, and the risk-freerates.
  (3)Compute the following statistics for the excess HFR index returns and compare them with the statistics for the excess S&P 500 and excess Bitcoin returns for the period from January 2015 to December 2018: mean, standard deviation, Sharpe ratio, skew, kurtosis, and value-at-risk (VaR) at the 5% level.
  (4)Benchmark performance analysis 
  (I): Now use the monthly Bitcoin returns as the benchmark to evaluate whether the actively managed crypto hedge funds add any additional value to investors. Regress the excess HFR index returns on the excess Bitcoin returns: Report the coefficient estimates and t-statistics for α and β. Also compute the information ratio (IR) as the intercept (α) divided by the standard deviation of the residuals (ԑ).(5)Benchmark performance analysis 
  (II): Expand the benchmark performance analysis in (4) and include an additional term to test whether actively managed crypto hedge funds have any timing ability:푟퐻퐹−푟푓=훼+훽1[푟퐵푇푈−푟푓]+훽2[푟퐵푇푈−푟푓]++휀,where [푟퐵푇푈−푟푓]+=max⁡(0,푟퐵푇푈−푟푓). In this specification, a positive and statistically significant 훽2would indicate that the hedge fund managers have timing ability.Based on the results from Python analyses, write a brief summary on the risk-return characteristics of Bitcoin investment and whether actively managed crypto hedge funds add any additional value to investors (i.e., beyond the value investors receive from a passive long position in Bitcoin).
