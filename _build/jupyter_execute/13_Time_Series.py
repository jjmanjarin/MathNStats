# Time Series

Let's give a brief introduction to the analysis of time series. We will not enter into the Gauss-Markov conditions this time and just deal with the basic properties of this type of data. However, we should keep in mind that in this context we seem to be loosing one of the main ideas in Stats, that of randomness: the sample we take is not random, but a series of values indexed by time. 

Then we must think of our collection of data as one of the possible outcomes of a stochastic process over time, i.e. each of the values in the time series is the result of a random process and it is only one of the possible outcomes of that process.

## Components of a Time Series

A general times series can be thought of as been composed by four different parts:

  * The **trend**: it is the long-run smooth and regular movement of the series
  * The **cycle**: it is the oscillatory part of the the trend that represents a smooth and continuous evolution in the long-run (repeated each certain years)
  * The **seasonality**: it is the systematic repeated pattern observed "yearly" in the series
  * The **irregular component**: it is represented by the random errors

Note that since the cycle is sometimes assumed to be part of the trend, since both represent the long-run and then trend consists in a *pure trend* part and the *oscillatory trend*.

There are three different ways in which we may combine the different components into just one single time series, but in general we may say that the response $y_i$ is a function

\begin{equation}
y_i = f(T_i,\,C_i,,S_i,\,I_i)
\end{equation}

  This function may represent one of the following three schemes
  
  * **Additive scheme**: This is the case in which the components are represented in terms of a linear combination
  
  \begin{equation}
  y_i = T_i + C_i + S_i + I_i
  \end{equation}
  
  this case is used whenever there is not a high variability of the series along time. The series in case is known as **linear** or **arithmetic**.
  
  * **Multiplicative scheme**: In this case we represent the output as the product of all the components, then
  
  \begin{equation}
  y_i = T_i\cdot C_i \cdot S_i\cdot I_i
  \end{equation}
  
  this scheme is used when it is much better to represent the changes in terms of percentages than in absolute terms. Generally, whenever the varaibility of the series is not constant long time. The series in case is known as **exponential** or **geometric**.
  
  * **Mixed scheme**: In this case we typically have the trend, cycle and seasonality in a multiplicative scheme and the irregular component in addicitve scheme
  
  \begin{equation}
  y_i = (T_i \cdot C_i \cdot S_i) + I_i
  \end{equation}
  
  this scheme is relevant when the variability of the seasonal oscillations can be assumed to be constant but their magnitude grows with time.

Whenever we have a multiplicative scheme, it will always be transformed into an additive one by means of a logarithmic transformation

  \begin{equation}
  \log(y_i) = \log(T_i) + \log(C_i) +\log(S_i) + \log(I_i)
  \end{equation}

then whatever techinque that may be applied to the additive scheme can be applied to the multiplicative one if it has been transformed already.

## Detection of the Scheme

There are different ways to determine the proper scheme of a time series. Here we are only going to deal with the method of the **seasonal differences and ratios**. In this case we consider the set of differences and ratios of the observations with one seasonal lag of difference

\begin{equation}
\begin{array}{rcl}
d_{i,t} & = &  y_{i,t} - y_{i,t-1} \\
r_{i,t} & = & \displaystyle\frac{y_{i,t}}{y_{i,t-1}}
\end{array}
\end{equation}

i.e. we consider the differences and ratios of, for example, the same month in lagged consecutive moments, for example, January_2018 and January_2019. Then both, differences and ratios, will be two distributions of values in which we can compute the mean and the standard deviation such that we determine the **coefficients of variation** as

\begin{equation}
CV(d) = \frac{\bar d}{sd_d},\qquad CV(r) = \frac{\bar r}{sd_r}
\end{equation}

then the criteria are the following

 * If $CV(d)<CV(r)$ then the scheme is additive
 * If $CV(d)>CV(r)$ then the scheme is multiplicative

i.e. we consider the scheme corresponding to the distribution with the smallest variability.

### Examples of Time Series Models

There are many different models we may consider but from one perspective we consider a first classification with respect to the moment in which the variables of interest affect the response. In this sense we will find

  * **Static Models**, in which we model contemporaneous variables, for example the *static Phillips curve* that relates the inflation with the unemployment
  
  \begin{equation}
  \text{inf}_t = \beta_0 + \beta_1\,\text{unem}_t + u_t
  \end{equation}
  
  which assumes a constant natural rate of unemployment and constant inflationary expectations
  
  * **Finite Distributed Lag Models**, where we allow for some *x*'s variables to affect the response with a lag, as could be that the interest rate is a function of contemporaneous inflation rate and also of the inflation rate in two previous years
  
  \begin{equation}
  \text{int}_t = \beta_0 + \beta_1\,\text{inf}_t + \beta_2\,\text{inf}_{t-1}+ \beta_3\,\text{inf}_{t-2} + u_t
  \end{equation}
  
  in this case, we know $\beta_1$ as the **impact propensity** or the **short-run elasticity** if it is a log-log model and $\beta_1+\beta_2+\beta_3$ as the **long-run propensity** or **long-run elasticity** in log-log models. In Python we can create this lagged variables as `df['laginf'] = df['inf'].shift(-1)`, where the number in the shift method is the amount of lags we are going to compute.

## Time Series in Python

Let's begin, as usual, by loading all the modules we may need at some point or another of the study

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from datetime import datetime

from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

In this document we are going to use the well known dataset "AirPassengers" from Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) *Time Series Analysis, Forecasting and Control.* Third Edition. Holden-Day. Series G. which can be downloaded [here](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv) and which contains the values of the monthly totals of international airline passengers from 1949 to 1960. Once downloaded we can load it as

from google.colab import drive
drive.mount('mydrive')

mydata = pd.read_csv('/content/mydrive/My Drive/IEXL - Bootcamp - Math&Stats 20-21 September/data/AirPassengers.csv')
mydata.head()

print(mydata.dtypes)

idx = pd.to_datetime([datetime.strptime(i, '%Y-%m') for i in mydata['Month']])
mydata.set_index(idx, inplace=True)
mydata.drop('Month', axis = 1, inplace=True)
mydata

See, however, that the object is not a time series because the *Month* column does not have a time-consistent format. To change it, we do

dateparse = lambda dates : pd.datetime.strptime(dates, '%Y-%m')
mydata = pd.read_csv('/content/mydrive/My Drive/IEXL - Bootcamp - Math&Stats 20-21 September/data/AirPassengers.csv', 
                     parse_dates = ['Month'], 
                     index_col = 'Month', 
                     date_parser = dateparse)

What this code does is: specify the column with the time information with **parse_dates**, then specify the index of the pandas dataframe with **index_column** and finally  use an annonymous function stored in *dateparse* to convert the string year-month into a date with the year-month-day format

mydata.head()

if we print the index of the set

mydata.index

we can see that the **dtype** is now a **datetime** type and so, it is a Time Series.

An interesting point with these time series is that we may subset with respect to the year as

mydata['1949']

## Measures of Dependence

The complete description of a time series may need the following joint probability distribution 

\begin{equation}
F_{t_1t_2\dots t_n}(c_1,c_2,\dots,c_n)=P(x_{t_1}\leq c_1,\dots x_{t_n}\leq c_n)
\end{equation}

extendend to arbitrary time points with $n$ arbitrary random variables. However, this distribution is an unwidly tool for describing the time series. In this situation it is much better to work with the marginal distributions 

\begin{equation}
F_t(x)=P(x_t\leq x)
\end{equation}

which, when they exist, offer valuable marginal information of the series. One of the main functions we are going to define in this context is the **Autocovariance Function**

\begin{equation}
\gamma_x(s,t)=E[(x_s-\mu_s)(x_t-\mu_t)]
\end{equation}

defined for all times $s$ and $t$ and which measures the *linear* dependence between two points on the same series at different times.

This function is, therefore, measuring if observations at *any* moment of the series are still affecting distant in time observations. Note that smooth series will exhibit an autocorrelation high when $s$ and $t$ are far appart while choppy series will have autocorrelations close to 0 (remember that as usual there may still be other types of relationships).

As an example, consider a three-point moving average of white noise (mean 0 and variance $\sigma_w^2$)

\begin{equation}
v_t = \frac{1}{3}(w_{t-1} + w_t + w_{t+1})
\end{equation}

it is easy to check that the autocorrelation function is

\begin{equation}
\gamma_v(s,t)=\left\{\begin{array}{ll} \frac{3}{9}\sigma_w^2, & s=t\\\frac{2}{9}\sigma_w^2,  & |s-t|=1\\\frac{1}{9}\sigma_w^2,  & |s-t|=2\\0, & |s-t|>2\end{array}\right.
\end{equation}

Now, as usual, we need should introduce the **Autocorrelation Function**, ACF, which measures the *strength* of the dependency

\begin{equation}
\rho_{xy}(s,t)=\frac{\gamma(s,t)}{\sqrt{\gamma(s,s)\gamma(t,t)}}
\end{equation}

which, as with the usual correlation, is a number between $[-1,1]$ due to the Cauchy-Schwarz inequality and which measures the linear predictabiity of the series at time $t$, $x_t$, using only the value $x_s$. Later we will define another function, the **Partial Autocorrelation Function**.

### Stationarity

different models that we may consider in the Time Series context which may be divided into **stationary** and **non-stationary**. A process is known as stationary if its statistical characteristics do not change over time, i.e. if, for example, its mean or its standard deviation are constant along the whole process. 

Formally it can be seen that stationarity has to do with the joint distribution of the process as it evolves in time. Then we say that a series is **strictly stationary** if 

\begin{equation}
P(x_{t_1}\leq c_1,\dots x_{t_n}\leq c_n)=P(x_{t_1+h}\leq c_1,\dots x_{t_n+h}\leq c_n)
\end{equation}

however, requiring this condition is too much for most applications


and joint distributions can be described as dependent or independent. In this context we can find

  * **(Highly) Persistent** (strongly dependent) time series, when an infinitesimal shock affects future time values. The longer the persistence, the longer the *memory* of the system.
  * **Weakly Dependent** time series, when both, the mean value is constant and does not depend on time and the autocovariance function depends only in the difference between the time moments, $|s-t|$, and not on the location of the two points. 

From now on, whenever we say *stationary* we actually mean *weakly dependent*. 

For our case before of the three-point moving average white noise, we can find that the mean is 0, because it is white noise and the moving average does not change this, while the autocorrelation function is

\begin{equation}
\rho_v(h)=\left\{\begin{array}{ll} 1, & h=0\\\frac{2}{3},  & h=\pm 1\\\frac{1}{3},  & h=\pm 2\\0, & |h|>2\end{array}\right.
\end{equation}

which only depends on the lag. We can plot it as follows

acf = pd.DataFrame({'ACF': [0, 1/3, 2/3, 1, 2/3, 1/3, 0],
                    'LAG': [-3, -2, -1, 0, 1, 2, 3]})

plt.bar(acf['LAG'], acf['ACF'], width = 0.05)
plt.title("3-Point Moving Average Autocorrelation", fontsize = 20)
plt.xlabel("LAG", fontsize = 15)
plt.ylabel("ACF", fontsize = 15)
plt.show()

Let's work out our example of the Flight Passengers. To see if our dataset describes a stationary process, we can just plot the number of passengers as a function of time

plt.plot(mydata['#Passengers'])

plt.xlabel('Year', fontsize = 15)
plt.ylabel('Number of Passengers', fontsize = 15)
plt.title('Passengers per Year', fontsize = 20)
plt.show()

From where we may inmediately see that the values do not distribute around a constant average number of passengers, but it shows a general increasing trend. There are ways to actually test if there is stationarity or not, for example the Dickey-Fuller test, where the null hypothesis is **non-stationarity**

print('Results of Dickey-Fuller Test:')
dftest = adfuller(mydata['#Passengers'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p-value','Number of Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

we obtain a p-value of $0.99$ which implies that we fail to reject the null hypothesis and conclude that there is no evidence against non-stationarity.

Graphically we may compute the **Moving Averages** and/or the **Moving Standard Deviations** which are procedures to smooth-out the time series that let us see how the data behaves in order to see if it is stationary or not. Let's compute these two quantitites using a window of $12$ observations, i.e. the equivalent to one year

m_aver = mydata.rolling(12).mean()
std_aver = mydata.rolling(12).std()

Now we can plot the three different time series together in one single graph using

orig = plt.plot(mydata, color='blue',label='Original')
mean = plt.plot(m_aver, color='red', label='Moving Average')
std = plt.plot(std_aver, color='black', label = 'Moving Standard Deviation')
plt.legend(loc='best')
plt.title('Moving Average & Standard Deviation - Original Data')
plt.show(block=False)

From the graph we can inmediately see the same behaviour we found with the test, a general tendency to increase the number of passengers and a slightly changing variability.

## Times Series as a Linear Model

Let's begin the analysis by dealing with the time series variable as if we were using a linear model. In this case all the particularities of the series (trend and seasonality) may be described using a series of categorical variables.

In the plot before we identify some of the basic components of a time series: trend and seasonality. To deal with them and remove their non-stationarity we have to understand these components a bit more in depth

### Trend

**Trend** is the general idea that the series have a common tendency over the time. The simplest way to deal with trends is just include a time variable (a sort of index) in our model. As we know, we may be involved in a **linear** or an **exponential** scheme in which case we must be careful with the interpreations. The linear case is
 
 \begin{equation}
 y_t = \alpha_0 + \alpha_1 t + u_t
 \end{equation}
 
 such that the **slope** measures the arithmetic change in the response from one time period to the next one, holding everything else fixed. In an exponential behaviour we will write the response as
 
 \begin{equation}
 \log(y_t) = \alpha_0 + \alpha_1 t + u_t
 \end{equation}
 
 where now the **slope** measures the growth rate from one time period to another.
 
 Note that **NOT** including the trend time variable may lead to the discovery of spurious correlations just because the variables included in the model do follow the same trend. 
 
 Also, note that there are trends that are not so straightforward to detect and remove (we do not just see a steady constant evolution of the time series) and then we must deal with, for example, quadratic trends

 \begin{equation}
 \log(y_t) = \alpha_0 + \alpha_1 t + \alpha_2t^2+ u_t
 \end{equation}
 
 where we capture the not constant elasticity of the response, or any other functional form.
 
 In general, the trend variable must be understood as containing all the effects correlated with the response that we do not consider in the regressors explicitely included in the model and that affect its behaviour.

#### Spurious Correlation: an Example

As a side example, take the following case: the data set we are going to use is that of the housing investment in the USA from 1947 to 1988

housing = pd.read_csv('/content/mydrive/My Drive/IEXL - Bootcamp - Math&Stats 20-21 September/data/hseinv.csv')
housing.head()

Let's take a look at the shape of the log transformed investment and price variables

fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(housing.linvpc, color = 'blue', label = 'Log(investment)')
ax2.plot(housing.lprice, color = 'red', label = 'Log(price)')

ax.set_xlabel('Years')
ax.set_ylabel('Log(investment)')
ax2.set_ylabel('Log(price)')

plt.show(block=False)

from the graph we may identify that there exists some degree of positive correlation between both variables since they seem to behave more or less in the same way, i.e. when one increases the other increases with some minor differences in certain periods.

This may lead us to find the model

\begin{equation}
\log(\text{investment}) = \beta_0 + \beta_1\,\log(\text{price})
\end{equation}

as

model = ols('linvpc ~ lprice', data = housing).fit()
print(model.summary())

which allow us to write the model as

\begin{equation}
\log(\text{investment}) = -0.5502 + 1.2409\,\log(\text{price})
\end{equation}

with an adjusted $\bar R^2 = 0.189$. Then even though the explanatory power of the model is only the 18.9%, we can identify that the price variable is actually individually relevant and that it has a positive impact in the increase of the housing price with respect to the investment, in fact, for each 1% increase in the price, the investment increases 1.2409%.

Now, let's see what happens when we add the time variable to our model. For that we first create the time order variable as

housing['time'] = housing.index + 1
housing.head()

and now we run the second model as

model2 = ols('linvpc ~ lprice + time', data = housing).fit()
print(model2.summary())

which allow us to write the model as

\begin{equation}
\log(\text{investment}) = -0.9131 -(0.3810)\,\log(\text{price}) + 0.0098\,t
\end{equation}

with an adjusted $\bar R^2 = 0.307$. Then the explanatory power of the model has increased but there are two major changes: the price variable has become individually irrelevant to explain the behaviour of the investment, but even more, its impact has changed the sign, so now we may say that for each 1% increase in the price, the investment decreases 0.381%.

### Seasonality

**Seasonality** is the idea that some property of the time series is repeated in constant steps, for example a decrese in unemployment or an increase in the number of passengers in summer. To deal with seasonalities, we include dummy variables for the different "seasons", known as **seasonal dummy variables** we are interested on, for example
 
 \begin{equation}
 y = \beta_0 + \beta_1\,x_1 + \beta_2\,x_2+\delta_1\,\text{summer}+\delta_2\,\text{winter}+\delta_3\,\text{spring}
 \end{equation}


Many time series exhibit a clear correlation in the evolution of their time series, see [this](https://www.tylervigen.com/spurious-correlations) link for some examples that show that in most of these cases what we are actually seeing is a spurious correlation.

### Removing Trend and Seasonality

There are many different procedures to remove trends and seasonalities from a time series. Here we are only going to consider two of them:

 * Analytic method: here we explicitely add the trend variable (as mentioned above)
 * Differences method: considering lagged variables

Appart from these, we will see how to find the decomposition of the time series on its basic components.

#### Analytic method: The linear model

As we have already mentioned, the trend variable must be understood as containing all the effects correlated with the response that we do not consider in the regressors explicitely included in the model and that affect its behaviour. Then, once we add this variable we are **detrending** the time series. The interpretation of this detrending is the following. Suppose we have the estimated detrended model

\begin{equation}
\hat y_t = \hat\beta_0 + \hat\beta_1\,x_{1,t} + \hat\beta_2\,x_{2,t} + \hat\beta_3\,t
\end{equation}

The estimation of the coefficients in this detrended time series can be obtained in the following way. 

  * First regress all the variables involved in the model on a constant and the time only and obtain the residuals for each of these models (in our case we only have the response, so we may just need one set of residuals). The residuals must be thought as being *linearly detrended*, since they are
  
  \begin{equation}
  \hat e_{y,t} = y_t - \hat\alpha_0 - \hat\alpha_1t 
  \end{equation}
  
  i.e. we are substracting the trend.
  
  * Next we run the regression of the residuals on the residuals (without constant term now). The estimates of the slopes of this model are precisely
  
  \begin{equation}
  \hat e_{y,t} = \hat\beta_1\hat e_{x_1,t} +\hat\beta_2\hat e_{x_2,t} + \dots
  \end{equation}

How should we understand this result? First, we conclude that if the trend variable is relevant in our model, we should not trust the results without the trend. Second, that as soon as any of th independent variables shows is trending we should add the trend even if the response is not trending. Third, that if we do not add the trend we may see that one or more variables are related to the response wimply because they exhibit a trend.

#### Differences method

This section let us introduce an operator, the **back-shift** operator by considering a method that is specific to Time Series: the method of the differences. 

The *back-shift operator*, $B$, is defined as

\begin{equation}
B^kx_t= x_{t-k}
\end{equation}

i.e. it is the operator that returns the lagged variable of order $k$. Then it can be used to define the **differences of order $d$** as

\begin{equation}
\nabla^d = (1-B)^d
\end{equation}

The first difference is an example of what is known as a **linear filter**, applied to remove the trend of a series. Then for a model with trend

\begin{equation}
x_t = \mu_t + y_t
\end{equation}

where $y_t$ is a stationary process, it can be proved that 

\begin{equation}
\nabla x_t = x_t - x_{t-1}
\end{equation}

has an autocovariance depending only on the lags and is, therefore, stationary.

Let's apply this method to the Passengers dataset, however, since from the graph of our original series we can detect an exponential scheme, let's first add the logarithm of the time series

mydata['lpassengers'] = np.log(mydata['#Passengers'])

If we plot this logarithmically transformed variable

plt.plot(mydata['lpassengers'])

plt.xlabel('Year', fontsize = 15)
plt.ylabel('Log(Number of Passengers)', fontsize = 15)
plt.title('Passengers per Year (Additive)', fontsize = 20)

plt.show()

we see that, as promised, once we take the logs and penalyze the large and small values of the time series, the scheme is arithmetic.

In the **differences** method we consider the difference between the time series at time $t$ and at time $t-1$, which gives rise to the following time series

\begin{equation}
z_t = y_{t} - y_{t-1}
\end{equation}

The idea is that now we can see if $z_t$ increases, decreases or is ocillating. In the last case, we may say that we have detrended the series. If we were in the first case then we should find the differences of $z_t$ and $z_{t-1}$ and so on.

This method is not free from problems:

  * we lose observations, and
  * since we do not identify the trend, forecasting is rather difficult after it

difference = mydata['lpassengers'] - mydata['lpassengers'].shift()

just as before we can plot the moving averages and moving standard deviations along with the difference time series

dm_aver = difference.rolling(12).mean()
dstd_aver = difference.rolling(12).std()

orig = plt.plot(difference, color='blue',label='Original')
mean = plt.plot(dm_aver, color='red', label='Moving Average')
std = plt.plot(dstd_aver, color='black', label = 'Moving Std')
plt.legend(loc='best')
plt.title('Moving Average & Standard Deviation')
plt.show(block=False)

from where we see a reasonably constant behaviour of the series. However, we can perform the formal test. Then let's use the Dickey-Fuller test for this difference as

print('Results of Dickey-Fuller Test:')
dftest = adfuller(difference.dropna(), autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', 'Number of Lags Used', 'Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

from where we see that for a 10% of significance level, we can't detect any non-stationarity.

## Scheme Decomposition

Another approach that we can use is to model the **decomposition** of the time series on its basic components, i.e. in the trend, seasonality and residuals. For that we instantiate a decomposition of the logarithmic data

decomposition = seasonal_decompose(mydata['lpassengers'])

from this model decomposition we can obtain the **trend**, the **seasonality** and the **residuals**

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

which can be represented as

plt.suptitle('Model Decomposition', fontsize = 20)

plt.subplot(411)
plt.plot(mydata['lpassengers'], label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')

plt.tight_layout(rect = (0,0,1,0.94))
plt.show()

Now, since the conditions are on the residuals of this decomposition, then let's test for non-stationarity in the residuals of the model

print('Results of Dickey-Fuller Test:')
dftest = adfuller(residual.dropna(), autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', 'Number of Lags Used', 'Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

## Box-Jenkins Methodology

Let's now make a summary of the general Box-Jenkins methodology for ARIMA models:

 * **Collect the data**: Box and Jenkins recommend a minimum of 50 observations and, when working with monthly series have between 6 and 10 years of information (similar bounds for other seasonal periods).
 * **Represent the series**: Always make the graphical representation of the series. This may let us identify different properties as schemes, trends or stationarity. An interesting tool is that of the moving averages or standard deviations to detect if the statistical properties are changing with time.
 * **Transform the series**: If the series is not stationary or we detect a precise scheme we should always consider the transformation of variables. Typically into logarithms, but we can do a more general Box-Cox type transformation.
 * **Remove trends**: By decomposition or by direct observation of the graph we can detect the existence and type of a trend. There are different ways to remove it: differences or analytically. In the ARIMA model we can consider that if the trend is **linear** then we take d=1 and for a non-linear trend we use d=2 (the python function does not allow for higher values of d).
 * **Identify the model**: Identify the order of the autorregressive and moving averages of the regular and seasonal components. In absence of any other guide, take $(p, q)\in[1,2]$ with all their combinations. These values should be found by means of the autocorrelation or partial correlation functions and graphs (take a look at the **acf** and **pacf** functions from **statsmodels.tsa.stattools** module as well as to the **plot_acf** and **plot_pacf** from statsmodels.graphics.tsaplots) as well as by minimization of the different information criteria. If we have doubts with respect to which model we should take, estimate some of them and compare later (with respect to this, see if the ARIMA(2,1,2) would be better than the ARIMA(3,1,3) we have found before)
 * **Estimate the coefficients**: Usual estimation process.
 * **Validate the model**: There are many different tests that can be made to validate the model. One option similar to one that we have covered in linear regression is that when the model does not actully fit the observed series, the residuals will be autocorrelated values. In this case we should use **Durbin-Watson** for first order autocorrelation or **Wallis** for fourth order autocorrelations.
 * **Analyize the errors**: This is the usual residuals study, mostly looking for high residual points (outliers) where the difference between the prediction and the observed values is too different
 * **Select the model**: In view of the previous steps, select one of the models.
 * **Forecasting**: Once we have the model we can use it as the basis for forecasting.

from where we inmediately see that the p-value is significant for a $10^{-6}\%$ and we can then assume non-stationarity.

## Predictions

One of the key points in any modelization of a time series is to have the ability to predict future values from known data. In this context we are going to see two different approaches

  * ARIMA models, developed by Box and Jenkins, whose name means Autorregresive (AR) Integrated (I) Moving Averages (MA) where we use the estimation of an explicit linear model under the idea that it should contain all the needed elements but only the minimum of them (with the recommendation of at least 50 observations)
  
  * Facebook's Prophet, developed by Facebook, using a Bayesian approach.

## ARIMA models

The **ARMA** and **ARIMA Models**, developed by Box and Jenkins, where the names stand for *AutoRegressive Moving Averages* and *AutoRegressive Integrated Moving Averages*, are weakly dependent models we will use for forecasting. Let's study each of the components separatedly.

### MA(q)

To write the general form of the **Moving Average processes of order q**, MA(q) or ARIMA(0,0,q), we are going to introduce the **moving average operator**

\begin{equation}
\alpha(B)=1+\alpha_1B+\alpha_2B^2+\dots\alpha_qB^q
\end{equation}

then the model can be written generally as

\begin{equation}
y_t = \alpha(B)e_t = e_t - \alpha_1\,e_{t-1} - \alpha_2\,e_{t-2} -\dots - \alpha_p\,e_{t-p}
\end{equation}
  
in particular, for $q=1$, the model MA(1) or ARIMA(0,0,1) is

\begin{equation}
y_t = e_t + \alpha\,e_{t-1}
\end{equation}
  
where the moving averages are lagged forecast errors in the prediction equation. The number $q$ represents the number of terms to be included in the equation (the equation we have written is a case for MA(1). This model MA(1) is a stationary, weakly dependent sequence).
  
  For our log of the number of passengers we can find the MA(1) model as

ma_model = ARIMA(mydata['lpassengers'], order=(0, 0, 1)).fit()  
print(ma_model.summary())

We can plot the previously detrended difference time series with the fitted values of this model to compare the results

plt.plot(difference, color = "DeepSkyBlue", label = 'Difference')
plt.plot(ma_model.fittedvalues, color='OrangeRed', label = 'MA Fitted')
plt.legend(loc = 'best')
plt.xlabel('Year', fontsize = 15)
plt.title('Detrended Series (MA(1))', fontsize = 20)
plt.show()

from where we see a close behaviour even though different in magnitude

### AR(p)

These **autoregressive processes of order p**, AR(p) or ARIMA(p,0,0), models are written generally as

  \begin{equation}
  x_t = \rho_1 x_{t-1} + \rho_2 x_{t-2} + \dots+ \rho_q x_{t-q} + e_t
  \end{equation}
  
  and for $p=1$ we have the AR(1) or ARIMA(1,0,0) as
  
  \begin{equation}
  x_t = \rho x_{t-1} + e_t
  \end{equation}
  
  where the autoregressive terms are lags of the dependent variable. The number $p$ represents, again the number of terms to be included in the equation, so the previous is an AR(1) model.

ar_model = ARIMA(mydata['lpassengers'], order=(1, 0, 0)).fit()
print(ar_model.summary())

plt.plot(difference, color = "DeepSkyBlue", label = 'Difference')
plt.plot(ar_model.fittedvalues, color='OrangeRed', label = 'AR Fitted')
plt.legend(loc = 'best')
plt.xlabel('Year', fontsize = 15)
plt.title('Detrended Series (AR(1))', fontsize = 20)
plt.show()

If $\rho=1$ this model is known as a **random walk**

### ARMA(p,q)

With the two pieces we have just seen we may find the **ARMA** model (without the Integrated component), so that ARMA(p,q) is equivalent to ARIMA(p,0,q). For us this would simply mean that we put the integrated value to 0, then

arma_model = ARIMA(mydata['lpassengers'], order=(1, 0, 1)).fit()
print(arma_model.summary())

we may see a graphical comparison of the three models

plt.plot(ar_model.fittedvalues, color='blue', label = 'AR(1) model')
plt.plot(ma_model.fittedvalues, color='red', label = 'MA(1) model')
plt.plot(arma_model.fittedvalues, color='green', label = 'ARMA(1,1) model')

plt.title('Comparison of Models', fontsize = 20)
plt.xlabel('Year', fontsize = 15)
plt.legend(loc = 'best')
plt.show()

We see that the three of them give the same behaviour but AR and ARMA are closer to each other. In fact, if the model is AR or ARMA, the ACF does not tell us too much about the dependency. In this case we need to introduce the **Partial Autocorrelation Function**, PACF, defined as

\begin{equation}
\rho_{xy|z} = corr(x - \hat x, y - \hat y)
\end{equation}

where we compute the correlation between $x$ and $y$ with the linear effect of $z$ removed. note that $\hat x$ and $\hat y$ are the regressions of $x$ and $y$ on $z$.

### ACF and PACF Plots

There is a series of plots made from the **Autocorrelation Function** and the **Partial Autocorrelation Function** that will be of help when we want to decide the order of the model. These functions (and the plots) are found for an arbitrary large number of lags. The behaviour of these functions is the following

<br>
<table>
<tr>
<th></th><th> AR(p) </th><th> MA(q) </th><th> ARMA(p,q) </th>
</tr>
<tr>
<td> ACF </td><td> Tails Off </td><td> Cuts off after lag q </td><td> Tails off </td>
</tr>
<tr>
<td> PACF </td><td> Cuts off after lag p </td><td> Tails off </td><td> Tails off </td>
</tr>
</table>
<br>

plt.figure(figsize = (12,6))
plt.subplot(211)
plot_acf(mydata['lpassengers'], lags = 80, ax = plt.gca())
plt.ylabel('ACF')

plt.subplot(212)
plot_pacf(mydata['lpassengers'], lags = 80, ax = plt.gca())
plt.ylabel('PACF')
plt.xlabel('Lags', fontsize = 15)

plt.tight_layout(rect = (0,0,1,0.94))
plt.show()

In these graphs we see that the autocorrelations are significant for a high number of lags (13!!) However, from the PACF we see that probably this is due to propagation of autocorrelation, and only the 2 first lags seem relevant, and then it just cuts off. This seems to imply that a good model is just **AR(1)**!! (plus a difference of order 1, note however, that the MA(1) coefficient in the model was actually relevant)

There is, however, a function in the `pmdarima` package that is very helpful since it iterates over all the possible models and tells you which is the optimum one. The instruction would be (Try it!!)

```python
from pmdarima.arima import auto_arima
Arima_model = auto_arima(mydata['lpassengers'], 
                       start_p=1, 
                       start_q=1, 
                       max_p=8, 
                       max_q=8, 
                       start_P=0, 
                       start_Q=0, 
                       max_P=8, 
                       max_Q=8, 
                       m=12, 
                       seasonal=True, 
                       trace=True, 
                       d=1, 
                       D=1, 
                       error_action='warn', 
                       suppress_warnings=True, 
                       random_state = 20, 
                       n_fits=30)
print(Arima_model.summary())
```

### I(d)

The **Integrated processes of order d**, I(d) or ARIMA(0, d, 0), is a time series that becomes white noise (pure random process) once it is differentiated $p$ times. It may be written generally as
  
  \begin{equation}
  (1-B)^dx_t =  e_t
  \end{equation}
  
  the case of I(1) corresponds to the usual difference time series, since $B$ is the back-shift operator, then
  
  \begin{equation}
  (1-B)x_t = x_t - x_{t-1} = e_t
  \end{equation}
  
  this is why we where comparing the different AR, MA and ARMA models with the differences. 
  
Altogether we can write the equation of a general ARIMA(p,q,d) model as

\begin{equation}
\left(1-\sum_{i=1}^p\rho_iB^i\right)(1-B)^d y_t = \left(1-\sum_{i=1}^q\alpha_iB^i\right)e_t
\end{equation}

where we have used the decomposition of the $y_t$ variable only, but we can always add any number of independent variables to the right hand side of this equation.

The ARIMA(1,1,1) model for our case before is

arima_model = ARIMA(mydata['lpassengers'], order=(1, 1, 1)).fit()
print(arima_model.summary())

note that the impact of the AR coefficient changes completely (even its sign). Let's write the algebraic equation in view of the coefficients:

\begin{equation}
\text{lpass}_t + 0.4174\,\text{lpass}_{t-1} -0.5826\,\text{lpass}_{t-2} = 0.0098 + \hat e_t - 0.8502\,\hat e_{t-1}
\end{equation}

where we have made $\rho_1 = -0.5826$, $\alpha_1=0.8502$ and $p$, $d$ and $q$ equal to 1. We can now represent this model against the difference time series as

plt.plot(difference, color = "DeepSkyBlue", label = 'Difference')
plt.plot(arima_model.fittedvalues, color='OrangeRed', label = 'ARIMA Fitted')

plt.legend(loc = 'best')
plt.xlabel('Year', fontsize = 15)
plt.title('Detrended Series (ARIMA(1,1,1))', fontsize = 20)

plt.show()

This is clearly not a close model. Then the usual question that arises is how can we choose the values of $p$, $d$ and $q$. We cannot go into the details since we have not defined in any moment of the course the different **Information Criteria**, but if we write the **Akaike Information Criterion** (AIC) as

\begin{equation}
AIC = -2\log(L) + 2(p+d+q) 
\end{equation}

where $L$ is the log likelihood of the data, i.e. the logarithm of the probability of the observed data coming from the estimated model. We may choose the values of the iterations by imposing that this quantity is a minimum, i.e.

\begin{equation}
(p,d,q) = \text{argmin}(AIC)
\end{equation}

In our case we see that ARIMA(3,1,3) has a lower AIC than ARIMA(1,1,1) and then would be a better model.

arima313_model = ARIMA(mydata['lpassengers'], order=(3,1,3)).fit()
print(arima313_model.summary())

from the plots

plt.plot(difference, color = 'SkyBlue', label = 'Difference')
plt.plot(arima_model.fittedvalues, color='OrangeRed', label = 'ARIMA(1,1,1)')
plt.plot(arima313_model.fittedvalues, color='ForestGreen', label = 'ARIMA(3,1,3)')

plt.xlabel('Year', fontsize = 15)
plt.legend(loc = 'best')
plt.show()

we see that, effectively, the ARIMA(3,1,3) is closer to the differences than the ARIMA(1,1,1).

### ARIMA Predictions

The last graph showed that the model follows reasonably well the detrended model, then we can try to find the predictions of the model in the usual way from the fitted values

predictions_ARIMA = pd.Series(arima313_model.fittedvalues, copy=True)
print(predictions_ARIMA.head())

however, we see that it does not begin exactly in the same time as the original series, but one month later due to the 1-lag we have used in the shift method. Now, to obtain the predictions on the oringal (exponentiated) variable, we can do the following:

  * First consider the cumulative sum of the predictions
  * Add to the previously defined *lpassengers* variable this cumulative sum
  * Take the exponential of this series

predictions_ARIMA_cumsum = predictions_ARIMA.cumsum()
predictions_ARIMA_log = pd.Series(mydata['lpassengers'].iloc[0], index = mydata['lpassengers'].index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_cumsum, fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(mydata['#Passengers'], label = 'Original')
plt.plot(predictions_ARIMA, label = 'ARIMA(3,1,3)')

plt.xlabel('Year', fontsize = 15)
plt.ylabel('Passengers', fontsize = 15)
plt.title('Number of Passengers per Year', fontsize = 20)
plt.legend(loc = 'best')

plt.show()

We see that the prediction follows the known time series rather closely.

## SARIMA Models

As we have seen, the ARIMA models do not consider the seasonality questions. There is, however, an extension of this model that allow us to consider it: the SARIMA models (Seasonal AutoRegressive Integrated Moving Averages), whose implementation in Python is done with the `SARIMAX` function in `statsmodels`. 

Now, apart from the usual three parameters $(p,d,q)$ we also have:

 * **P**: seasonal autoregressive order
 * **D**: seasonal differences order
 * **Q**: seasonal moving average order
 * **m**: the number of steps in one single period

The way to decide this parameters follows closely the ACF and PACF arguments, but now it must be done with the deseasonalized series: Then we should find the differences of a shift of 12 in the log transformed variables differences of a shift of 1 (**we leave this as an exercise**). In this case we will use $P=0$, $D=1$, $Q=1$ and $m=12$ in an ARIMA(1,1,1)

sarima011 = sm.tsa.SARIMAX(mydata['lpassengers'], order = (0,1,1), seasonal_order = (0,1,1,12)).fit()
print(sarima011.summary())

In this case there is a function to have some relevant plots for the diagnostics of the model: `plot_diagnostics`

sarima011.plot_diagnostics()
plt.tight_layout(rect = (0,0,1,0.94))
plt.show()

From these graphs we see that the residuals of the model behave as they should.

prediction = pd.DataFrame(sarima011.predict(n_periods=185), index = mydata['lpassengers'].index)
prediction.columns = ['Number of Passengers']

plt.figure(figsize=(15,10))
plt.plot(mydata['#Passengers'], label='Original')
plt.plot(np.exp(prediction), label='SARIMA(0,1,1)(0,1,1)12')

plt.legend(loc = 'best')
plt.show()

Apart from the strange prediction in 1950, we see that this model follows very closely the original series. Way better than a pure ARIMA.

## Facebook's Prophet

The first thing we need is the library installed (follow the instructions in their webpage), then we call it as

!pip install fbprophet

from fbprophet import Prophet

Now we create an instance of the Prophet model object

prophet_model = Prophet(interval_width=0.95)

and fit it using the original dataset (be careful with not include the column with the log transformed series)

df = pd.read_csv('/content/mydrive/My Drive/IE Bootcamp - Math & Stats /data/AirPassengers.csv')
df['Month'] = pd.DatetimeIndex(df['Month'])

df = df.rename(columns = {'Month': 'ds',
                          '#Passengers': 'y'})
df.head()

prophet_model.fit(df)

the forescast procedure needs a dataframe with a date series where we will store the predictions (mean value and confidence intervals, see that the model has been instanciated with a 95% interval width)

future_dates = prophet_model.make_future_dataframe(periods=12, freq='MS')
future_dates.head()

Now we can find the predictions of the model in the usual way (predict function)

forecast = prophet_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

To see the predictions, we can just make a plot of them

prophet_model.plot(forecast,
                   uncertainty=True)

We could make a joint plot with the original time series, the ARIMA model and the Prophet model in order to compare the performance of both methods

forecast = forecast.set_index('ds')

plt.plot(mydata['#Passengers'], label = 'Original')
plt.plot(np.exp(prediction), label = 'SARIMA(0,1,1)(0,1,1)12')
plt.plot(forecast['yhat'], label = 'Prophet')

plt.legend(loc = 'best')
plt.show()

From the graph we see that both models make a good work in fitting the original data, although the SARIMA is closer to the dataset values.