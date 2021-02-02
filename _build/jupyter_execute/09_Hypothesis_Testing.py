# Hypothesis Testing

### Prof. Dr.Juanjo Manjarín
**Statistics & Data Analysis**


---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import statsmodels.stats.weightstats as smw
import statsmodels.stats.proportion as smp

plt.style.use("seaborn")

from google.colab import drive
drive.mount('IEStats')

In this case we do not look for a range in which we may be confident that our parameter will be, but conditions that may allow us to say that our sample gives support to a hypothesis or to an alternative hypothesis.

## <font color="Blue">Decision Schemes</font>

In general we will have a scheme such that there are only two hypothesis:

  * The **Null Hypothesis**, denoted as $H_0$, it is the statement that is most costly to wrongly reject. Most of our arguments will be around this hypothesis which is **assumed to be true**
  * The **Alternative Hypothesis**, denoted as $H_1$, we will take it as the complementary hypothesis to $H_0$, although it should be clear that this is not a neccesary general condition and will be relaxed once we deal with the power of the test.

Note, that these definitions lead to a general idea: *All the content of the hypothesis is contained in $H_1$*, i.e. if for example we want to test if a business is going to be profitable, the decision scheme is

\begin{equation}
H_0:\{\text{it is not profitable}\},\quad H_1:\{\text{it is profitable}\}
\end{equation}

and so in any other test we want to make. Then it is clear that the question we want to answer with our data must be very clear

According to these hypotheses and the meaning of statistical hypothesis, we can split the whole sampling space into two different regions:

  * The **Critical Region**: Denoted as $C$, is the region such that if the contrast statistic is in it, we say that there is strong evidence that supports the alternative hypothesis and then we **reject the null hypothesis**
  * The **Acceptance Region**: Denoted as $C^∗$, we will **fail to reject the null hypothesis** if the contrast statistic lies in it since there is no strong evidence that supports the alternative.
  
It is import to make emphasis on the language: we reject or fail to reject the null hypothesis but never accept. The reason for this is at least twofold:

 * First, that we assume that the null hypothesis is true then there is no reason to accept it
 * Second, that all the content is in the alternative hypothesis then either find evidence to reject our assumption or do not find it (fail to reject).

## <font color="Blue">Types of Errors</font>

If we consider that our decision may be right or wrong, we can write the following table

<br>

<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th, td {
  padding: 0px;
  text-align: center;    
}
</style>

<table style="width:100%">
  <tr>
    <th></th>
    <th></th>
    <th colspan="2">Decision</th>
  </tr>
  <tr>
    <th></th>
    <th></th>
    <th> Reject $H_0$ </th>
    <th> Fail to Reject $H_0$ </th>
  </tr>
  <tr>
    <th rowspan="2">True</th>
    <td>$H_0$</td>
    <td> <font color = "Red"> Type I Error </font></td>
    <td></td<td> <font color = "Green"> Right </font> </td>
  </tr>
  <tr>
    <td> $H_1$</td>
    <td> <font color = "Green"> Right </font> </td>
    <td> <font color = "Red"> Type II Error </font> </td>
  </tr>
</table>

<br>


Then we will say that there are two different types of errors depending on our conclusions once we perform the test:

 * A **ype I Error** occurs when we reject the NULL hypothesis but this is true, i.e. our sample value lies in the critical region but $H_0$ is the true statement: $\{X \in C|H_0\}$
 * A **Type II Error** occurs whe we fail to reject the NULL hypothesis but the true statement is $H_1$, i.e. the sample value lies in the acceptance region but $H_1$ is true: $\{X \in C^*|H1\}$

In any other circumstance  our decision is right. In this context we define

  * The **Significance Level** is the probability of a Type I error

\begin{equation}
\alpha = P(\text{Type I error}) = P(X \in C|H_0)
\end{equation}

In other words, it is the probability of a False Positive, i.e. the probability of rejecting $H_0$ being true. See that it is complementary
to the **Precision**.

  * The **Power of the test** is the complementary to a Type II error. If we denote as $\beta$ the probability of a Type II error, then

\begin{equation}
\text{Power} = 1 − \beta = P(\text{Reject $H_0$ being false}) = P(X\in C|H_1)
\end{equation}

Then, if $\beta$ is the probability of failing to reject $H_0$ being false, the power of a test is the probability of rejecting $H_0$ being false

# <font color="Red">Types of Tests</font>

Now we focus on whether or not we find evidence against a basic assumption, which is called **NULL hypothesis**, and is always assumed to be true.

Since it is assumed true, whatever we want to test will be in $H_1$, the **alternative hypothesis**. Then having $H_0$ and $H_1$ defines what we know as the **decision scheme**

This implies that we can have three different types of test:

 * **Right-tailed test**, if the alternative hypothesis is answering the question: *Is it greater than...?* 
 * **Left-tailed test**, if the alternative hypothesis is answering the question: *Is it smaller than...?* 
 * **Two-tailed test**, if the alternative hypothesis is answering the question: *Is it not...?* 

Of course, the $H_0$ is the complementary to this $H_1$.

All the tests need a parameter which is known as $\mu_0$, when we test the mean of a population or $P_0$, when we test the proportion of a population.

Then we can write the **right-tailed tests** as

\begin{equation}
H_0:\{\mu\leq \mu_0\}, \quad H_1:\{\mu > \mu_0\}
\end{equation}

the **left-tailed test** as

\begin{equation}
H_0:\{\mu\geq \mu_0\}, \quad H_1:\{\mu < \mu_0\}
\end{equation}

and the **two-tailed tests** as

\begin{equation}
H_0:\{\mu = \mu_0\}, \quad H_1:\{\mu \neq \mu_0\}
\end{equation}

## <font color ="Blue"> Tests on the Mean of a Normal Population</font>

We have exactly the same cases that we saw in confidence intervals:

 * **Population variance known**, in which case we can use the normal distribution ($z$)
 * **Population variance unknown and a large sample** ($n>40$), in which case we can use the normal distribution, but we use the sample standard deviation in the computation
 * **Population variance unknown and a small sample** ($n<40$), in which case we use the t-distribution, and we use the sample standard deviation in the computation

## <font color="Blue">Tests on the Proportion of a Population</font>

In this case we always take the normal approximation to the binomial distribution. This is possible only when

\begin{equation}
np>10,\quad nq=n(1-p)>10
\end{equation}

in this case we ONLY use the normal distribution

# <font color="Red">Performing a Test</font>

There are two different procedures to be used:

 * Using a **test statistic**, which is the standarized value from our sample under the NULL hypothesis
 * Using a **p-value**, which is the probability associated with the test statistic, then it is the smallest significance level to reject the NULL hypothesis

A standarization is

\begin{equation}
x \longrightarrow z=\frac{x - \bar x}{s_x}
\end{equation}

in our case we standarize the sample mean, then we use the expected value and the standard deviation (standard error) of the sampling distribution of sampling means

\begin{equation}
\bar x \longrightarrow z=\frac{\bar x - \mu}{\sigma/\sqrt{n}}
\end{equation}

if we are in a hypothesis scheme, we are assuming that $\mu=\mu_0$, then we write the standarization as 

\begin{equation}
\bar x \longrightarrow z=\frac{\bar x - \mu_0}{\sigma/\sqrt{n}}
\end{equation}

the the statistic to be used in the test is

\begin{equation}
z_{stat}=\frac{\bar x - \mu_0}{\sigma/\sqrt{n}}
\end{equation}

where $\sigma$ is the population standard deviation. When we do not know it, we estimate it using the sample variance, then we have two other statistics to be used:

\begin{equation}
z_{stat}=\frac{\bar x - \mu_0}{s_x/\sqrt{n}},\quad\text{ large sample}
\end{equation}

and 

\begin{equation}
t_{n-1}^{stat}=\frac{\bar x - \mu_0}{s_x/\sqrt{n}},\quad\text{ small sample}
\end{equation}

note that the difference is not evident in the statistic, but in the distribution that we use in the test.

For the **Proportion of a normal population**, then the statistic is

\begin{equation}
z_{stat} = \frac{\hat p - P_0}{\sqrt{\frac{\hat p(1-\hat p)}{n}}}
\end{equation}

To perform the test we need a boundary that may let us decide if we reject the NULL or we fail to Reject the NULL. This value is known as **Critical value** and is the value such that either its cumulative or its survival probability is $\alpha$ (or $\alpha/2$ in a two-tailed test). Then

 * For a **right-tailed** test, the critical value is the *inverse survival function* for $\alpha$
 * For a **left-tailed** test, the critical value is the *percentile probability function* for $\alpha$
 * For a **two-tailed** test, the critical value is the *inverse survival function* for $\alpha/2$

then the test proceed as follows: We **Reject the NULL hypothesis** if the test statistic is larger than the critical value in the direction of the test.

# <font color="Red">Python Tests</font>

We, again, have two options:

 * Using the formula
 * Using `statsmodels`, which cannot be used in the case of a test for the mean of a normal population with known population variance.
  * For $\sigma$-unkonwn and a large sample: `ztest_mean()` which returns the z-statistic and its p-value
  * For $\sigma$-unkonwn and a small sample: `ttest_mean()` which returns the t-statistic, its p-value and the degrees of freedom ($n-1$)
  * For a population proportion: `proportions_ztest()` which returns the z-statistic and its p-value

## <font color="Blue">Examples</font>

Let's load the dataset and perform some tests

truancy = pd.read_excel('/content/IEStats/My Drive/IE - 2021 - Statistics and Data Analysis/DataSets/truancy.xlsx')
truancy.head()

### <font color="blue">Proportions</font>

**From the `truancy.xlsx` dataset, test if the average `prepct` for females is larger than 12.**

the decision scheme is

\begin{equation}
H_0:\{\mu\leq 12\},\quad H_1:\{\mu>12\}
\end{equation}

Since we do not know the population variance, and the sample size is

pre_f = truancy[truancy['gender'] == 'f']['prepct']
n = len(pre_f)

print('Sample Size: ', n)

this sample size is larger than 40, then we are allowed to use the normal approximation.

 * Using `statsmodels`:

# Data
mu0 = 12
SL = 0.05

# Descriptive Stats
des_f = smw.DescrStatsW(pre_f)
zstat, pval = des_f.ztest_mean(value = mu0, alternative='larger')

# Test
zcrit = ss.norm.isf(SL)

print('z-statistic: ', round(zstat,2))
print('z-critical:', round(zcrit,2))
if zstat > zcrit:
  print('Reject the NULL hypothesis')
else:
  print('Fail to Reject the NULL')

Since we Fail to Reject the NuLL hypothesis, we do not find evidence against $H_0$ for a 5% of significance level. Then we cannot say that the average `prepct` is larger than 12 for females

 * Using the formula

# Data
mu0 = 12
SL = 0.05

# Descriptive Stats
xmean = pre_f.mean()
stdev = pre_f.std()
zstat = (xmean - mu0)/(stdev/np.sqrt(n))

# Test
zcrit = ss.norm.isf(SL)

print('z-statistic: ', round(zstat,2))
print('z-critical:', round(zcrit,2))
if zstat > zcrit:
  print('Reject the NULL hypothesis')
else:
  print('Fail to Reject the NULL')

### <font color="blue">Proportions</font>

**Test for a 1% if the sample is not gender-balanced**

The decision scheme is then

\begin{equation}
H_0:\{P = 0.5\},\quad H_1:\{ P\neq 0.5\}
\end{equation}

i.e. assume a balanced dataset as one with a representation of the 50% for each category.

tr_fem = truancy[truancy['gender'] == "f"]

x = len(tr_fem)
n = len(truancy)
phat = x/n

NP = n*phat
NQ = n*(1-phat)

if NP >= 10:
  if NQ >= 10:
    print("We CAN use the Normal Approximation since np = {:<3.0f} and nq = {:<3.0f}".format(NP,NQ))
else:
  print("We CANNOT use the Normal Approximation since np = {:<3.0f} and nq = {:<3.2f}".format(NP,NQ))

# data
p0 = 0.5
SL = 0.01

# statsmodels
zstat, pval = smp.proportions_ztest(x, n, value = p0)

# Critical value
zcrit = ss.norm.isf(SL/2)

print('z-statistic: ', round(zstat,2))
print('z-critical:', round(zcrit,2))
if zstat > zcrit:
  print('Reject the NULL hypothesis')
else:
  print('Fail to Reject the NULL')

now using the formula directly

# data
p0 = 0.5
SL = 0.01

# formula
zstat = (phat - p0)/np.sqrt(phat*(1-phat)/n)

# Critical value
zcrit = ss.norm.isf(SL/2)

print('z-statistic: ', round(zstat,2))
print('z-critical:', round(zcrit,2))
if zstat > zcrit:
  print('Reject the NULL hypothesis')
else:
  print('Fail to Reject the NULL')

We obtain the same result in which we fail to reject the NULL, implying that for a 1% we do not find evidence against the hypothesis that the population of this study is balanced

Let's now use the **p-value**

# data
p0 = 0.5
SL = 0.01

# statsmodels
zstat, pval = smp.proportions_ztest(x, n, value = p0)

print('p-value: ', round(pval,4))
print('alpha:', round(SL,4))
if pval < SL:
  print('Reject the NULL hypothesis')
else:
  print('Fail to Reject the NULL')

# Data
p0 = 0.5
SL = 0.01

zstat = (phat - p0)/np.sqrt(phat*(1-phat)/n)
pval = 2 * ss.norm.sf(zstat)

print('p-value: ', round(pval,4))
print('alpha:', round(SL,4))
if pval < SL:
  print('Reject the NULL hypothesis')
else:
  print('Fail to Reject the NULL')

**It has been argued that the proportion of females is slightly greater, in fact, that it is 52%. Find the probability that you may detect this bias in the sample if it is actually true**

The decision scheme is then

\begin{equation}
H_0:\{P = 0.5\},\quad H_1:\{ P = 0.52\}
\end{equation}

# Data
p0 = 0.5
p1 = 0.52
SL = 0.01

# Standard Error
SE = np.sqrt(phat*(1-phat)/n)

# Critical Value
zcrit = ss.norm.isf(SL/2)

# Effect Size
size = (p0-p1)/SE

# Power
power = ss.norm.sf(size+zcrit) + ss.norm.cdf(size-zcrit)
beta = 1-power

power, beta

---
## <font color="slateblue">Waiting at the ER</font>

**ER wait times at a hospital were being analyzed. The previous year's average was 128 minutes. Suppose that this year's average wait time is 135 minutes. We would like to know whether this year average waiting time is just an "accident" and we  can still consider that the average wating time has not changed, or whether the average waiting time is now different from 128 minutes.**

  * **Provide the hypotheses for this situation in plain language**
  * **If we plan to collect a sample size of $n=64$, what values could $\bar{x}$ take so that we reject $H_0$? Suppose the sample standard deviation (39 minutes) is the population standard deviation. You may assume that the conditions for the nearly normal model for $\bar{x}$ are satisfied.**
  * **Calculate the probability of a Type 2 error.**

\begin{equation}
H_0:\{\mu = 128\},\quad H_1:\{\mu\neq 128\}
\end{equation}

# Data
mu0 = 128
n = 64
stdev = 39
SL = 0.05

# Margin of Error
zcrit = ss.norm.isf(SL/2)
ME = zcrit * stdev / np.sqrt(n)

lower = mu0 - ME
upper = mu0 + ME
lower, upper

# Power
mu1 = 135
low = (lower - mu1)/(stdev/np.sqrt(n))
up =  (upper - mu1)/(stdev/np.sqrt(n))
power = ss.norm.sf(up) + ss.norm.cdf(low)
print('\nNormal Approximation')
print('-'*20)
print('The power using the normal appoximation is: ', round(power, 4))

# Power using the t-distribution
mu1 = 135
low = (lower - mu1)/(stdev/np.sqrt(n))
up =  (upper - mu1)/(stdev/np.sqrt(n))
power = ss.t.sf(up, n-1) + ss.t.cdf(low, n-1)
print('\nt-Distribution')
print('-'*20)
print('The power using the t-distribution is: ', round(power, 4))

# Power using statsmodels
from statsmodels.stats.power import TTestPower
analysis = TTestPower()
powerTest = analysis.power(effect_size=(mu1-mu0)/stdev, nobs=n, alpha=SL, alternative='two-sided')

print('\nStatsmodels')
print('-'*20)
print('The Probability of a Type II error is the {:4.2%}'.format(1 - powerTest) +
      '\nThe Power of the Test is the {:4.2%}'.format(powerTest))