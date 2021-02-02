# <font color = "Red">Cases and Conditions</font> 
 
 * Estimation of the mean
  * Known the population variance: *Normal distribution*
  * Unkown population variance:
    * Small Sample: *t-Student*
    * Large Sample: *Approximate with Normal*
 * Estimation of the proportion: *Normal distribution*

Before any estimation you have to check:

 * **Independency**: Satisfied as long as the sample size is less than the 10% of the population size
 * **Normality**: 
  * For the **mean**: qqplot (normality plot), as long as the sample size is greater or of the order of 40/50
  * For a **proportion**: $np$ and $nq = n(1-p)$ are both greater than 10

<font color="red">Python Approach</font>

 * `statsmodels`: can only be used when you have a dataset
 * The formulas: can be used anytime

Which are the formulas?

* Mean of a normal population with known variance:

\begin{equation}
\mu \in \bar x\pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}
\end{equation}

This one **CANNOT** be done in statsmodels, onyl with the formula.

 * Mean of a normal population with unkown variance and small sample

\begin{equation}
\mu \in \bar x\pm t_{n-1,\alpha/2}\frac{s}{\sqrt{n}}
\end{equation}

 * Mean of a normal population with unkown variance and large sample

\begin{equation}
\mu \in \bar x\pm z_{\alpha/2}\frac{s}{\sqrt{n}}
\end{equation}

 * Porportion normal population

\begin{equation}
P \in \hat p \pm z_{\alpha/2}\sqrt{\frac{\hat p(1-\hat p)}{n}}
\end{equation}

# <font color="Red">Application</font>

import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.stats.weightstats as smw
import statsmodels.stats.proportion as smp
import matplotlib.pyplot as plt
import math

plt.style.use('seaborn')

from google.colab import drive 
drive.mount('IEStats')

anorexia = pd.read_excel('/content/IEStats/My Drive/IE - 2021 - Statistics and Data Analysis/DataSets/anorexia.xlsx')
anorexia.head()

Let's estimate for a 95% of CL the Pre-treatment weight.

First directly with the **formula**

prew = anorexia['prewt']
n = len(prew)

print('Since {:2.0f} > 40, we can use the normal approximation'.format(n))

# Sample information
xmean = prew.mean()
stdev = prew.std()
n = len(prew)

# Distribution value
CL = 0.95
SL = 1 - CL # alpha
zcrit = ss.norm.isf(SL/2)
# zcrit = -ss.norm.ppf(SL/2)

# Confidence Interval
ME = zcrit * stdev /np.sqrt(n)

lower = xmean - ME
upper = xmean + ME

# Print the output
print('The {:2.0%} CI for the weight before the treatment is [{:4.2f}, {:4.2f}]'.format(CL, lower, upper))

Let's now use `statsmodels`

des_prew = smw.DescrStatsW(prew)
lower, upper = des_prew.zconfint_mean(SL) # Note that the only argument needed is the "alpha"

# Print the output
print('The {:2.0%} CI for the weight before the treatment is [{:4.2f}, {:4.2f}]'.format(CL, lower, upper))

prew_1 = anorexia[anorexia['group'] == 1]['prewt'] # 1st is the filter, the second the selection
n1 = len(prew_1)

des_prew_1 = smw.DescrStatsW(prew_1)

if n1 > 40:
  lower_1, upper_1 = des_prew_1.zconfint_mean(SL)
else:
  lower_1, upper_1 = des_prew_1.tconfint_mean(SL)

print('The {:2.0%} CI of the group 1 pre-treatment weight is [{:4.2f}, {:4.2f}]'.format(CL, lower_1, upper_1))

## <font color="Blue">Proportions</font>

# Dataset for the patients who gained weight after the treatment
greater_we = anorexia[anorexia['difwt'] > 0]

# We want to find the proportion of patients who gained weight
x = len(greater_we['difwt']) # number of people with a higher weight
n = len(anorexia['difwt'])

phat = x/n

# Confidence Interval with the formula
np = n*phat
nq = n*(1-phat)

print('The values for the normality approximation are:')
print(round(np, 0), round(nq, 0), '\n')

zcrit = ss.norm.isf(SL/2) # The binomial approximates to the Normal, NEVER to the t-distribution
ME = zcrit * math.sqrt(phat*(1-phat)/n)

lower = phat - ME
upper = phat + ME

print('The {:2.0%} CI for the proportion of patients who gained weight is [{:4.2f}, {:4.2f}]'.format(CL, lower, upper))

Now with statsmodels

lower, upper = smp.proportion_confint(np, n, alpha = SL)
print('The {:2.0%} CI for the proportion of patients who gained weight is [{:4.2f}, {:4.2f}]'.format(CL, lower, upper))

Then, we obtain the same result.