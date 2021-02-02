---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "FQIl1fzGoRI4"}

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

+++ {"id": "WUWvbyIvpxR3"}

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

+++ {"id": "iC6DdEzBrCh1"}

# <font color="Red">Application</font>

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 74
executionInfo:
  elapsed: 2549
  status: ok
  timestamp: 1604085342936
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: bxndwDMVprHz
outputId: 08e6d795-0b5d-46ea-bd5b-69859fa48a82
---
import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.stats.weightstats as smw
import statsmodels.stats.proportion as smp
import matplotlib.pyplot as plt
import math

plt.style.use('seaborn')
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 29326
  status: ok
  timestamp: 1604085371070
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: 5hkdZnqLoQBI
outputId: 3f878773-c63a-49a7-d2c7-f47ebba92263
---
from google.colab import drive 
drive.mount('IEStats')
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 206
executionInfo:
  elapsed: 1563
  status: ok
  timestamp: 1604085371796
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: ciJHMg-wrsaR
outputId: 9232995d-f31b-467f-80f5-0561d1094c7b
---
anorexia = pd.read_excel('/content/IEStats/My Drive/IE - 2021 - Statistics and Data Analysis/DataSets/anorexia.xlsx')
anorexia.head()
```

+++ {"id": "rNh3gIF7r9gp"}

Let's estimate for a 95% of CL the Pre-treatment weight.

First directly with the **formula**

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 824
  status: ok
  timestamp: 1604085376150
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: J7aZUXCAsF3l
outputId: c88e8590-099e-4cfa-d947-1a018c9dc647
---
prew = anorexia['prewt']
n = len(prew)

print('Since {:2.0f} > 40, we can use the normal approximation'.format(n))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 577
  status: ok
  timestamp: 1604085380484
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: aGeq8Ox4sqxG
outputId: 4c21ba71-fc14-4880-86dc-ec48eac5a353
---
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
```

+++ {"id": "f8JDwoVzuVGi"}

Let's now use `statsmodels`

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 628
  status: ok
  timestamp: 1604085383218
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: 2MegVAzquXX4
outputId: ebd90d53-4998-46b2-b923-05bb8685cdb4
---
des_prew = smw.DescrStatsW(prew)
lower, upper = des_prew.zconfint_mean(SL) # Note that the only argument needed is the "alpha"

# Print the output
print('The {:2.0%} CI for the weight before the treatment is [{:4.2f}, {:4.2f}]'.format(CL, lower, upper))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 648
  status: ok
  timestamp: 1604085405040
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: p_DhHT-0wDGk
outputId: bd70e107-3356-4e0a-a645-e294c4be0031
---
prew_1 = anorexia[anorexia['group'] == 1]['prewt'] # 1st is the filter, the second the selection
n1 = len(prew_1)

des_prew_1 = smw.DescrStatsW(prew_1)

if n1 > 40:
  lower_1, upper_1 = des_prew_1.zconfint_mean(SL)
else:
  lower_1, upper_1 = des_prew_1.tconfint_mean(SL)

print('The {:2.0%} CI of the group 1 pre-treatment weight is [{:4.2f}, {:4.2f}]'.format(CL, lower_1, upper_1))
```

+++ {"id": "coapcTgvyYFO"}

## <font color="Blue">Proportions</font>

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 90
executionInfo:
  elapsed: 604
  status: ok
  timestamp: 1604085443426
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: KTFX_YngyXBx
outputId: a78a7fe4-d172-4068-cdd0-c729e7353dca
---
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
```

+++ {"id": "j-3AEhPp8XkR"}

Now with statsmodels

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 557
  status: ok
  timestamp: 1604085502730
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: 4ka0oS2I8Y_O
outputId: 0e4fc1a0-e091-4fa4-8d47-1e594583a353
---
lower, upper = smp.proportion_confint(np, n, alpha = SL)
print('The {:2.0%} CI for the proportion of patients who gained weight is [{:4.2f}, {:4.2f}]'.format(CL, lower, upper))
```

+++ {"id": "Ihmqu7BE8m3I"}

Then, we obtain the same result.
