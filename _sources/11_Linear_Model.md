---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Linear Regression

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 73
executionInfo:
  elapsed: 2295
  status: ok
  timestamp: 1603188950389
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: CItCQvgrPfZ2
outputId: e5ebf491-ab76-4855-8430-d1ecd57ea69a
---
import pandas as pd
import numpy as np
import scipy.stats as ss
import random as rd

# plots
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rc('font', size = 12)
plt.rc('figure', titlesize = 20)
plt.rc('axes', labelsize = 12)
plt.rc('axes', titlesize = 15)

# models
import statsmodels.stats as sts
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.stats.anova as aov
import statsmodels.stats.outliers_influence as sso
import statsmodels.stats.diagnostic as ssd
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.compat import lzip

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 20533
  status: ok
  timestamp: 1603188968646
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: O-cqNGP7Phr-
outputId: 971738c8-f02e-49ab-bf94-53384cdcee58
---
from google.colab import drive
drive.mount('IEXL')
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 206
executionInfo:
  elapsed: 21427
  status: ok
  timestamp: 1603188969551
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: 8Wwdv6_nPwVt
outputId: d07f101d-9989-472b-9e0a-8b0deb7a25c6
---
gifted = pd.read_csv('/content/IEXL/My Drive/IEXL - Bootcamp - Math&Stats 20-21 September/data/gifted.csv')
gifted.head(5)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 21426
  status: ok
  timestamp: 1603188969560
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: -ays0rgrVE2K
outputId: 6ea2e6cb-8910-451b-f144-9e3097683c64
---
max(gifted['score']), min(gifted['score'])
```

+++ {"id": "MYzzCNB-Q40A"}

## Multicollinearity Analysis

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
executionInfo:
  elapsed: 38841
  status: ok
  timestamp: 1603188986982
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: bdOdlIe4QtCc
outputId: 302d286d-c04a-4235-b59e-e2445f9d513b
---
sns.pairplot(gifted.drop('score', axis = 1),
             height = 2)
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 269
executionInfo:
  elapsed: 38837
  status: ok
  timestamp: 1603188986988
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: aWqIKE7NRAIw
outputId: ab7e083c-e86c-4a51-c6c5-8fdd7ede1e92
---
gifted.drop('score', axis = 1).corr()**2
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 181
executionInfo:
  elapsed: 38836
  status: ok
  timestamp: 1603188986994
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: 62R59tmqRL4f
outputId: edbfa395-63db-469e-eb19-681b3bd29fc7
---
X = gifted.drop('score', axis = 1)
X['Intercept'] = 1

vif = pd.DataFrame()
vif['variables'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.dropna().values, i)for i in range(X.shape[1])]

print(vif)
```

+++ {"id": "X9auw6yhRaVR"}

We should drop either read/count and either edutv/cartoons

+++ {"id": "5fPj3paHRgwB"}

## Model

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 563
executionInfo:
  elapsed: 38831
  status: ok
  timestamp: 1603188986997
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: rDrOvTXmRZuQ
outputId: 9faddc4e-b107-4c91-992a-bb0f4f46f0a7
---
model = ols('score ~ fatheriq + motheriq + speak + count', data = gifted).fit()
print(model.summary())
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 38824
  status: ok
  timestamp: 1603188986999
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: SmwkFPkITx1D
outputId: 193e6456-4c7f-4c84-ef66-7f91fb1f80fa
---
#RSE
resids = model.resid
n = model.nobs
K = len(model.model.exog[0]) - 1

RSS = sum(resids**2)
RSE = np.sqrt(RSS/(n-K-1))

print("The Residuals Standard Error is {:4.2f}".format(RSE))
```

## Hypothesis Testing in the Linear Model

+++ {"id": "3tPTrP4Pv7gO"}

### <font color = "blue">Multiple Restrictions </font>

When we speak about multiple restrictions, we focus on a set of tests that affect only to a subset of the variables in the model, in particular, we may consider:

 * Tests to see if the impact of two different variables is not the same
 * Tests to see if a subset of the variables is relevant to explain the behaviour of the response

+++

#### Same Impact

Let's first test if the impact of fathers and mothers is the same:

\begin{equation}
H_0:\{\beta_{fatheriq}=\beta{motheriq}\}
\end{equation}

then we define

\begin{equation}
d = \beta_{fatheriq} - \beta{motheriq}
\end{equation}

and the model to estimate becomes

\begin{equation}
\text{score} = \beta_0+\beta_1\, TotalIQ + d\,\text{motheriq}+ \beta_3\,\text{speak} + \beta_4\,\text{count}
\end{equation}

+++ {"id": "r9EOfs_wyCen"}

#### Subset Relevance

The second test that we are going to do is the analysis of 

\begin{equation}
H_0:\{\beta_{speak} = 0, \beta_{count}=0\}
\end{equation}

I need to estimate the models with and without the variables involved

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 527
executionInfo:
  elapsed: 38820
  status: ok
  timestamp: 1603188987002
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: jTO_4A1Yyg_j
outputId: b3946f31-91b0-4e99-8942-88fd2ac70768
---
model_restricted = ols('score ~ fatheriq + motheriq', data = gifted).fit()
print(model_restricted.summary())
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 112
executionInfo:
  elapsed: 38816
  status: ok
  timestamp: 1603188987006
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: k5gVcTRRyskV
outputId: 12430a5c-721c-4a41-c09d-875c55405fdd
---
aov.anova_lm(model_restricted, model)
```

+++ {"id": "yBvJZAjty773"}

since the p-value is smaller than the SL we reject the NULL hypothesis and conclude that the model WITH the variables included is better to explain the variability of the response

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 490
executionInfo:
  elapsed: 38808
  status: ok
  timestamp: 1603188987009
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: 3QwhHn_ozwrS
outputId: cc8dcf20-3c93-472f-843c-79abf1b6d145
---
model_restricted_2 = ols('score ~ speak + count', data = gifted).fit()
print(model_restricted_2.summary())
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 112
executionInfo:
  elapsed: 38802
  status: ok
  timestamp: 1603188987010
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: p9oAu_Wzz6rG
outputId: 5a520447-6085-408c-b8f8-2f8efedc7226
---
aov.anova_lm(model_restricted_2, model)
```

+++ {"id": "ie4D7bu_0s-h"}

## <font color = "Red"> Validation </font>

The validation goes along these lines:

 * Multicolinearity (already checked)
 * Linear Description: absence of misspecification problems (Ramsey. Harvey-Collier)
 * Zero expected value and normality of the residuals (Shapiro-Wilk, D'Agostino)
 * Independency of the residuals (Durbin-Watson test)
 * Homoskedasticity (White, Breusch-Pagan,...)

+++ {"id": "vScy1L_b1eBT"}

### <font color="Blue"> Linear Description </font>

The H0 of the test is that the model is properly specified (no need of extra variables)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 53
executionInfo:
  elapsed: 38797
  status: ok
  timestamp: 1603188987012
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: V46rba5K125V
outputId: b9334d86-74ef-415b-9ea1-2ea1716feb3e
---
sso.reset_ramsey(model, degree = 2)
```

+++ {"id": "iJc8NEzU2gGp"}

the p-value implies that there are no problems with misspecification of the model (we don't need to include higher degrees in the variables)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 38792
  status: ok
  timestamp: 1603188987013
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: gCzUsm8Z2rbV
outputId: 44142ec8-2fd4-4230-cfe5-bf2e324c4285
---
sms.linear_harvey_collier(model)
```

```{code-cell} ipython3
---
executionInfo:
  elapsed: 38790
  status: ok
  timestamp: 1603188987015
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: QfzlT74f6lao
---
# The residuals are "resids"
fitted = model.fittedvalues
abs_resids = np.abs(resids)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 463
executionInfo:
  elapsed: 38783
  status: ok
  timestamp: 1603188987016
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: c36PeFD66476
outputId: 5a8f4002-f45b-45ed-b873-e806dd5fbfff
---
f = plt.figure()
f.set_figheight(6)
f.set_figwidth(15)

f.axes[0] = sns.residplot(fitted, 'score', data = gifted,
                          lowess = True,
                          scatter_kws = {'color': 'blue'},
                          line_kws = {'color': 'red', 'lw': 2, 'alpha': 0.5})
f.axes[0].set_title("Residuals vs. Fitted")
f.axes[0].set_xlabel("Fitted Values")
f.axes[0].set_ylabel("Residuals")

plt.show()
```

+++ {"id": "IWxi6rJkOCoj"}

### <font color ="Blue">Zero Expected Value and Normality </font>

To check the expected value, we just find the average of the residuals

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 38778
  status: ok
  timestamp: 1603188987017
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: MTCK_xYaOOie
outputId: bd659a91-7882-4e0d-ff47-55b169264548
---
resids.mean()
```

+++ {"id": "fSGR_2VROkez"}

To check normality we are going to use:

 * Shapiro-Wilk test
 * D'Agostino

 In any of these tests,
 
\begin{equation}
H_0:\{\text{ normality} \}
\end{equation}

Then we can make a qq-plot (normal).

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 90
executionInfo:
  elapsed: 38771
  status: ok
  timestamp: 1603188987018
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: nF6cwKshOcVz
outputId: 257b502d-c6e5-475f-8085-cf5cefc5028f
---
s_stat, s_pval = ss.shapiro(resids)
d_stat, d_pval = ss.normaltest(resids)

print("Shapiro-Wilk statistic: {:4.2f}\np-value: {:4.4f}".format(s_stat, s_pval))
print("D'Agostino statistic: {:4.2f}\np-value: {:4.4f}".format(d_stat, d_pval))
```

+++ {"id": "CsZ7V4LtPtc4"}

since the p-values are big enough we don't find evidence against the normality of the residuals.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 298
executionInfo:
  elapsed: 39020
  status: ok
  timestamp: 1603188987273
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: hmMtqOYWQBmd
outputId: 0b5ddcd6-8b1c-4967-c50b-b4ca525cdb4d
---
ss.probplot(resids, plot = sns.mpl.pyplot)
plt.show()
```

+++ {"id": "Qtr2iv1Q1YdC"}

#### <font color = "darkBlue"> Transformation of Variables </font>

Suppose that the normality of the residuals is not satisfied, then we may need a transformation of the response variable $y$ in order to get the right distribution. The best way is using the Box-Cox transformation 

\begin{equation}
y^{(\lambda)} = \left\{\begin{array}{lr}
\displaystyle\frac{y^\lambda - 1}{\lambda}, & \lambda\neq 0 \\[2ex]
\log\lambda, & \lambda = 0
 \end{array}\right.
\end{equation}

But the method relies in finding the proper value of $\lambda$. This is done by using the value such that likelihood of normality is maximized. This can be found to be

\begin{equation}
\log L(\beta,\sigma)=-\frac{n}{2}\left( y^{(\lambda)T}\left(I - X(X^TX)^{-1}X^T\right)y^{(\lambda)}\right) + \log J
\end{equation}

where $J$ is the jacobian of the transformation, $\beta$ are the coefficients of the linear model and $\sigma$ is the variance of the errors.

Then, let's see how to find it

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 265
executionInfo:
  elapsed: 40020
  status: ok
  timestamp: 1603188988279
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: 04pHjJOD3DPk
outputId: 11514591-378a-4e70-fc28-1a8920e24092
---
def log_likelihood(lmbda, x, y):
    n, p = x.shape
    log_jacobi = (lmbda - 1) * np.sum(np.log(y))
    trans_y = ss.boxcox(y, lmbda=lmbda)
    coeffs = np.linalg.inv(np.matmul(np.transpose(x), x))
    K = np.subtract(np.identity(n), np.matmul(np.matmul(x, coeffs), np.transpose(x)))
    rss = np.matmul(np.matmul(np.transpose(trans_y), K), trans_y)
    return - n / 2.0 * np.log(rss) + log_jacobi

lambda_values = np.arange(-5,5,0.01)
x = np.array(gifted[['fatheriq', 'motheriq', 'speak', 'count']])
y = gifted['score']
likelihoods = np.array([log_likelihood(l, x, y) for l in lambda_values])

box_cox_powers = pd.DataFrame({'lambda': lambda_values, 'likelihood': likelihoods})
power = box_cox_powers.loc[box_cox_powers['likelihood'].idxmax(), 'lambda']

plt.plot(lambda_values, likelihoods)
plt.vlines(power, -1000, -100)
plt.text(2,-400, r'$\lambda\simeq 1.5$')
plt.show()
```

+++ {"id": "HfiAPMUDQsG8"}

### <font color = "Blue"> Autocorrelation </font>

We speak of autocorrelation when the values of the variable depend on the values of the variable itself. This, known as serial correlation can be written in the form of a linear model

\begin{equation}
e_t = \rho\, e_{t-1}+ u_t
\end{equation}

where the null hypothesis for the relevancy of the model is

\begin{equation}
H_0:\{\rho = 0\}
\end{equation}

and this implies that if we fail to reject the hypothesis, there is no serial correlation. 

In our case we may lag the residuals, estimate the corresponding model and then analyse the individual significance of the variable


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 472
executionInfo:
  elapsed: 40013
  status: ok
  timestamp: 1603188988281
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: AXDDUuQMGuQU
outputId: 753d7fc4-de14-4b6f-98ee-a84c9947a180
---
residuals = pd.DataFrame({'residuals': resids,
                          'lag_resids': resids.shift()})
autocorrelation_model = ols('residuals ~ lag_resids', data = residuals).fit()
print(autocorrelation_model.summary())
```

+++ {"id": "GMDxuQwnHciB"}

from here we identify a p-value of 0.946 for $\rho$ which implies that we fail to reject the NULL hypothesis and the conclusion is that we do not find any evidence of serial correlation.

+++ {"id": "1FJ8FhoGGB-u"}

#### <font color = "darkBlue"> Durbin-Watson Statistic </font>

There is another form of testing for autocorrelation by using the Durbin-Watson statistic, defined as

\begin{equation}
DW = \dfrac{\sum_{t=2}^n(\hat e_t-\hat e_{t-1})^2}{\sum_{t=1}^n \hat e_t^2}\approx 2(1-\hat\rho)
\end{equation}

due to the last relation the test on $\rho$ is completely equivalent to the test on DW. Now, we see that if $\hat\rho = 0$, i.e. it is statistically not significant, $DW=2$, reason why we will usually see that when the value of the DW statistic is around 2, we can declare independency. In the same way, when $\hat\rho >0$ (positive correlation) we find $DW < 2$ and the other way around, when $\hat\rho <0$ (negative correlation) we find $DW > 2$.

In the case of our modelwe have that

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 40007
  status: ok
  timestamp: 1603188988284
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: AI5yxnJXRSkQ
outputId: 33634877-e912-4330-d306-98915b2eec0e
---
sts.stattools.durbin_watson(resids)
```

+++ {"id": "hY_ckjriSyNA"}

and the we may say that the value is close enough to 2 as to assume independency or that, in any case we detect a very low negative autocorrelation.

+++ {"id": "L57ruC4uJ9M_"}

However, the relation is not exact, just an approximation (due to the denominator) and then the probability distribution fo DW depends on the values of the independent variable, the number of regressors, the sample size and on whether or not the model contains an intercept. Too many things to immediately declare independency with a value around 2. 

Formally we can find a tabulation the values of the DW distribution (Savin-White) that return a high ($d_U$) and low ($d_L$) values such that

 * If $DW < d_L$ we reject the NULL in favor of $H_1:\{\rho >0\}$
 * If $DW > d_U$ we fail to reject the NULL, and do not find evidence of autocorrelation
 * If $d_L < DW < d_U$ the test is inconclusive

Then again, in our case we have

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 53
executionInfo:
  elapsed: 40005
  status: ok
  timestamp: 1603188988288
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: 0gBYlU73L8g_
outputId: 05033067-454e-4b16-a433-60fc6c732235
---
print('{:<20} {:>4.0f}\n{:<20} {:>4.0f}'.format('sample size: ', n, 'Number of Regressors: ', K))
```

+++ {"id": "nx8u6RPDMvJL"}

looking at the tables this implies that $d_L=1.043$ and $d_U = 1.513$, then our value is greater than the upper limit and then we fail to reject the null hypothesis, meaning that we do not find evidence against the lack of autocorrelation.

In any case, the three methods we have followed contain the same conclussion: there is no autocorrelation in the residuals.

+++ {"id": "UhACfwp8T1kO"}

### <font color = "Blue"> Heteroskedasticity </font>

In this case the decision scheme is

\begin{equation}
H_0:\{\text{homoskedastic}\},\qquad H_1:\{\text{heteroskedastic}\}
\end{equation}

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 90
executionInfo:
  elapsed: 40002
  status: ok
  timestamp: 1603188988291
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: NbJqiVseUB9u
outputId: df49524c-f992-496c-9fc7-11913fe85068
---
values = ['Multiplier', 'p-value', 'F statistic', 'Fp-value']
bp_test = ssd.het_white(resids, model.model.exog)
lzip(values, bp_test)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 90
executionInfo:
  elapsed: 39998
  status: ok
  timestamp: 1603188988294
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: dXXJviJwVPzK
outputId: 2844bbde-ad09-48d5-a354-ca3870a26274
---
values = ['Multiplier', 'p-value', 'F statistic', 'Fp-value']
bp_test = ssd.het_breuschpagan(resids, model.model.exog)
lzip(values, bp_test)
```

+++ {"id": "wjwlgKoyYPI2"}

Both tests return p-values large enough as to consider that we have problems, then we do not find evidence agains homoskedasticity.

+++ {"id": "EZRmqic7DEgw"}

## <font color = "Red"> Influential Points </font>

The pieces of information that we need are

 * Studentized (standarized) residuals
 * Leverage
 * Cook's Distance
 * Squared and Absolute Value Residuals

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 226
executionInfo:
  elapsed: 39994
  status: ok
  timestamp: 1603188988296
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: iHHPrUhYDef4
outputId: 44edb875-a982-46b6-ae19-350abeb99b7c
---
influential = model.get_influence()
inf_data = influential.summary_frame()
inf_data.head()
```

```{code-cell} ipython3
---
executionInfo:
  elapsed: 39992
  status: ok
  timestamp: 1603188988297
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: YwmPvGU8D-DA
---
std_resids = inf_data.student_resid
leverage = inf_data.hat_diag
cooks = inf_data.cooks_d
sq_resid = np.sqrt(np.abs(std_resids))
norm_resids = influential.resid_studentized_internal

n = model.nobs
K = len(model.model.exog[0]) - 1
```

+++ {"id": "PaH4fAIVDdve"}

### <font color = "Blue"> Outliers </font>

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 39988
  status: ok
  timestamp: 1603188988299
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: 7RJOwOOpGmEt
outputId: 5d19c3f4-963f-4190-fdb9-284e1c6a5903
---
t_crit = abs(ss.t.ppf(0.05/(2*n), n-K-1))
outlier = [i for i in abs(std_resids) if i >= t_crit]
std_resids.index[std_resids > t_crit]
```

+++ {"id": "31rBgg7aEoWb"}

### <font color = "Blue"> Leverage </font>

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 39984
  status: ok
  timestamp: 1603188988301
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: pVGtm4opEqL5
outputId: 8bde935e-1762-4c96-8b5c-3a099c306349
---
boundary = 2*(K+1)/n
high_leverage = [i for i in leverage if i >= boundary]
leverage.index[leverage > boundary]
```

+++ {"id": "3ewpXvViEuAN"}

### <font color = "Blue"> Influential Points </font>

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 39979
  status: ok
  timestamp: 1603188988302
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: s6Ql_gMTIeWh
outputId: a038171f-59e6-461a-cf03-3c572680cefd
---
D = 4/(n+K-1)
influential = [i for i in cooks if i >= D]
cooks.index[cooks > D]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 181
executionInfo:
  elapsed: 39972
  status: ok
  timestamp: 1603188988303
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: Orkjty14JEYY
outputId: 6d2ebd0f-4d6b-40be-965d-72186a00e83d
---
gifted.iloc[23, :]
```

+++ {"id": "hiJVyHF9JnNY"}

### <font color = "Blue"> Graphical Analysis </font>

We are going to find the residuals plot of the model and then use the critical t-values to mark the range where we can find the outliers

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 448
executionInfo:
  elapsed: 40279
  status: ok
  timestamp: 1603188988616
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: QQWk-IOQJnqJ
outputId: 82fcc911-cd0e-4dcb-f24d-99cc90691fad
---
plt.figure(figsize = (15, 5))
# sns.residplot(fitted, 'score', data = gifted)
sns.regplot(fitted, std_resids,
            scatter = True,
            lowess = True,
            scatter_kws = {'color': 'blue'},
            line_kws = {'color': 'red', 'lw': 2, 'alpha': 0.5})

plt.title("Residuals vs. Fitted", fontsize = 20)
plt.xlabel('Fitted values', fontsize = 15)
plt.ylabel('Studentized Residuals', fontsize = 15)

abs_resids_top = abs_resids.sort_values(ascending=False)[:4]
for i in abs_resids_top.index:
  plt.annotate(i, xy = (fitted[i],
                        resids[i]))

plt.hlines(ss.t.ppf(0.05/(2*n), n-K-1), min(fitted), max(fitted), color = "red", linestyles='--', linewidth = 1)
plt.hlines(ss.t.isf(0.05/(2*n), n-K-1), min(fitted), max(fitted), color = "red", linestyles='--', linewidth = 1)
plt.hlines(ss.t.ppf(0.01/(2*n), n-K-1), min(fitted), max(fitted), color = "green", linestyles='--', linewidth = 1)
plt.hlines(ss.t.isf(0.01/(2*n), n-K-1), min(fitted), max(fitted), color = "green", linestyles='--', linewidth = 1)

plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 415
executionInfo:
  elapsed: 40722
  status: ok
  timestamp: 1603188989067
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: RMHr8b87QKCj
outputId: 8157f42f-da0d-4c02-8ec2-cfedaefc9346
---
def graph(formula, x_range, label = None):
  x = x_range
  y = formula(x)
  plt.plot(x, y, label = label, lw = 1, ls = '--', color = "red")


plt.figure(figsize = (15, 5))
sns.regplot(leverage, std_resids,
            scatter = True,
            lowess = True,
            scatter_kws = {'color': 'blue'},
            line_kws = {'color': 'red', 'lw': 2, 'alpha': 0.5})

plt.title("Residuals vs. Fitted", fontsize = 20)
plt.xlabel('Leverage', fontsize = 15)
plt.ylabel('Studentized Residuals', fontsize = 15)

cooks_top = cooks.sort_values(ascending=False)[:5]
for i in cooks_top.index:
  plt.annotate(i, xy = (leverage[i],
                        std_resids[i]))

graph(lambda x: np.sqrt((D*(K+1)*(1-x))/x),
      np.linspace(0.001, max(leverage), 50),
      'Cook\'s distance')
graph(lambda x: -np.sqrt((D*(K+1)*(1-x))/x),
      np.linspace(0.001, max(leverage), 50))
plt.legend(loc = 'best')
plt.ylim(-4,4)

plt.show()
```

+++ {"id": "NdSj_0i_VTNQ"}

## <font color = "Blue"> Categorial Variables </font>


+++ {"id": "aknLRaqrVTHo"}

Since we do not have any categorial variable in the dataset, we are going to *invent* one using a random generator

```{code-cell} ipython3
---
executionInfo:
  elapsed: 40722
  status: ok
  timestamp: 1603188989070
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: j8EQX-MvVUUV
---
def rand_gl(x, m, name = None, labels = None):
    '''
    Creation of a randomly generated factor series of pandas
    
    INPUT:
    x: value that determines the number of possible categories, since we begin at 0, a velue of 1 will imply 2 categories
    m: length of the vector
    name: default values of the input values: 0, 1,... The number of names cannot be in disagreement with the value of x
    labels: labels of the categories. The number of labels cannot be in disagreement with the value of x
    
    OUTPUT:
    the output is a categorical vector of length m with labels given by the names or the labels with 
    '''
    cat = []
    for i in range(m):
        cat += [rd.randint(0, x)]
    
    if name is None:
        cat = pd.Series(cat, dtype = 'category', name = 'categories')
    else:
        cat = pd.Series(cat, dtype = 'category', name = name)
    
    if labels is None:
        return(cat)
    else:
        cat = cat.cat.rename_categories(labels)
        return(cat)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 206
executionInfo:
  elapsed: 642
  status: ok
  timestamp: 1603190314568
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: xJRL-mAAyJlH
outputId: a20bff03-46a5-41a2-ab5c-fa58b387cec2
---
np.random.seed(1)
gifted['female'] = rand_gl(1, len(gifted), labels = ['1', '0'])
gifted.head()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 581
executionInfo:
  elapsed: 825
  status: ok
  timestamp: 1603190484868
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: dpBLymjlzhoq
outputId: 1c99f49e-f504-43ca-b7ef-46955314d435
---
model_cat = ols('score ~ fatheriq + motheriq + speak + count + C(female)', data = gifted).fit()
print(model_cat.summary())
```

+++ {"id": "5cbJx5QWl9ZW"}

Now we are going to use `patsy` to define the interactions between the variable and any other variable in the model

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 654
executionInfo:
  elapsed: 791
  status: ok
  timestamp: 1603190638419
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: DNwB4PyXl5UM
outputId: a10708aa-71f2-436d-8471-9632ee564a45
---
features = ['fatheriq', 'motheriq', 'speak', 'count']
X = gifted[features]

model_cat_full = ols('score ~ C(female) * X', data = gifted).fit()
print(model_cat_full.summary())
```

+++ {"id": "9QyF3FJAnUcP"}

The model we find is

\begin{equation}
\text{score} = 43.44+0.32\,\text{fiq} + 0.45\,\text{miq}\dots +0.13\,\text{fiq}\cdot\text{female} +\dots + 8.39\,\text{female}
\end{equation}

then, we can split the model in the two possible categories:

\begin{equation}
\text{score}^f \approx 51 + 0.45\,\text{fiq} + 0.33\,\text{miq}\dots\qquad \text{score}^{nf} \approx 43 + 0.32\,\text{fiq} + 0.45\,\text{miq}\dots
\end{equation}

We can interpret for example the impact of being a female with respect to the `fatheriq` variable: *for the same levels of `motheriq`, `speak` and `count`, for each unit increase in the father's iq, on average, female gifted children will have a higher iq of 0.13 units*

+++ {"id": "oKFhEb-SooZd"}

### <font color = "Blue"> Chow Test </font>

if we have a model like

\begin{equation}
\text{response} = \beta_0 + \beta_1\,x_1 + \beta_2\,x_2 +\delta_1\text{cat} + \delta_2\,x_1\cdot\text{cat}+\delta_3\,x_2\cdot\text{cat}
\end{equation}

we can perform two different multiple restriction tests:

\begin{equation}
H_0:\{\delta_1=0,\,\delta_2=0,\,\delta_3=0\}\qquad H_0:\{\delta_2=0,\,\delta_3=0\}
\end{equation}

the first one is the classical Chow test and the F-dsitribution statistic that we have to compute is

\begin{equation}
F = \frac{SSR_p - (SSR_1 + SSR_0)}{SSR_1 + SSR_0}\frac{T-2(K+1)}{K+1}
\end{equation}

ini here we have three models:

 * **Pooled model**: The model when we do not consider the categorial variable (in any sense). Here we find $SSR_p$ and $T$ which is just the number of observations in the pooled model (sample size)
 * Models for each of the categories estimated separatedly (**model 1** and **model 0**), here we find $SSR_1$ and $SSR_0$

In any of the models we find the value of $K$: the number of regressors.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 762
  status: ok
  timestamp: 1603193284355
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: bqvhX8H0vWHF
outputId: b561a071-1277-413a-90ac-e2571658ed63
---
# Models
pooled = ols('score ~ fatheriq + motheriq + speak + count', data = gifted).fit()
model_0 = ols('score ~ fatheriq + motheriq + speak + count', data = gifted[gifted.female == '0']).fit()
model_1 = ols('score ~ fatheriq + motheriq + speak + count', data = gifted[gifted.female == '1']).fit()

# T and K
T = pooled.nobs
K = len(pooled.model.exog[0]) - 1

# Sum of square residuals
SSRp = sum(pooled.resid**2)
SSR0 = sum(model_0.resid**2)
SSR1 = sum(model_1.resid**2)

# F-statistic
Fstat = ((SSRp - (SSR0 + SSR1))/(SSR0 + SSR1))*((T - 2*(K+1))/(K+1))
Fpval = 1 - ss.f.cdf(Fstat, K+1, n-2*(K+1))

Fpval
```

+++ {"id": "Tv9NeFMWxVjj"}

If we want to test only the slopes we use the following statistic

\begin{equation}
F = \frac{SSR_p - (SSR_1 + SSR_0)}{SSR_1 + SSR_0}\frac{T-2(K+1)}{K}
\end{equation}

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 626
  status: ok
  timestamp: 1603193452503
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -120
id: 3JFiceZKxpML
outputId: b3cfb885-ece9-48d6-9488-9078a7825cda
---
# F-statistic
Fstat = ((SSRp - (SSR0 + SSR1))/(SSR0 + SSR1))*((T - 2*(K+1))/(K))
Fpval = 1 - ss.f.cdf(Fstat, K, n-2*(K+1))

Fpval
```
