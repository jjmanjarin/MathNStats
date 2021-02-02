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

+++ {"id": "gAuMiPYL5unN"}

# ANOVA

In this document we are going to develop and implement the basic ideas of the One-way ANOVA for the comparison of different population means. In this context we must say that ANOVA is just a hypothesis testing procedure that considers all the populations at once instead of go using all the possible two populations tests.

ANOVA is the usual name that a general class of linear models receive: those in which we only have **categorical explanatory variables**. Also, as a sort of field on its own, it has its own nomeclature:

 * We will denote as **Factors** to all the predictors (explanatory variables) of the model
 * We will denote as **Levels** to the observations of the factors, i.e. to all the possible categories
 * We will denote as **Effects** to the estimated parameters of the regression

With respect to this last idea, let's make another observation: with respect to the nature of these effects we may have three different types of models:

 * **Fixed Effects Models**: in this case the effects (estimated parameters) are constant in time, i.e. they are real numbers
 * **Random Effects Models**: in this case the estimated effects are random variables
 * **Mixed Effects Models**: in this case the estimated effects are both, constant and/or random variables

All the models we will consider in this document are Fixed Effects, i.e. all the estimated quantities will be constants.


+++ {"id": "xzYyqx8ft0XU"}

## General Description

Suppose a model in which we want to study if the the belonging to a certain group may affect the value of a response variable. We consider these groups as the different levels of one single categorical variable (with any number of possible levels)

We, then, propose the following model

\begin{equation}
x_{ij} = m + \tau_i + e_{ij}
\end{equation}

where:

  * $x_{ij}$ is the *j*th observation of the *i*th population
  * $\tau_i$ is the effect of belonging to the *i*th population
  * $e_{ij}$ is the random error of each observation
  * $m$ is the overall mean of all the observations

Remember that in any linear regression model, the response is always the conditional expected value of the response given that the regressors can take a given value.

In this context, the main idea is that if there is no difference between the different populations, then the variable $\tau_i$ must be irrelevant and then the expected value of each observation is the overall mean, up to some random errors. 

Then, the decision scheme that we use is

\begin{equation}
H_0:\,\{\mu_1=\dots=\mu_n\},\quad H_1:\,\{\text{at least one mean is different}\}
\end{equation}

which is basically an **overall significance test** for the regressors.

A technical note is that usually this is a **not of full rank model**, which means that, in general, there is not a unique solution for the estimators. 

+++ {"id": "9BQ4z6OFw0m8"}

## Model Assumptions

Just like for any linear model we are going to impose the following conditions:

 * The residuals must be normally distributed with zero expected value and same variance: $N(0,\,\sigma^2)$ (**homoskedasticity** and **zero conditional mean**).
 * The observations must be **independent**, satisfied as usual if the sample size is at most the 10% of the population size (in any case there is a formal test)
 

+++ {"id": "SWdfgKZuvhJi"}

## ANalysis Of VAriance

Using the first condition, we can take the expected value in the equation of the model and see that there are some valid estimators for the **overall mean**:

\begin{equation}
\hat m = {\bar{\bar x}}
\end{equation}

and for the **group effect**:

\begin{equation}
\hat\tau = \bar x_i -\bar{\bar x}
\end{equation}

if we plug this last equation in the model above we find an equation for the **residuals**

\begin{equation}
\hat e_{ij} = x_{ij} - \bar x_i
\end{equation}

from here we can find two different sum of squares for this model:

  * The **sum of squared residuals** (SSR or SSW)
  <br>
  \begin{equation}SSW = \sum_{ij} \hat e_{ij}^2 = \sum_{i,j} (x_{ij} - \bar x_i)^2
  \end{equation}
  
   which is known in this context as the Sum of Squares Within groups or **Unexplained Variability**. This last name implies that it is the part of variability that we will always have since it is due to the random nature of our variables, in the end it is nothing more than the variance. Now, what we usually want is this quantity divided by the number of degrees of freedom, a sort of average, in this case, since there are $N$ observations and $K$ populations, we have $N-K$ degrees of freedom, then we define
      
  \begin{equation}MSR = \frac{SSR}{n - K}\end{equation}

&nbsp;

  * The **sum of squares between groups** (SSG)
  
  \begin{equation}SSG = \sum_{i} \hat \tau_{i}^2 = \sum_{i} (\bar x_i -\bar{\bar x})^2\end{equation}
  
   which is known in this context as the **Explained Variability**, which implies that this is the variability of the response that we will be able to explain using our model. See that it is just the difference from each population mean to the overall mean, then if the model is relevant these differences will be significantly important. Just as before, we can find the degrees of freedom, which in this case it is simply $K-1$, then
      
    \begin{equation}MSG = \frac{SSG}{K-1}\end{equation}

Now we can define an F-ratio as

\begin{equation}F = \frac{MSG}{MSW}\end{equation}

We use this value as the test statistic, then:

  * If the model is useful, and let us explain the variability of the response in terms of the effect of the population, i.e. if there is a significant difference in the mean of the populations, then the $MSG$ will be significantly greater than $MSW$ and then the F-value will be large and its p-value small
  
  * If the model is not useful, then the $MSW$ will be greater than $MSG$, the F-value will be small and the p-value big. This implies that there is no significant difference bewteen the population means

+++ {"id": "HZXcV_9L8Ttb"}

## ANOVA in Python

Let's perform this analysis of variance in the same dataset we used for the linear models: `forestarea`. Then let's first load the packages we need

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
executionInfo:
  elapsed: 2557
  status: ok
  timestamp: 1606651453604
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: gYaw2qQs9fH3
outputId: c9a23208-c822-496a-c379-7ba02b38d3ce
---
import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
from scipy.stats import mstats
from sklearn.model_selection import train_test_split
```

+++ {"id": "TqISFXBP93jG"}

now we load the dataset

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 129
executionInfo:
  elapsed: 20531
  status: ok
  timestamp: 1572941774168
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: 3OYF-Wcm944b
outputId: dbbf9801-acbb-4d31-ff7d-18ce0dc6f34f
---
from google.colab import drive
drive.mount('mydrive')
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 206
executionInfo:
  elapsed: 1222
  status: ok
  timestamp: 1572941777322
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: bZLwzoTa-KAb
outputId: 79627156-0be1-42d7-d23b-24c7eab8468b
---
mydata = pd.read_csv("/content/mydrive/My Drive/IE Bootcamp - Math & Stats /data/forestarea.csv")
mydata.head()
```

+++ {"id": "9iTyNLWB-OUH"}

Since we saw that the best model we could obtain was with the log transformed response variable (taking `anwwith2014` as response) let's find that variable and add it to the dataset (as well as the other transformed variables)

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 243
executionInfo:
  elapsed: 849
  status: ok
  timestamp: 1572941780282
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: IKQTk-tk-gpO
outputId: 2c34921f-99c0-4fb5-afef-b4018b5faf14
---
mydata['lforar2014'] = np.log(mydata.forar2014)
mydata['lanwwith2014'] = np.log(mydata.anwwith2014)
mydata['lavprec2014'] = np.log(mydata.avprec2014)
mydata.head()
```

+++ {"id": "wMlc4w-AB8e3"}

### Replacing Codes

Let's replace the `continent` code by their names

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 206
executionInfo:
  elapsed: 749
  status: ok
  timestamp: 1572941794230
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: na7g5-w-CAdN
outputId: 3a3215d4-451c-429a-92ad-4e3c43f6a860
---
mydata['continent'].replace({1: 'Africa', 2: 'America', 3: 'Asia', 4:'Australia', 5:'Europe'}, inplace= True)
mydata.head()
```

+++ {"id": "biSg4zDjCGpq"}

### Train/Test splitting

Now let's preform the usual 80/20 splitting for the train and test sets

```{code-cell}
:id: mNX672ElA9mW

rand_state = np.random.RandomState(1)
df_train, df_test = train_test_split(mydata, 
                                   test_size = 0.20,
                                   random_state = rand_state)
```

+++ {"id": "y7Kw1JZxCWPn"}

### Descriptive

Let's use the `describe` function to have a first look at the distributions

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 238
executionInfo:
  elapsed: 823
  status: ok
  timestamp: 1572941801056
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: _8fkZtfACcmw
outputId: 376f56b1-8743-4868-fe02-37289120d555
---
df_train.groupby('continent')['lanwwith2014'].describe()
```

+++ {"id": "VCeyC-tsCjD2"}

Taking a look at the standard deviations, it seems that there are two continents that may be significantly different with respect to the mean and that may also give problems with the equality of variances: Africa and Australia.

Note also that from this table:

  * the data is unbalanced, i.e. the number of observations is different in each group
  * We do not have enough observations in Asia nor in Australia to proceed with them (we should drop them in the analysis)

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 206
executionInfo:
  elapsed: 902
  status: ok
  timestamp: 1572941804395
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: 9ZrB3EtJD77O
outputId: 4e9008fd-8f50-4fa1-d424-0ca7f19c743d
---
dataset = df_train[(df_train['continent'] != 'Asia') & (df_train['continent'] != 'Australia')]
dataset.head()
```

+++ {"id": "xPYfo4seHvBM"}

From now we will work with these three continents only: Europe, Africa and America

+++ {"id": "ZfmonRG1pq_N"}

\begin{equation}
E[\text{response}|\text{categorical}] = \beta_0 + \beta_1\,\text{categorical} + \text{error}
\end{equation}

+++ {"id": "SY9_CvD09ucq"}

### Checking the Assumptions

Since this ANOVA test is made on top of a linear model, we must check the assumptions of these type of models. In this case we will only consider:

  * Independency of the observations
  * Homoskedasticity of the residuals
  * Normality of the residuals

All these require that we have a model fitted and then from its residuals check the assumptions. However sometimes it is assumed (although formally wrong) that they may be directly checked with the data. In this case we can use the common rules:

 * Independency can be assumed if we have a sample with a size less than the 10% of the population size.
 * Homoskedasticity can be seen graphically with a comparison of boxplots and analytically with a Levene or Bartlett test (comparisons of multiple variances)
 * Normality can be checked with the normality of the sample (see in section **Failure of Normality** below)

Let's see how this may work

 * The sample is of size 34 (in the `df_train` once the NaN have been dropped) while the total number of countries is 195. This is the 17% so in this case we may have problems with correlations between different data
 * For the homoskedasticity we use

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 647
  status: ok
  timestamp: 1572941809590
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: Iq5oimwQ9zKw
outputId: 6c78e281-2298-4130-aa33-f9cf6f73231b
---
ss.levene(dataset['lanwwith2014'][dataset['continent'] == 'Africa'].dropna(),
          dataset['lanwwith2014'][dataset['continent'] == 'America'].dropna(),
          dataset['lanwwith2014'][dataset['continent'] == 'Europe'].dropna())
```

+++ {"id": "DWxSWnp7DMFV"}

To use Bartlett test we should use

```python
ss.bartlett(dataset['lanwwith2014'], dataset['continent'])
```

but with the continent encoded as a numerical variable, i.e. the original encoding of the dataset. We leave that for you.

In any case, we can see that both tests return a p-value higher the the common significance levels, then we fail to reject the Null hypothesis and must conclude that we do not find evidence against the equality of variances. See that the boxplots may have led us to a different conclusion

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 414
executionInfo:
  elapsed: 820
  status: ok
  timestamp: 1572941886022
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: Ti-qzFTEDgzF
outputId: 38b7f2e7-873c-40de-fb58-b8e3d26df091
---
plt.figure(figsize = (12,6))
g = sns.boxplot(data = dataset, x = 'continent', y = 'lanwwith2014')
g.axes.set_title('Annual Water Withdrawal vs. Continent', fontsize = 20)
g.axes.set_xlabel('Continent', fontsize = 15)
g.axes.set_ylabel('Log(Annual Water Withdrawal)', fontsize = 15)
plt.show()
```

+++ {"id": "9tCnruSt5tTn"}

## ANOVA - statsmodels

Now we define the ols model for the ANOVA test

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 473
executionInfo:
  elapsed: 680
  status: ok
  timestamp: 1572941890646
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: kdmVcs5U6H95
outputId: cc842b9b-1bfb-469e-ba0e-6dd33c64ec16
---
model = ols('lanwwith2014 ~ C(continent)', data = dataset).fit()
model.summary()
```

+++ {"id": "tkwlOcnB6I-M"}

From the summary table we see that the base group is **Africa** and all the other continents are significant and relevant to explain the variability of the response. Now the ANOVA table can be found as

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 112
executionInfo:
  elapsed: 565
  status: ok
  timestamp: 1572941896448
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: 8IOZChRh6Nyt
outputId: 46206ee4-40f4-4786-9872-c408e2fa130c
---
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table
```

+++ {"id": "FhKdGHtF6RHh"}

from where we see that the p-value is $0.0145$. So if our test were for a 5%, we Reject the Null hypothesis and conclude that we find evidence against the equality of means from our sample.


+++ {"id": "4ll8HKMtKytf"}

## ANOVA - scipy

We can also use the `scipy.stats` package where its `f_oneway` function gives the same answer, then

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
executionInfo:
  elapsed: 796
  status: ok
  timestamp: 1572941902594
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: GJwhk3QtK15l
outputId: 437da539-7dc9-4379-ce3d-d2b08fa49c72
---
ss.f_oneway(dataset['lanwwith2014'][dataset['continent'] == 'Africa'].dropna(),
            dataset['lanwwith2014'][dataset['continent'] == 'America'].dropna(),
            dataset['lanwwith2014'][dataset['continent'] == 'Europe'].dropna())
```

+++ {"id": "f-foBHaJLJQb"}

So the p-value is the same as before and, therefore, the conclusion too.

+++ {"id": "fdAJ7keWJOka"}

## Post-hoc Analysis

If we Reject the NULL hypothesis there are two different approaches (there are more) we can take in order to find out which group is actually significantly different:

  * Bonferroni: we set a penalty in the significance level and then compute all the two population differences for this new significance level
  * Tukey: looks for a **minimum significant difference** (HSD or MSD) that may let us declare two populations as different. To do it we introduce the **Studentised Interquartile Range** distribution or Tukey's-q

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 164
executionInfo:
  elapsed: 591
  status: ok
  timestamp: 1572941937290
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: 2r3USXjXhgvA
outputId: 416e6e56-89e8-4fa1-d080-8f385efe5a57
---
dataset_nona = dataset.dropna()
mc = MultiComparison(dataset_nona['lanwwith2014'], dataset_nona['continent'])
print(mc.tukeyhsd())
```

+++ {"id": "njoEBnA8hn2Y"}

Since the null hypothesis is the equality of means, from the output we see that **Africa** is significantly different to **America** but not to **Europe** and that **America** and **Europe** are not significantly different (all for a 5%)

## Failure of Normality

To test for normality we can look at the table p-values for JB and Omnibus (0.568 and 0.357) respectively and use the functions we used in linear models to find the p-values for Shapiro-Wilk and D'Agostino tests

```{code-cell}
:id: qdgpSrE3M5Ar

def NormalityTests(x, sig_level):
    '''
    This function computes the p-value and statistics of the Shapiro-Wilk and D'Agostino tests for normality
    It also includes the set of libraries to be loaded in the test (no cheks done)
    
    Inputs:
    
     - x: array of values of the variable to be tested
     - sig_level: significance level to be used in the decision of the test
    
    Output
    
     - p-value, statistic and decision for both tests    
    '''
    from scipy.stats import shapiro
    from scipy.stats import normaltest
    
    shap_stat, shap_p = shapiro(x)
    k2_stat, k2_p = normaltest(x)
    
    print("From the Shapiro Wilk test:\n\nStatistic: ", shap_stat, "\np-value: ", shap_p)
    if shap_p > sig_level:
        print("Fail to reject Normality: No evidence found against normality\n\n")
    else:
        print("Reject Normality: Evidence found against normality\n\n")
    
    print("From the D'Agostino test:\n\nStatistic: ", k2_stat, "\np-value: ", k2_p)
    if k2_p > sig_level:
        print("Fail to reject Normality: No evidence found against normality\n\n")
    else:
        print("Reject Normality: Evidence found against normality\n\n")
             

def HisQQplots(x):
    '''
    This function plots the histogram and qq-plot of an array in order to perform a visual analysis of normality
    
    Inputs:
    
     - x: array to plot
    
    Output:
    
     A plot consisting in two subplots (one for each of the previous ones)
    '''
    # define the different regions
    f, (ax_box, ax_hist) = plt.subplots(2, 
                                        sharex = False, 
                                        gridspec_kw={"height_ratios": (.25, .75)})
    f.set_figheight(8)
    f.set_figwidth(8)
    plt.suptitle('Normality Plots', fontsize = 20)
    # Add a graph in each part
    sns.distplot(x, hist = True, 
                 kde = False, 
                 bins = 10, 
                 hist_kws={'edgecolor':'black'},
                 ax=ax_box)
    ss.probplot(x, plot=sns.mpl.pyplot)
    plt.tight_layout(rect=(0,0,1,0.94))
```

+++ {"id": "TNla2qncNmEe"}

Then

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 274
executionInfo:
  elapsed: 570
  status: ok
  timestamp: 1572941985532
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: o7vlzNCDM7TV
outputId: 4b40bb9f-c036-4233-91eb-944d40b98437
---
NormalityTests(model.resid, 0.01)
```

+++ {"id": "eEejY_tHNneJ"}

And Graphically we can do

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 585
executionInfo:
  elapsed: 1251
  status: ok
  timestamp: 1572941989028
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: 93nb_ZNONJaP
outputId: 13c4d4d2-c8f6-4d3d-9135-d44115df39b6
---
HisQQplots(model.resid)
```

+++ {"id": "3ajeABRhNj-6"}

So in our case we do not find any problems with normality of our residuals but sometimes we will. In these cases, we may still perform a test for both:

 * Equatility of variances: Since Levene test is robust against outliers, can still be used when normality fails
 * Comparison of population: We use the **Kruskal-Wallis procedure**, a non-parametric test that does not require any normality in the variables


+++ {"id": "vKPIEy9qVtHq"}

### Kruskal-Wallis

This procedure tests if the **median** of all the populations are equal and the only requirement is that the distributions of the populations are of the same *type*, i.e. all rigth-skewed, all leptokurtic,... In Python there are different implementations of it, we will use the one in **scipy.stats**

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 127
executionInfo:
  elapsed: 730
  status: ok
  timestamp: 1572941994953
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: Hu8B_RhmhrKp
outputId: bd64691c-cb91-446c-c208-5fa19e30c0b7
---
print("Kruskal Wallis H-test test:\n")

H, pval = mstats.kruskalwallis(dataset['lanwwith2014'][dataset['continent'] == 'Africa'].dropna().values,
                               dataset['lanwwith2014'][dataset['continent'] == 'America'].dropna().values,
                               dataset['lanwwith2014'][dataset['continent'] == 'Europe'].dropna().values)

print("H-statistic:", H)
print("P-Value:", pval)

if pval < 0.05:
    print("\nReject the NULL hypothesis for a 5%: There is evidence in favor of significant differences between the populations.")
if pval > 0.05:
    print("\nFail to Reject the NULL hypothesis for a 5%: There is no evidence in favor of significant differences between the populations")
```

+++ {"id": "GW4K0rRqhzpd"}

The only point we should be careful with when we run this function is that it only accepts arrays as inputs, that's why we have added the **.values** at the end.

The p-value is in no contradiction with the previous ANOVA result then we could perform a Post-hoc analysis. In this case we should use any of the following

 * Conover test.
 * Dunn test.
 * Dwass, Steel, Critchlow, and Fligner test.
 * Mann-Whitney test.
 * Nashimoto and Wright (NPM) test.
 * Nemenyi test.
 * van Waerden test.
 * Wilcoxon test.

which are for non-parametric designs. In Python we have all these functions in the `scipy.stats` package or from the `scipy.stats.mstats`. Mann-Whitney test, for example, require that the categories are also numerical variables.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 368
executionInfo:
  elapsed: 6001
  status: ok
  timestamp: 1572942013796
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: z05DBeKJZucq
outputId: e4813e14-cc5d-4628-a508-3c200ce7901f
---
!pip install scikit_posthocs
```

```{code-cell}
:id: glb2MKBnZ2Xe

import scikit_posthocs as sp
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 143
executionInfo:
  elapsed: 625
  status: ok
  timestamp: 1572942018718
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -60
id: RgAcVERUZ3qc
outputId: cbef90a3-eece-4dc8-b10d-9c303c0844ab
---
sp.posthoc_conover(dataset_nona, val_col='lanwwith2014', group_col='continent', p_adjust = 'holm')
```

+++ {"id": "g_rVH8KfnfQa"}

In the table we see the p-values of the comparisons of the continents in groups of 2. If we were using a 1%, none of the differences would be significant. However, for a 10%, only the difference bewteen Europe and America would not be significant. In fact these results coincide with those of Tukey's post-hoc.
