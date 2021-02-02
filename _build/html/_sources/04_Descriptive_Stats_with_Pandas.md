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

+++ {"id": "CGwzyLNwUGub", "colab_type": "text"}

# Descriptive Statistics with Pandas


In this document we are going to see how to compute the basic numerical values needed to perform a descriptive analysis of a data set.

We are going to split the document in two main parts:

 * One for **Numerical** variables
 * One for **Categorical** variables
 
most of all because categorical variables are slightly subtle in Python. Let's load all the different packages we need for **all** the analysis to be made

```{code-cell}
:colab: {}
:colab_type: code
:id: dMhwInJFi5Us

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
```

+++ {"id": "sNRuyIXKegA5", "colab_type": "text"}

## 1.- The Data

Let's connect to our drive and load a dataset

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 131
colab_type: code
executionInfo:
  elapsed: 25608
  status: ok
  timestamp: 1571728164468
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: arUct2llUD89
outputId: 4439f9a2-36ce-4646-b5ee-d6013807fbb7
---
from google.colab import drive
drive.mount('mydrive')
```

+++ {"id": "iuR-A-lofaNY", "colab_type": "text"}

now we can load the dataset

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 206
colab_type: code
executionInfo:
  elapsed: 1189
  status: ok
  timestamp: 1571728264674
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: NqR10vNteqKa
outputId: e8738d3c-fb8a-4a3e-aabf-bff32446d2e1
---
mydf = pd.read_csv("/content/mydrive/My Drive/IE Bootcamp - Math & Stats /data/forestarea.csv")
mydf.head()
```

+++ {"id": "1tSmHpDRfd9c", "colab_type": "text"}

This dataset is actual data from the [World Bank Database](https://data.worldbank.org/) corresponding to the different Forest Areas in years 2013 to 2015, the Average Precipitation in 2014 and the Annual Freshwater Withdrawals in 2014. We also have two other variables, one with the names of the countries and another with the continent number following the code

 * 1: Africa
 * 2: America
 * 3: Asia
 * 4: Australia
 * 5: Europe

+++ {"id": "y-XRFSZ0h_xk", "colab_type": "text"}

## 2.- Numerical Data

The descriptive analysis we are going to perform is both: graphical and analytical. For this type of data we are going to use the following information

 * Graphical:
   * Histogram
   * Boxplot
   * Scatterplot
 * Analytical Measures:
   * Central Tendency
     * Mean
     * Median
     * Mode
   * Variability
     * Variance
     * Standard Deviation
     * IQR
   * Shape
     * Skewness
     * Kurtosis
   * Association
     * Covariance
     * Correlation
     
In particular we are going to focus in the `forar2014` variable and we are going to describe it using these different pieces of information

+++ {"id": "_suo3dmJzBY2", "colab_type": "text"}

### 2.1.- Graphical Analysis

 Let's begin with the graphical analysis

+++ {"id": "BMjyDyPIvYCJ", "colab_type": "text"}

#### 2.1.1.- Histograms

Remember that the histogram is an approximated description of the variable: along its construction we make different assumptions that will produce different results. 

In Python all these decisions can be left to the compiler by default, as the boundaries of the classes or the number of classes, for example. We can play with all of them but it is usually not recommended.

The following graph shows how different the histogram can be if we choose a different number of classes

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 405
colab_type: code
executionInfo:
  elapsed: 32228
  status: ok
  timestamp: 1569245421309
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: o_pCoW5PnCuK
outputId: 603d158b-4a7e-4a0c-e767-9fbffe32dffb
---
binSet = [5, 10, 20, 40]
coords = [(0,0), (0,1), (1,0), (1,1)]

plt.style.use("seaborn")

for i,j in zip(binSet, coords):
  plt.subplot2grid((2,2),(j[0], j[1]))
  mydf.forar2014.hist(bins = i, ec = "white", density = True)
  plt.title("bins = " + str(i))


plt.tight_layout()
plt.show()
```

+++ {"id": "k2RoGbcdw0gc", "colab_type": "text"}

we see that while for n=5 we have a unimodal right-skewed desitribution, for n=10, we may say that the distribution is bimodal, and in fact the value around 35 gains importance in subsequent plots.

The number of bins in matplotlib is found using `numpy.histogram` which sets the number of bins to 10 by default but it also allows for predefined forms, as those of Sturges, Rice or Scott (among others). If we use some of these *optimized* forms we obtain more or less the same structure

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 405
colab_type: code
executionInfo:
  elapsed: 32956
  status: ok
  timestamp: 1569245422122
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: _yt5Qt8Kx8jx
outputId: cccf96d9-2435-46ed-9bb3-9d0b84d593b9
---
binSet = ["auto", "sturges", "scott", "rice"]
coords = [(0,0), (0,1), (1,0), (1,1)]

plt.style.use("seaborn")

for i,j in zip(binSet, coords):
  plt.subplot2grid((2,2),(j[0], j[1]))
  mydf.forar2014.hist(bins = i, ec = "white", density = True)
  plt.title("bins = " + i)


plt.tight_layout()
plt.show()
```

+++ {"id": "gbeSfJgEywXd", "colab_type": "text"}

#### 2.1.2.- Boxplot

The boxplot, also known as box-and-whiskers plot, is a representation of the robust features of the distribution. In then we see:

 * The **median**, as the line inside the box
 * The **first** and **third quartiles**, as the limits of the box
 * The **Tukey's limits**, as the whiskers (in some cases the whiskers will show the maximum and minimum values of our distribution but this will **not** be our cases)

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 347
colab_type: code
executionInfo:
  elapsed: 32911
  status: ok
  timestamp: 1569245422134
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: fhYBx_MYyvf-
outputId: 26c8399f-bfa9-4235-9df9-a190913e458e
---
plt.boxplot(mydf.forar2014,
            patch_artist = True,
            showmeans = True,
            widths = 0.6,
            whis = 1.5,
            flierprops = dict(marker = 'o',
                              markerfacecolor = 'red'),
            labels = ["Forest Area 2014"])

plt.show()
```

+++ {"id": "cgzOuQOCy0uK", "colab_type": "text"}

#### 2.1.3.- Scatterplot

The scatterplot is the representation of a bidimensional distribution, which implies that we need to use two of the variables of the dataset.

Since these plots are mostly used to have a visual inspection of the association and relationship between two variables, when we choose them we must decide which is going to be the dependent and which the independent.

For example, in the following case

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 361
colab_type: code
executionInfo:
  elapsed: 33258
  status: ok
  timestamp: 1569245422537
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: 9O7XDj9Ly255
outputId: b9c36820-b4d5-4955-e304-f32e1e144602
---
plt.scatter(x = "forar2014", y = "avprec2014", data = mydf)
plt.xlabel("Forest Area")
plt.ylabel("Average Precipitation")

plt.show()
```

+++ {"id": "x3_xMUy1m3mL", "colab_type": "text"}

we are explicitely saying that the Average Precipitation depends on the Forest Area and not the other way around.

The way these graphs are read go in three different sides:

 * From the **Association**, where we just see *if* there exists any dependency between both variables. In our case we see that when the forest area increases, the average precipitation increases, so we see a positive association then **covariance** will be different from zero and positive
 * From the **Correlation**, where we measure the strength of the linear association between the variables. In our case, since the points are not too aligned we may expect a positive but weak linear correlation
 * From the **relationship**, where we determine the formal functional form that may relate both variables. In this case a straight line can be used (although it will not be a good description since correlation is weak)
 
There are two quantitites we can find in this context: covariance and correlation. In Python we find them as follows: for the covariance we use the `cov()` function which returns the **covariance matrix**, i.e. one with the following structure

\begin{equation}
\begin{pmatrix} s_x^2 & s_{xy} \\
s_{xy} & s_y^2\end{pmatrix}
\end{equation}

i.e. the diagonal elements of the matrix are the variances of the variables and the off-diagonal are the covariance. Remember that this covariance is given by

\begin{equation}
s_{xy} = \frac{1}{n-1}\sum_{i=1}^n(x_i-\bar x)(y_i-\bar y)
\end{equation}


In our case we find

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 112
colab_type: code
executionInfo:
  elapsed: 33226
  status: ok
  timestamp: 1569245422541
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: wZS8434Aosjy
outputId: 44febb04-98ef-41d7-9b5b-bead8bc3e20d
---
mydf[["forar2014","avprec2014"]].cov()
```

+++ {"id": "k1IXJH0gpsZw", "colab_type": "text"}

we see that covariance is 9615.44. Remember that the magnitude of the covariance is not relevant, so this values does not imply that there is a high association.

To find the correlation we use the `corr()` function which returns rhe **correlation matrix**, which has the structure

\begin{equation}
\begin{pmatrix} 1 & r \\
r & 1\end{pmatrix}
\end{equation}

i.e. its diagonal elements are always 1 and the off-diagonal are the linear correlation of the corresponding variables, shich is given by

\begin{equation}
r = \frac{s_{xy}}{s_x\cdot s_y}
\end{equation}

Let's see it

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 112
colab_type: code
executionInfo:
  elapsed: 33187
  status: ok
  timestamp: 1569245422544
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: _ZgzZDU9qMin
outputId: 1bd55443-6943-4cbc-d222-0d3555365e0c
---
mydf[["forar2014","avprec2014"]].corr()
```

+++ {"id": "qQmB5wH1qoXj", "colab_type": "text"}

remember that the strength is actually measured using $r^2$, then 

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 112
colab_type: code
executionInfo:
  elapsed: 33498
  status: ok
  timestamp: 1569245422899
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: z2D3gKcjqu0v
outputId: ec9d906e-8b26-4ef0-aa49-4e96c4332a10
---
mydf[["forar2014","avprec2014"]].corr()**2
```

+++ {"id": "Usj8oDIZqxOH", "colab_type": "text"}

we see a correlation of 0.3255 which is, as expected, a weak one.

+++ {"id": "JPZ4ZHgWy3H3", "colab_type": "text"}

### 2.2.- Analytical Measures

Let's see how to use methods in the pandas dataframe to obtain all the different quantities. In general we can use the `describe()` method to find most of them

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 183
colab_type: code
executionInfo:
  elapsed: 33467
  status: ok
  timestamp: 1569245422901
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: P5V1xt7H98XV
outputId: ed93c9ff-ac9a-4443-a427-bafb1350a4cd
---
mydf["forar2014"].describe()
```

+++ {"id": "HS2zxFD7-ezh", "colab_type": "text"}

or, if we want to find these summaries by continents we can use the `groupby()` function

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 238
colab_type: code
executionInfo:
  elapsed: 33436
  status: ok
  timestamp: 1569245422902
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: qMsKzujI-oej
outputId: ff09774b-ba47-4b97-e794-f3c96df05c35
---
mydf.groupby("continent")["forar2014"].describe()
```

+++ {"id": "DLbuCwaT-xU6", "colab_type": "text"}

which returns a dataframe with indexes given by the continent variable (do not confuse the fact that it begins by 1 because it is Africa with the 0-based index)

+++ {"id": "g0Mpzl2H880Y", "colab_type": "text"}

#### 2.2.1.- Central Tendency

Roughly speaking we can say that these are the measures around which we should describe the distribution. However, not all of them are relevant in all the situations and, in fact, some should not be used in others.

+++ {"id": "ailHsBNB9V7U", "colab_type": "text"}

##### 2.2.1.1.- Mean

Or, to be more precise, the **arithmetic mean**. This is the usual average on a set of independent values

\begin{equation}
\bar x = \frac{1}{n}\sum_{i=1}^n x_i
\end{equation}

which can be written in terms of the total and/or relative frequencies. 

In Python we will just use the method `mean()` directly over the dataframe (or any subset of it), then

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33413
  status: ok
  timestamp: 1569245422904
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: 3a5BKBTu_WOk
outputId: 43290c43-05c4-4826-ffd9-89919b8275fc
---
forar_mean = mydf["forar2014"].mean()
forar_mean 
```

+++ {"id": "l-zHZlyZ_ty6", "colab_type": "text"}

We must be careful with the mean since, despite its great mathematical properties, it has some big drawbacks that can make it not recommended, irrelevant or even useless to describe the distribution. Cases are:

 * Presence of outliers in the distribution
 * Not unimodal distribution
 * Asymmetric distribution
 
in countinuous distributions the problem with the outliers can be so deep as to render the mean inexistent (and the standard deviation and any other moments of the distribution)

+++ {"id": "0AZO4bJs_qdC", "colab_type": "text"}

##### 2.2.1.2.- Median

The median is, by definition, the midpoint of the distribution then it is a more robust measure against most of the problems of the mean. However, the mathematical properties of the median are not so nice and its treatment becomes less straightforward.

To find it in Python we use the `median()` method of the pandas dataframe, then

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33392
  status: ok
  timestamp: 1569245422906
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: w41A_4ljBkEJ
outputId: e15050e1-ebad-4fce-ef41-12316024c7fd
---
forar_med = mydf["forar2014"].median()
forar_med
```

+++ {"id": "CtQbqy9_vri1", "colab_type": "text"}

Note that, just as the mean, the median may not be an actual number in the distribution of values. This will depend on whether the number of observations is even or odd. To find it in general cases we first need to locate its position in a reordered from low to high set, then we determine its value.

Let's see this with a small example

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33364
  status: ok
  timestamp: 1569245422906
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: aMEtmxDdwGF2
outputId: 75dc12b5-6186-49ec-aca2-8746dd0c085e
---
random.seed(101)
example = np.random.randint(1, 20, 13)
example
```

+++ {"id": "fzOonar2xIvx", "colab_type": "text"}

once we have the data, we determine the location o the median

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33315
  status: ok
  timestamp: 1569245422909
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: IGj7ThsQwxwY
outputId: 72d5ae09-5f47-4aae-dc9c-f401bc17ffcb
---
if len(example) % 2 == 0:
  med_pos = int(len(example)/2) + 0.5
else:
  med_pos = int((len(example) - 1)/2) + 1

print("The location of the median is the in the " + str(med_pos) + "th observation")
```

+++ {"id": "MBGr7GJvyS56", "colab_type": "text"}

now we must reorder from low to high

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33266
  status: ok
  timestamp: 1569245422909
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: rBn8wtDcyV7C
outputId: eac1ecca-5fbd-404c-f580-abd788643883
---
ord_ex = np.sort(example)
ord_ex
```

+++ {"id": "E2_0TJIHyWKn", "colab_type": "text"}

and now we can find the value of the median

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33218
  status: ok
  timestamp: 1569245422911
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: vONoUdmWyY9J
outputId: 16b2daae-2346-4673-902a-f0f36f5b8221
---
if len(example) % 2 == 0:
  ex_median = (ord_ex[int(np.ceil(med_pos)) - 1] + ord_ex[int(np.floor(med_pos)) - 1])/2
else:
  ex_median = ord_ex[int(np.floor(med_pos)) - 1]

print("The median is " + str(ex_median))
```

+++ {"id": "wLNRA_W9DTD4", "colab_type": "text"}

##### 2.2.1.3.- Mode

The mode is the most common observation and it may or may no extist. We say that it does not exist when all the observations have the same frequency. In other cases we can find **unimodal**, **bimodal**, **trimodal**,... Distributions.

The function the find this in Python is `mode()`. In our case it will return all the values of the distribution (check it!), a common situation for continuous variables. However, we can use the example data set for the median and see

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 81
colab_type: code
executionInfo:
  elapsed: 33172
  status: ok
  timestamp: 1569245422912
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: spqro9R06_-I
outputId: 4d8292fe-7f00-41ad-dce1-7805ef315719
---
pd.DataFrame(example).mode()
```

+++ {"id": "Ylxv9_KW75CX", "colab_type": "text"}

#### 2.2.2.- Variability Measures

Let's now see how to evaluate the *spreading* of the values. We are going to use two main quantitites:

 * The **standard deviation** if the relevant central tendency is the mean
 * The **IQR** if the relevant central tendency is the median

+++ {"id": "viRYi5WlCM0B", "colab_type": "text"}

##### 2.2.2.1.- Standard Deviation

The standard deviation is defined as the square root of the quasi-variance (the actual difference bewteen variance and quasi-variance will only be clear once we see estimation theory so by now it remains as just a definition). It is defined in a sample as

\begin{equation}
s_x = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2}
\end{equation}

then it can be seen as the average distance to the mean. In symmetric distributions it denotes the distances from the mean value (both sides) where we will find most of the values of the distribution.

In Python we use the `std()` function to find it

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33121
  status: ok
  timestamp: 1569245422913
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: Az5cl_JFDTTW
outputId: 7caf1c49-f521-4871-f8c7-1370813e0a1d
---
forar_sd = mydf["forar2014"].std()
forar_sd
```

+++ {"id": "oBbqbRQREBXE", "colab_type": "text"}

##### 2.2.2.2.- IQR

The Interquartile Range is the distance from the first to the third quartile of the distribution. It represents the range where we can find the middle 50% of the values of the distribution.

To compute it we follow

\begin{equation}
IQR = Q_3 - Q_1
\end{equation}

then we may just find the corresponding quartiles (or percentiles) and substract them

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33078
  status: ok
  timestamp: 1569245422915
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: DYa8TAAmGTI-
outputId: 939c5edd-4759-4a79-9b31-104c22e91338
---
forar_iqr = mydf["forar2014"].quantile(0.75) - mydf["forar2014"].quantile(0.25)
forar_iqr
```

+++ {"id": "eoVVIY7BG11c", "colab_type": "text"}

Since we have not seen the **scipy** package, we are not using it yet. However, let's mention that there is an `iqr()` function in it that can be used to obtain the same result

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33435
  status: ok
  timestamp: 1569245423324
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: 4aTuahSTGiim
outputId: c39191ee-41be-4fb8-b627-1e158ae4fc3c
---
import scipy.stats as ss
ss.iqr(mydf["forar2014"])
```

+++ {"id": "Vyk2kpmrHYVZ", "colab_type": "text"}

#### 2.2.3.- Shape Measures

As we know, the central tendency and variability measures are definitely not enough since the same values may correspond to completely different distributions

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 369
colab_type: code
executionInfo:
  elapsed: 33934
  status: ok
  timestamp: 1569245423869
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: rtqg__JHI6P4
outputId: 1d6818e5-3237-4b17-c579-ec46c03e978d
---
from scipy.stats import norm
x1 = np.arange(-3, 3, 0.01)
y1 = norm(0,1).pdf(x1)

x2 = np.array([-1, 0, 1])
y2 = np.array([1, 1 , 1])


plt.figure(figsize = (10,5))

plt.subplot(121)
_ = plt.plot(x1, y1)
_ = plt.xlabel("Values")
_ = plt.ylabel("Density")
_ = plt.title("mean = 0, sd = 1")

plt.subplot(122)
_ = plt.bar(x2, y2, width = 0.25)
_ = plt.xlabel("Values")
_ = plt.ylabel("Frequency")
_ = plt.title("mean = 0, sd = 1")

plt.tight_layout()
plt.show()
```

+++ {"id": "xp3Dp3N0MLzN", "colab_type": "text"}

It is clear that we need other quantities to determine how the distribution is. These are the shape measures, in particular we will see

 * The **skewness**, which lets us find if there is contribution from long tails to one of the sides of the mean, then
 
 
|      <0     |     0     |      >0      |
|-------------|-----------|--------------|
| left-skewed | symmetric | right-skewed |
 

where *left-skewed* means that there is a tail to the left of the mean and the same interpretation for the right side.
 
 * The **kurtosis**, which tells us how heavy are the tails of the distribution, i.e. if there is a significant number of outliers. We do not compute this as an absolute value, but compared to the normal distribution, then 
 
|      <0     |     0    |     >0      |
|-------------|----------|-------------|
| light-tails | no-tails | heavy-tails |
 
where *no-tails* means "same tails as the normal distribution", i.e. a normal number of outliers.

To find the skewness we use the `skew()` function as

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33892
  status: ok
  timestamp: 1569245423872
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: ACAQ5aP8JHP5
outputId: 17bc486d-89b7-4b0d-b7b6-9a43294d4b8c
---
forar_skew = mydf["forar2014"].skew()
forar_skew
```

+++ {"id": "KAVZCe1oOZ9f", "colab_type": "text"}

while for the kurtosis we use the `kurt()` function

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
executionInfo:
  elapsed: 33853
  status: ok
  timestamp: 1569245423875
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: AqAxFdV6OAPB
outputId: 44e9270a-9126-40fb-8e99-8b3cb043b119
---
forar_kurto = mydf["forar2014"].kurt()
forar_kurto
```

+++ {"id": "z5PJ-Ge4Pfp9", "colab_type": "text"}

### Summary

Let's make a brief summary of the values we have found

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 128
colab_type: code
executionInfo:
  elapsed: 33819
  status: ok
  timestamp: 1569245423882
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: 1snQIuyPPsJy
outputId: 2531afe7-1577-4b79-84f8-e41975129d29
---
print("The MEAN value is " + str(round(forar_mean, 2)))
print("The MEDIAN value is " + str(round(forar_med, 2)))
print("The STANDARD DEVIATION value is " + str(round(forar_sd, 2)))
print("The IQR value is " + str(round(forar_iqr, 2)))
print("The SKEWNESS value is " + str(round(forar_skew, 2)))
print("The KURTOSIS value is " + str(round(forar_kurto, 2)))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 404
colab_type: code
executionInfo:
  elapsed: 34209
  status: ok
  timestamp: 1569245424325
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: HM9sLyroQrXB
outputId: d5664a15-b8e4-49c6-be2a-113f9a5ff527
---
plt.suptitle("Forest Area Distribution", fontsize = 20)

plt.subplot2grid((1,2), (0,0))
plt.hist(mydf["forar2014"],
         color = "lightgreen",
         ec = "darkgreen",
         density = True)

plt.plot(np.arange(-3, 90, 0.01),
         norm.pdf(np.arange(-3, 90, 0.01),
                  forar_mean,
                  forar_sd),
         color = "Black",
         lw = 1)
plt.vlines(x=forar_mean, ymin=0, ymax= 0.025, label="mean", color="red")
plt.vlines(x=forar_med, ymin=0, ymax= 0.025, label="median", color="blue")
plt.hlines(xmin = forar_mean - forar_sd, 
           xmax= forar_mean + forar_sd,
           y = 0.011,
           label="st. deviation", color="darkorange")

plt.title("Histogram", fontsize = 15)
plt.xlabel("Values", fontsize = 15)
plt.ylabel("Density", fontsize = 15)
plt.legend(loc = "best")

plt.subplot2grid((1,2), (0,1))
plt.boxplot(mydf["forar2014"],
            patch_artist = True,
            showmeans = True,
            widths = 0.6,
            whis = 1.5,
            labels = ["Forest Area"],
            boxprops = dict(facecolor = "lightgreen"),
            flierprops = dict(marker = 'o',
                              markerfacecolor = 'red'))
plt.title("Boxplot", fontsize = 15)

plt.annotate("Outlier", xytext = (0.7, 88), xy = (0.98, 88), 
             ha = "center", 
             va = "top", 
            arrowprops= dict(arrowstyle = "->", 
                             connectionstyle = "angle3", 
                             color = "black" ))

plt.show()
```

+++ {"id": "_0GUjaGOVMlw", "colab_type": "text"}


---

Given this summary, answer the following questions

 1.- From both, the numerical value and the graphs, describe the skewness of the distribution
 
 2.- From both, the numerical value and the graphs, interpret the value of the kurtosis
 
 3.- Which central tendency measure would you use to describe the distribution? Why?
 
 4.- Which variability measure would you use to describe the distribution? Why?
 
 5.- Explain the outliers of the distribution
 
 6.- Considering that the variable we are using is the forest area in different countries around the world, describe it using the previous information.

---


+++ {"id": "KLyZP3SaZbXN", "colab_type": "text"}

## 3.- Categorical Variables

The main point we must keep in mind when we work with categorical variables is that we cannot compute usual values as mean or standard deviation since it does not make any sense: think of the mean hair color in a group of people...

This seems to confuse some people when we work with categorical variables which have been transformed into numbers, either for conveniences as when we assign (0,1) to a (head, tails) flip of a coin or when there exists a natural ordering as a ranking or a questionnaire ranging from 1 to 5. When in doubt, try to make sense of the difference between two of the values in different positions: we can substract 5 and 3 or 10 and 8 and we obtain a number meaning exactly the same. If you do this same in the questionnaire you will see that it does not have any meaning.

There is, however, one quantitiy we may still find for these categorical variables: the frequency, which at the end will become the **proportion**.

Now, in Python there is no native way of working with categories since there is no such data type. However, pandas introduces it and we are going to see how to work with it (a bit) here. 

There are different approaches here, but we can use the `Categorical()` function to generate a pandas series of categorical nature

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 54
colab_type: code
executionInfo:
  elapsed: 34605
  status: ok
  timestamp: 1569245424757
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: Nf7hThgVrkpL
outputId: b644a7e8-0713-4463-e169-1342f7ddbbe3
---
pd.Categorical(["yes", "no", "no", "yes", "yes"], categories = ["yes", "no"])
```

+++ {"id": "8TAdHZinrk9S", "colab_type": "text"}

Let's now create a random dataset with a set of binary variables

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 206
colab_type: code
executionInfo:
  elapsed: 34581
  status: ok
  timestamp: 1569245424759
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: 0fIfcUI_ni3d
outputId: 5f6c0b4c-7c97-4f09-fbab-6dba3a742abd
---
random.seed(101)
catdf = pd.DataFrame({"married": pd.Categorical(np.random.randint(0,2,100)),
                      "siblings": pd.Categorical(np.random.randint(0,2,100)),
                      "female": pd.Categorical(np.random.randint(0,2,100))})
catdf.head()
```

+++ {"id": "7eDzuD5HsY5H", "colab_type": "text"}

As expected, if we use the `describe()` function here, we will find that there is nothing similar to the output we obtained with numerical variables (to actually compare, drop the pd.Categorical and see the output) 

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 175
colab_type: code
executionInfo:
  elapsed: 34557
  status: ok
  timestamp: 1569245424762
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: CgDgXWepsZMY
outputId: 56054879-ce7e-4a55-9b6d-cb99fa87039e
---
catdf.describe()
```

+++ {"id": "2JgFp_QGtZwv", "colab_type": "text"}

the most we get are the frequencies of the most repeated category. In this case it is useful because we only have two of them, but with 3 or more, this values is not meaningful.

If we want an individual summary we need the `value_counts()` method

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 72
colab_type: code
executionInfo:
  elapsed: 34539
  status: ok
  timestamp: 1569245424765
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: QyA47nbZuDcr
outputId: efcb8095-683f-481c-d167-d88a4ae3f214
---
catdf["married"].value_counts()
```

+++ {"id": "egSfRKGQuhb_", "colab_type": "text"}

### 3.1.- Contingency Tables

When we have more than one categorical variable and we want to summarize the frequencies of the observations fitting into all the possible combinations of categories we need a **contingency table**. 

In pandas we have the `crosstab()` function that finds these tables, for example, the contingency table of `female` and `married` variables is

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 144
colab_type: code
executionInfo:
  elapsed: 34523
  status: ok
  timestamp: 1569245424770
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: kFfikwckqVwt
outputId: bc657560-bb43-4bfe-cb08-232ed1f2df54
---
pd.crosstab(catdf.female, catdf.married)
```

+++ {"id": "U6zvmoYCvBog", "colab_type": "text"}

which means that there are 23 people in the sample which are not-female and not-married, and so on.

If we need this table in proportions, we have the `normalize` argument that let's us find the corresponding table

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 144
colab_type: code
executionInfo:
  elapsed: 34505
  status: ok
  timestamp: 1569245424773
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: h8pCdeCdvUXz
outputId: 934c908c-9477-441c-861b-c3fe63d50957
---
pd.crosstab(catdf.female, catdf.married, normalize = True)
```

+++ {"id": "V5v_dqlMvnVU", "colab_type": "text"}

Since the output of this function is a pandas data frame, we can use all the methods of this class, in particular we can plot it directly

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 405
colab_type: code
executionInfo:
  elapsed: 34864
  status: ok
  timestamp: 1569245425165
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: Mhvw1eKYv4Ne
outputId: 486c43dd-c705-40fc-81b9-7cf3ba05213c
---
labels = [0, 1]

fig, axes = plt.subplots(nrows=1, ncols=2)

pd.crosstab(catdf.female, catdf.married).plot(kind = "bar", ax = axes[0])
axes[0].set_xticklabels(labels, rotation = 0)
axes[0].set_xlabel("Female")

pd.crosstab(catdf.female, catdf.married).plot(kind = "bar", ax = axes[1], stacked = True)
axes[1].set_xticklabels(labels, rotation = 0)
axes[1].set_xlabel("Female")

plt.tight_layout()
plt.show()
```

+++ {"id": "G-Svhe3wvj48", "colab_type": "text"}

In many situations we will find that we have more than one categorical variable and we want to find all the crossed frequencies, the procedure is exactly the same but passing a list as the set of classifiers, then

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 175
colab_type: code
executionInfo:
  elapsed: 34805
  status: ok
  timestamp: 1569245425171
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: EwiIhaG0qCiv
outputId: ef7e5ebe-e725-47c5-d48a-e503f139e884
---
pd.crosstab(catdf.female, [catdf.married, catdf.siblings])
```

+++ {"id": "5JW7fkUV0USI", "colab_type": "text"}

so the 17 means that there are 17 people in this sample who are female, are married and have siblings.

Just as before, we can plot this contingency table directly using the dataframe methods

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 405
colab_type: code
executionInfo:
  elapsed: 35507
  status: ok
  timestamp: 1569245425915
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: 0aFUcwT60lJb
outputId: 835b4c2d-5e0e-4d94-9893-7dbb4f95d767
---
labels = [0, 1]

fig, axes = plt.subplots(nrows=1, ncols=2)

pd.crosstab(catdf.female, [catdf.married, catdf.siblings]).plot(kind = "bar", ax = axes[0])
axes[0].set_xticklabels(labels, rotation = 0)
axes[0].set_xlabel("Female")

pd.crosstab(catdf.female, [catdf.married, catdf.siblings]).plot(kind = "bar", ax = axes[1], stacked = True)
axes[1].set_xticklabels(labels, rotation = 0)
axes[1].set_xlabel("Female")
axes[1].legend(loc = "best")

plt.tight_layout()
plt.show()
```
