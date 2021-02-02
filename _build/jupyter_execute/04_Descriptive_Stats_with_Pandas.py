# Descriptive Statistics with Pandas


In this document we are going to see how to compute the basic numerical values needed to perform a descriptive analysis of a data set.

We are going to split the document in two main parts:

 * One for **Numerical** variables
 * One for **Categorical** variables
 
most of all because categorical variables are slightly subtle in Python. Let's load all the different packages we need for **all** the analysis to be made

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

## 1.- The Data

Let's connect to our drive and load a dataset

from google.colab import drive
drive.mount('mydrive')

now we can load the dataset

mydf = pd.read_csv("/content/mydrive/My Drive/IE Bootcamp - Math & Stats /data/forestarea.csv")
mydf.head()

This dataset is actual data from the [World Bank Database](https://data.worldbank.org/) corresponding to the different Forest Areas in years 2013 to 2015, the Average Precipitation in 2014 and the Annual Freshwater Withdrawals in 2014. We also have two other variables, one with the names of the countries and another with the continent number following the code

 * 1: Africa
 * 2: America
 * 3: Asia
 * 4: Australia
 * 5: Europe

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

### 2.1.- Graphical Analysis

 Let's begin with the graphical analysis

#### 2.1.1.- Histograms

Remember that the histogram is an approximated description of the variable: along its construction we make different assumptions that will produce different results. 

In Python all these decisions can be left to the compiler by default, as the boundaries of the classes or the number of classes, for example. We can play with all of them but it is usually not recommended.

The following graph shows how different the histogram can be if we choose a different number of classes

binSet = [5, 10, 20, 40]
coords = [(0,0), (0,1), (1,0), (1,1)]

plt.style.use("seaborn")

for i,j in zip(binSet, coords):
  plt.subplot2grid((2,2),(j[0], j[1]))
  mydf.forar2014.hist(bins = i, ec = "white", density = True)
  plt.title("bins = " + str(i))


plt.tight_layout()
plt.show()

we see that while for n=5 we have a unimodal right-skewed desitribution, for n=10, we may say that the distribution is bimodal, and in fact the value around 35 gains importance in subsequent plots.

The number of bins in matplotlib is found using `numpy.histogram` which sets the number of bins to 10 by default but it also allows for predefined forms, as those of Sturges, Rice or Scott (among others). If we use some of these *optimized* forms we obtain more or less the same structure

binSet = ["auto", "sturges", "scott", "rice"]
coords = [(0,0), (0,1), (1,0), (1,1)]

plt.style.use("seaborn")

for i,j in zip(binSet, coords):
  plt.subplot2grid((2,2),(j[0], j[1]))
  mydf.forar2014.hist(bins = i, ec = "white", density = True)
  plt.title("bins = " + i)


plt.tight_layout()
plt.show()

#### 2.1.2.- Boxplot

The boxplot, also known as box-and-whiskers plot, is a representation of the robust features of the distribution. In then we see:

 * The **median**, as the line inside the box
 * The **first** and **third quartiles**, as the limits of the box
 * The **Tukey's limits**, as the whiskers (in some cases the whiskers will show the maximum and minimum values of our distribution but this will **not** be our cases)

plt.boxplot(mydf.forar2014,
            patch_artist = True,
            showmeans = True,
            widths = 0.6,
            whis = 1.5,
            flierprops = dict(marker = 'o',
                              markerfacecolor = 'red'),
            labels = ["Forest Area 2014"])

plt.show()

#### 2.1.3.- Scatterplot

The scatterplot is the representation of a bidimensional distribution, which implies that we need to use two of the variables of the dataset.

Since these plots are mostly used to have a visual inspection of the association and relationship between two variables, when we choose them we must decide which is going to be the dependent and which the independent.

For example, in the following case

plt.scatter(x = "forar2014", y = "avprec2014", data = mydf)
plt.xlabel("Forest Area")
plt.ylabel("Average Precipitation")

plt.show()

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

mydf[["forar2014","avprec2014"]].cov()

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

mydf[["forar2014","avprec2014"]].corr()

remember that the strength is actually measured using $r^2$, then

mydf[["forar2014","avprec2014"]].corr()**2

we see a correlation of 0.3255 which is, as expected, a weak one.

### 2.2.- Analytical Measures

Let's see how to use methods in the pandas dataframe to obtain all the different quantities. In general we can use the `describe()` method to find most of them

mydf["forar2014"].describe()

or, if we want to find these summaries by continents we can use the `groupby()` function

mydf.groupby("continent")["forar2014"].describe()

which returns a dataframe with indexes given by the continent variable (do not confuse the fact that it begins by 1 because it is Africa with the 0-based index)

#### 2.2.1.- Central Tendency

Roughly speaking we can say that these are the measures around which we should describe the distribution. However, not all of them are relevant in all the situations and, in fact, some should not be used in others.

##### 2.2.1.1.- Mean

Or, to be more precise, the **arithmetic mean**. This is the usual average on a set of independent values

\begin{equation}
\bar x = \frac{1}{n}\sum_{i=1}^n x_i
\end{equation}

which can be written in terms of the total and/or relative frequencies. 

In Python we will just use the method `mean()` directly over the dataframe (or any subset of it), then

forar_mean = mydf["forar2014"].mean()
forar_mean 

We must be careful with the mean since, despite its great mathematical properties, it has some big drawbacks that can make it not recommended, irrelevant or even useless to describe the distribution. Cases are:

 * Presence of outliers in the distribution
 * Not unimodal distribution
 * Asymmetric distribution
 
in countinuous distributions the problem with the outliers can be so deep as to render the mean inexistent (and the standard deviation and any other moments of the distribution)

##### 2.2.1.2.- Median

The median is, by definition, the midpoint of the distribution then it is a more robust measure against most of the problems of the mean. However, the mathematical properties of the median are not so nice and its treatment becomes less straightforward.

To find it in Python we use the `median()` method of the pandas dataframe, then

forar_med = mydf["forar2014"].median()
forar_med

Note that, just as the mean, the median may not be an actual number in the distribution of values. This will depend on whether the number of observations is even or odd. To find it in general cases we first need to locate its position in a reordered from low to high set, then we determine its value.

Let's see this with a small example

random.seed(101)
example = np.random.randint(1, 20, 13)
example

once we have the data, we determine the location o the median

if len(example) % 2 == 0:
  med_pos = int(len(example)/2) + 0.5
else:
  med_pos = int((len(example) - 1)/2) + 1

print("The location of the median is the in the " + str(med_pos) + "th observation")

now we must reorder from low to high

ord_ex = np.sort(example)
ord_ex

and now we can find the value of the median

if len(example) % 2 == 0:
  ex_median = (ord_ex[int(np.ceil(med_pos)) - 1] + ord_ex[int(np.floor(med_pos)) - 1])/2
else:
  ex_median = ord_ex[int(np.floor(med_pos)) - 1]

print("The median is " + str(ex_median))

##### 2.2.1.3.- Mode

The mode is the most common observation and it may or may no extist. We say that it does not exist when all the observations have the same frequency. In other cases we can find **unimodal**, **bimodal**, **trimodal**,... Distributions.

The function the find this in Python is `mode()`. In our case it will return all the values of the distribution (check it!), a common situation for continuous variables. However, we can use the example data set for the median and see

pd.DataFrame(example).mode()

#### 2.2.2.- Variability Measures

Let's now see how to evaluate the *spreading* of the values. We are going to use two main quantitites:

 * The **standard deviation** if the relevant central tendency is the mean
 * The **IQR** if the relevant central tendency is the median

##### 2.2.2.1.- Standard Deviation

The standard deviation is defined as the square root of the quasi-variance (the actual difference bewteen variance and quasi-variance will only be clear once we see estimation theory so by now it remains as just a definition). It is defined in a sample as

\begin{equation}
s_x = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2}
\end{equation}

then it can be seen as the average distance to the mean. In symmetric distributions it denotes the distances from the mean value (both sides) where we will find most of the values of the distribution.

In Python we use the `std()` function to find it

forar_sd = mydf["forar2014"].std()
forar_sd

##### 2.2.2.2.- IQR

The Interquartile Range is the distance from the first to the third quartile of the distribution. It represents the range where we can find the middle 50% of the values of the distribution.

To compute it we follow

\begin{equation}
IQR = Q_3 - Q_1
\end{equation}

then we may just find the corresponding quartiles (or percentiles) and substract them

forar_iqr = mydf["forar2014"].quantile(0.75) - mydf["forar2014"].quantile(0.25)
forar_iqr

Since we have not seen the **scipy** package, we are not using it yet. However, let's mention that there is an `iqr()` function in it that can be used to obtain the same result

import scipy.stats as ss
ss.iqr(mydf["forar2014"])

#### 2.2.3.- Shape Measures

As we know, the central tendency and variability measures are definitely not enough since the same values may correspond to completely different distributions

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

forar_skew = mydf["forar2014"].skew()
forar_skew

while for the kurtosis we use the `kurt()` function

forar_kurto = mydf["forar2014"].kurt()
forar_kurto

### Summary

Let's make a brief summary of the values we have found

print("The MEAN value is " + str(round(forar_mean, 2)))
print("The MEDIAN value is " + str(round(forar_med, 2)))
print("The STANDARD DEVIATION value is " + str(round(forar_sd, 2)))
print("The IQR value is " + str(round(forar_iqr, 2)))
print("The SKEWNESS value is " + str(round(forar_skew, 2)))
print("The KURTOSIS value is " + str(round(forar_kurto, 2)))

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

---

Given this summary, answer the following questions

 1.- From both, the numerical value and the graphs, describe the skewness of the distribution
 
 2.- From both, the numerical value and the graphs, interpret the value of the kurtosis
 
 3.- Which central tendency measure would you use to describe the distribution? Why?
 
 4.- Which variability measure would you use to describe the distribution? Why?
 
 5.- Explain the outliers of the distribution
 
 6.- Considering that the variable we are using is the forest area in different countries around the world, describe it using the previous information.

---

## 3.- Categorical Variables

The main point we must keep in mind when we work with categorical variables is that we cannot compute usual values as mean or standard deviation since it does not make any sense: think of the mean hair color in a group of people...

This seems to confuse some people when we work with categorical variables which have been transformed into numbers, either for conveniences as when we assign (0,1) to a (head, tails) flip of a coin or when there exists a natural ordering as a ranking or a questionnaire ranging from 1 to 5. When in doubt, try to make sense of the difference between two of the values in different positions: we can substract 5 and 3 or 10 and 8 and we obtain a number meaning exactly the same. If you do this same in the questionnaire you will see that it does not have any meaning.

There is, however, one quantitiy we may still find for these categorical variables: the frequency, which at the end will become the **proportion**.

Now, in Python there is no native way of working with categories since there is no such data type. However, pandas introduces it and we are going to see how to work with it (a bit) here. 

There are different approaches here, but we can use the `Categorical()` function to generate a pandas series of categorical nature

pd.Categorical(["yes", "no", "no", "yes", "yes"], categories = ["yes", "no"])

Let's now create a random dataset with a set of binary variables

random.seed(101)
catdf = pd.DataFrame({"married": pd.Categorical(np.random.randint(0,2,100)),
                      "siblings": pd.Categorical(np.random.randint(0,2,100)),
                      "female": pd.Categorical(np.random.randint(0,2,100))})
catdf.head()

As expected, if we use the `describe()` function here, we will find that there is nothing similar to the output we obtained with numerical variables (to actually compare, drop the pd.Categorical and see the output)

catdf.describe()

the most we get are the frequencies of the most repeated category. In this case it is useful because we only have two of them, but with 3 or more, this values is not meaningful.

If we want an individual summary we need the `value_counts()` method

catdf["married"].value_counts()

### 3.1.- Contingency Tables

When we have more than one categorical variable and we want to summarize the frequencies of the observations fitting into all the possible combinations of categories we need a **contingency table**. 

In pandas we have the `crosstab()` function that finds these tables, for example, the contingency table of `female` and `married` variables is

pd.crosstab(catdf.female, catdf.married)

which means that there are 23 people in the sample which are not-female and not-married, and so on.

If we need this table in proportions, we have the `normalize` argument that let's us find the corresponding table

pd.crosstab(catdf.female, catdf.married, normalize = True)

Since the output of this function is a pandas data frame, we can use all the methods of this class, in particular we can plot it directly

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

In many situations we will find that we have more than one categorical variable and we want to find all the crossed frequencies, the procedure is exactly the same but passing a list as the set of classifiers, then

pd.crosstab(catdf.female, [catdf.married, catdf.siblings])

so the 17 means that there are 17 people in this sample who are female, are married and have siblings.

Just as before, we can plot this contingency table directly using the dataframe methods

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