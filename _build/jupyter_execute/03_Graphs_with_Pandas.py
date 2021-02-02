# Graphics with Pandas

We have seen how to use **matplotlib** to generate the basic graphs we may need in our statistical analysis. However, since the main data structure we are going to work with is the data frame which is defined in **pandas**, we may want to fully use this library to make the graphs.

If we use this pandas approach, we must understand that each of the graphs are just methods associated with the data frame structure and then they are called directly from the data frame. Let's see it.

## 1.- The Data

We already know how to connect to our drive, then let's load a data set from there and work with it

from google.colab import drive
drive.mount('mydrive')

now we load the dataset

import pandas as pd

remember that we need the pandas library!

mydf = pd.read_csv("/content/mydrive/My Drive/Statistics and Data Analysis - 2019/Data Sets/forestarea.csv")

mydf = pd.read_csv("/content/sample_data/california_housing_train.csv")
mydf.head()

## 2.- Histograms

Before going on, let's set the seaborn style for all our plots

import matplotlib.pyplot as plt
plt.style.use("seaborn")

Let's take as variable `total_rooms`, then its histogram is

mydf["total_rooms"].hist(color = "lightgreen", ec = "darkgreen")

plt.xlabel("Total Rooms")
plt.ylabel("Frequency")
plt.show()

since we are using matplotlib, everything we know about it can be used in these plots

## 3.- Boxplots

Boxplots have a particularity: the function does not accept a Series format, which means that we cannot use the single bracket notation (go back to the selection in pandas section) and must use double brackets

mydf[["total_rooms"]].boxplot(patch_artist = True,
                              showmeans = True,
                              widths = 0.6,
                              flierprops = dict(marker = 'o',
                                                markerfacecolor = 'red'))

## 4.- Bar Plots

Consider the same example we had in matplotlib in which we had a dataframe grouped by gender and the activity level, the dataframe was (since it was randomly generated without a seed the values may change)

df = pd.DataFrame({"females": [69, 81, 85], 
                   "males": [94, 95, 76]}, 
                  index = ["low", "mid", "high"])
df

For a data frame we can call directly to the **plot** function and then the corresponding graph, in this case the **bar**. Note that this can also be done using `plot(kind = "bar")`

df.plot.bar()

plt.xticks(rotation = 0)
plt.xlabel("Activity Level")
plt.ylabel("Frequency")

plt.show()

you can go back to matplotlib and see how pandas simplfies this considerably

## 5.- Scatter Plots

Just as we have done with the bar plots, the plot function can be used to call for other graphs. Of particular importance is the scatterplot which can be easily found as

mydf.plot.scatter("median_income", "median_house_value")
plt.show()

and now we are free to use all the layers we know from matplotlib