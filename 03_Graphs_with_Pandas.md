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

+++ {"id": "Xnlk-gmWujCv", "colab_type": "text"}

# Graphics with Pandas

We have seen how to use **matplotlib** to generate the basic graphs we may need in our statistical analysis. However, since the main data structure we are going to work with is the data frame which is defined in **pandas**, we may want to fully use this library to make the graphs.

If we use this pandas approach, we must understand that each of the graphs are just methods associated with the data frame structure and then they are called directly from the data frame. Let's see it.


+++ {"id": "Po0H-nf-036d", "colab_type": "text"}

## 1.- The Data

We already know how to connect to our drive, then let's load a data set from there and work with it

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 129
colab_type: code
executionInfo:
  elapsed: 142073
  status: ok
  timestamp: 1568641125183
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: je5NSoLI0QH4
outputId: d741c81c-0c41-43af-d0a2-a2bac0655e38
---
from google.colab import drive
drive.mount('mydrive')
```

+++ {"id": "9IB1K1P50Tq7", "colab_type": "text"}

now we load the dataset

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: HRP2xCQl0a1b

import pandas as pd
```

+++ {"id": "_Tb1Xip90bIe", "colab_type": "text"}

remember that we need the pandas library!

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: XD-mWn4gVtKU

mydf = pd.read_csv("/content/mydrive/My Drive/Statistics and Data Analysis - 2019/Data Sets/forestarea.csv")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 226
colab_type: code
executionInfo:
  elapsed: 582
  status: ok
  timestamp: 1567159109992
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: vxLqb9Hg0Vtp
outputId: 865a956f-9eee-47a1-deaf-09845ab0a7ed
---
mydf = pd.read_csv("/content/sample_data/california_housing_train.csv")
mydf.head()
```

+++ {"id": "iQKauFj8075O", "colab_type": "text"}

## 2.- Histograms

Before going on, let's set the seaborn style for all our plots

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: B451rmImj37E

import matplotlib.pyplot as plt
plt.style.use("seaborn")
```

+++ {"id": "bVzs_g4IkNC_", "colab_type": "text"}

Let's take as variable `total_rooms`, then its histogram is

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 361
colab_type: code
executionInfo:
  elapsed: 826
  status: ok
  timestamp: 1567159345351
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: _oNXXjJd0kH-
outputId: 176868d9-d1bc-40b3-b246-13fec30f1eab
---
mydf["total_rooms"].hist(color = "lightgreen", ec = "darkgreen")

plt.xlabel("Total Rooms")
plt.ylabel("Frequency")
plt.show()
```

+++ {"id": "WZO3QRQ4LRyp", "colab_type": "text"}

since we are using matplotlib, everything we know about it can be used in these plots

+++ {"id": "Tkg5Lu3K1iq8", "colab_type": "text"}

## 3.- Boxplots

Boxplots have a particularity: the function does not accept a Series format, which means that we cannot use the single bracket notation (go back to the selection in pandas section) and must use double brackets

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 365
colab_type: code
executionInfo:
  elapsed: 746
  status: ok
  timestamp: 1567163864262
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: Mmsx1ots1Ggx
outputId: 03a0ae06-9826-48d0-cf2c-cdb5b05c35f7
---
mydf[["total_rooms"]].boxplot(patch_artist = True,
                              showmeans = True,
                              widths = 0.6,
                              flierprops = dict(marker = 'o',
                                                markerfacecolor = 'red'))
```

+++ {"id": "9u37p09UMidw", "colab_type": "text"}

## 4.- Bar Plots

Consider the same example we had in matplotlib in which we had a dataframe grouped by gender and the activity level, the dataframe was (since it was randomly generated without a seed the values may change)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 144
colab_type: code
executionInfo:
  elapsed: 713
  status: ok
  timestamp: 1567163429773
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: 0miqjBIuNpwG
outputId: 624c4822-0f13-4b2d-aea9-549dad946397
---
df = pd.DataFrame({"females": [69, 81, 85], 
                   "males": [94, 95, 76]}, 
                  index = ["low", "mid", "high"])
df
```

+++ {"id": "eCKTA86nPJXb", "colab_type": "text"}

For a data frame we can call directly to the **plot** function and then the corresponding graph, in this case the **bar**. Note that this can also be done using `plot(kind = "bar")`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 361
colab_type: code
executionInfo:
  elapsed: 696
  status: ok
  timestamp: 1567163703652
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: ujm4ny2eOLnZ
outputId: 08254b02-3f20-4f85-ecda-23969a5e98cf
---
df.plot.bar()

plt.xticks(rotation = 0)
plt.xlabel("Activity Level")
plt.ylabel("Frequency")

plt.show()
```

+++ {"id": "rwluSe2ARLM3", "colab_type": "text"}

you can go back to matplotlib and see how pandas simplfies this considerably

+++ {"id": "k59yZXj4QrAe", "colab_type": "text"}

## 5.- Scatter Plots

Just as we have done with the bar plots, the plot function can be used to call for other graphs. Of particular importance is the scatterplot which can be easily found as

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 361
colab_type: code
executionInfo:
  elapsed: 1354
  status: ok
  timestamp: 1567164043087
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: oXEs0-tMP1Cw
outputId: ea6a22a9-e507-4294-ad0b-b9aa93b74c06
---
mydf.plot.scatter("median_income", "median_house_value")
plt.show()
```

+++ {"id": "u5cxxuUTQ3C7", "colab_type": "text"}

and now we are free to use all the layers we know from matplotlib
