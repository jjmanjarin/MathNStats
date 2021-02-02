# Introduction to Python


In this introductory document we are going to explore some of the basic operations with Python:

  * Packages and Modules
  * Data Types
  * The Pandas Data Frame
  * Reading and Saving external data
  * Google's Colaboratory Environment

## 1.- What is Python? Why Python?

Python is a programming language that can be either **compiled** or **interpreted**. This means that we can use it both to write complex programs that we compiled in executable files, libraries and other data, but it can also be used to write small scripts that are executed on the run.

Here we will focus on this last execution mode and a very simple and illustrative example of it is letting Python behave as a huge calculator. For example, let's add 10 and 10

10 + 10

## 2.- Packages and Modules

Python is a general purpose programming language. There resides most of its strengths but, at the same time, we cannot have everything we need or may need from the base installation.

There are many different Python projects that have built a set of consistent functions that we can use for our daily work. These functions are packed in external modules that we can load in our projects (we will use the words *module*, *package* and *library* to refer these packs, however there are subtle but important differences bewteen them).

The most important libraries we will use duing this course are:

- numpy (typically as `np`): For numerical anlysis. See [manual](https://www.numpy.org/devdocs/contents.html).
- scipy: For scientific computing. See the [scipy lecture notes](http://scipy-lectures.org/index.html). We will extensively use the submodule `stats` of the scipy library for statistical modeling.
- pandas:   For data structuring and manipulation. See [documentation](http://pandas.pydata.org/pandas-docs/stable/) and [cheatsheet](http://pandas.pydata.org/Pandas_Cheat_Sheet.pdf).
- matplotlib: For 2D plots. See [tutorials](https://matplotlib.org/tutorials/index.html).

The way we load them is through the use of different Python **keywords**, the main one is **import**. Then to load the **pandas** library we would do

```python
import pandas
```

however there are common abbreviations to these packages so that we do not have to write the whole name everytime, then we would do

```python
import pandas as pd
```

now once we want to use any of these pandas functions, say the **Series** one, we must do it as

```python
pd.Series()
```

in such a way that our interpreter knows that we are calling for the Series function inside the Pandas library, i.e. the abbreviation is a reference name for the whole library.

Some other times we do not want or need to import the whole set of functions in the library then we can restrict the importation as much as we want, for example, the **pyplot** module in **matplotlib** can be imported as

```python
import matplotlib.pyplot as plt
```

some others we just want a very small set of functions, for example the ones corresponding to the normal distribution from the **scipy** library, then we do

```python
from scipy.stats import norm
```

This modules structure is a common source of confusion when we take our first steps in Python, and most of our difficulties could be sum up in one single question: "how do I know where is the function I need or even if it exists?" The short answer is *with practise* which implies that many times we have to take a long tour through the documentation or the web looking for some of the answers.

## 3.- Types of Data

Just as with the functions of the base installations, which do not contain all the functions we may ever need, the base data types are rather general and powerful but do not satisfy the needs of data analysis. We are going to explain briefly these base ones because we will use them from time to time and then the ones that come with the modules important for us.

### 3.1.- The Base Library

There are four data types that are predifined in Python

  * Strings
  * Lists
  * Tuples
  * Sets
  * Dictionaries

To generate a new data structure we can use two different approaches:

 * Using the **constructor** of the type. This is a special type of function that will be `str()`, `list()`, `tuple()`, `set()` and `dict()`. This is usually the safest form (mostly at the beginning)
 * Using the **format** of the type. This will be `" "`, `[ ]`, `( )` and `{ }` (with a small subtlety for this last one)

The important observation to keep in mind is that although there are some basic common methods for all the types, some of them are specific of each class. Let's see the basic properties of each type

#### 3.1.1 Strings

A **string** is a literal, then if we write a number, this type represents the text written and not its numeric content (this is important in any arithmetic operation we make with them)

```python
"my number is 2"
```

is equivalent to 

```python
str("my number is 2")
```

a = str("my number is 2")
a

#### 3.1.2 Lists

A **list** is the first important data type in Python. In them we can store under one common name, a **not fixed** amount of sequential values.

The following two forms a equivalent, from the format

```python
[1, 2, "Hello"]
```

or with the constructor

```python
list([1, 2, "Hello"])
```

mylist = list([1, 2, "Hello"])
mylist

In this context, the word sequential means that we can access the information in the list in a sequential form. Lists can also be generated through the use of some list comprehensions as the `for` loops (out of this course)

```python
[x**2 for x in [1,2,3]]
```

We may have multidimensional lists which allow us to store in one single variable tables or higher dimensional structures. For bidimensional tables we may write

```python
[[1, 2, 3], ["a", "b", "c"], [15, 25, 30]]
```

which is a table with three rows and four columns

#### 3.1.3 Tuples

The **tuples** are very similar to lists with one exception: its elements cannot be changed: we cannot add, replace, remove or reorder them in any way.

We may create a tuple with the format

```python
(1, 2, 3)
```

with the constructor

```python
tuple([1, 2, 3])
```

or with the list comprehension together with the constructor

```python
tuple([x**2 for x in [1, 2, 3]])
```

mytuple = tuple([x**2 for x in [1, 2, 3]])
mytuple

```{warning}
The fact that this type cannot be changed can be easily seen if we try to add a new element to it using the index operator: the interpreter will throw an error. For example, the code `mytuple[3] = 2` will return `TypeError: 'tuple' object does not support item assignment`
```

#### 3.1.4 Sets

The **Sets** are used to store data without any order in them. Then we cannot access by index (see later) but by the explicit value of the input. The main advantage with respect to lists and tuples is that sets are way faster.

We can generate sets using the format

```python
{1, 2, 3}
```

or the constructor applied to a string, a list, a tuple or a list comprehension

```python
set([1, 2, 3])
```

obviously, we can generate a list or a tuple using their constructors with a set.

#### 3.1.5 Dictionaries

The **dictionaries** are keyed-sets, this means that in the definition we must incorporate a key for each of the data we want to store.

If we define the data from the format this may be

```python
{1: "John", 2: "Eve"}
```

if we use the constructor we have

```python
dict(a = "John", b = "Eve")
```

in this second case the keys must be *valid* and numbers are not, that's why we have changed them to *a* and *b*. A way to move around that imposibility is using the constructor on the format of a dictionary

```python
dict({1: "John", 2: "Eve"})
```

mydict = dict({1: "John", 2: "Eve"})
mydict

this class comes with some specific methods that we can invoke from the dictionary itself (take a look at the documentation), for example, to know the keys of the dictionary

mydict.keys()

### 3.2.- The Numpy Array

The Numpy array is one of the possible arrays we can find in the Python packages. To use it we need to import the numpy package so let's do it before any other thing

import numpy as np

the constructor for this data type is the `np.array()` function which can be used in the usual way

myarray = np.array([1, 2, 3])
myarray

you see that the output makes explicit reference to the array nature of the dataset.

Now, this structure is mostly like a list: both are iterated, can be sliced, are used to store data and can be indexed. However, while the list is **NOT** a vector and cannot be used in vector operations, the array can and will be used as such. Consider the following examples

np.array([1, 2, 3])/2

in this case we have a vectorized operation where division by 2 is applied to all the elements of the array. Try to do this same operation in a list and you will receive an error: run the following code

```python
[1, 2, 3]/2
```

In any case, you have to keep in mind that the numpy array behaves this way only if it has been defined from a list.

#### 3.2.1 Indexing and Slicing

An important observation is that in numpy the dimensions are known as **axes**. This is very relevant since in many operations you will be required to say explicitely the axis along which you want to perform it. 

Among these operations, the **indexing** and **slicing** are a must-know.

  * **Indexing** refers to the fact that each element in the data structure has an reference index that allow us to gain access to it individually. The only peculiar behaviour is that Python inherits from **C** the **0-based index**, i.e. the index begins at 0 not at 1, then the first element's index is always 0. The index is used with the squared brackets right after the name of the data structure

myarray[0]

> you see that the element with index 0 in `myarray` is the first element, in this case the number 1. We can also have negative indices, implying that we reverse the counting order (always keeping the first element as 0), then

myarray[-1]

> In multidimensional arrays we will find a different index per axis, then

mybidimarray = np.array([[1, 3, 5], [2, 4, 6]])
mybidimarray

> now the index is used as follows: we will have a pair of indices the first one for the first axis, the second one for the second axis (try to say which is going to be the output of the following code without executing it)

```python
mybidimarray[1,2]
```

How do we access more than one element in the data set? Through slicing

  * **Sclicing** refers to the use of the slicing operator "**:**" which is to be read as "init:end". Let's define a longer array and see how to use it

myarray = np.random.random(10)
myarray

> then the elements from the second to the fifth elements are

myarray[1:5]

> If we do not impose any of the boundaries the interpreter considers that we want every element in that direction, then for example, the three first elements are

myarray[:3]

> This can be used in a cumulative way in both boundaries as in

myarray[:1:3]

> in two dimensional arrays, we can slice in exactly the same way but considering the two axes in the data. Let's reshape our data in a 2x5 array

myarray = myarray.reshape(2,5)
myarray

> then to obtain the third column we would do

myarray[:, 2]

### 3.3.- The Pandas Data Frame

The Pandas package contains two relevant structures. One of them are the **series**, which are similar to arrays. In fact these are a sort of enhanced arrays. You can take a look at the documentation to see how they work.

Here we are going to concentrate on the data frames. These are the main data structure we are going to work with and just as the numpy array resembles the list, this data frame resembles the dictionaries, which is particularly important when we try to go through filter and selection operations.

In sort we can say that a data frame is a two dimensional labeled data structure. This is basically what we see in any spreadsheet we have worked with: rows and columns with names and indices.

To generate this structure we need to load the pandas library

import pandas as pd

now we are ready to use the basic constructor of this data type, the `pd.DataFrame()` function, let's see it using a randomly generated dataset

mydf = pd.DataFrame({"Age": np.random.randint(18, 65, 10), 
                     "Exper": np.random.randint(1, 15, 10), 
                     "Tenure": np.random.randint(1, 5, 10)
                    })
mydf

Note that what we had as keys in the dictionaries are the names of the variables/columns. Also see that in Python we will always have the extra column on the left which corresponds to the index, so now each observation/row will have an explicit index.

#### 3.3.1.- Selecting and Filtering

We have two different subsetting operations:

 * One in which we **select** different variables/columns out of the whole set
 * A second in which we **filter** to keep only some rows/observations based on some conditions

Now that we now the different types of labels in a data frame we must always keep in mind that **columns** are addressed by their name while **rows** are by their index.

Then if we want to select the column named *Age* in the previous dataframe we must be explicit with the name, usually written inside squared brackets. We must, however, be careful with the selection process: There is a huge difference in using single or double brackets

The selection with a **single bracket** returns

mydf["Age"]

but this is **NOT a data frame**, it is a pandas **series** as can be seen if we do

type(mydf["Age"])

the same result is obtained if we use a full-stop mark

mydf.Age

However if we use **double brackets** we have

mydf[["Age"]]

which can be seen, even from the output, to be a data frame. Finally sometimes we may need only the array of values stripped from all the pandas structure, this can be obtained using

mydf["Age"].values

which return a single numpy array with the values.

If we want to extract two different columns, we should think that the subset is a list, then it must be passed as a list argument

mydf[["Age", "Tenure"]]

When we want to **filter** some values we do it either through their index or based on a condition of the values of some variable (for example when we want to keep only one of the categories in a categorical variable)

In order to filter by index we use the same slicing operator as in numpy, then

mydf[2:5]

returns the rows with indices from the second to the fifth (see that the second value is not included in the interval). All the other properties of this operator are kept too.

The filter under conditions is done using the logical conditions as for example

mydf[mydf["Tenure"] >= 3]

or adding more conditions (careful with the parentheses)

mydf[(mydf["Tenure"] >= 3) & (mydf["Age"] <= 45)]

Finally, we have one option of filtering and selecting in one single step. Then if we want only the rows from the index 8 on and columns *Tenure* and *Exper* we can write

mydf[["Exper", "Tenure"]][8:]

Finally, there are two functions that we can use to select and filter: 

 * **iloc**, which is *integer location based index*, and
 * **loc**, which is just *location* (we are not going to use this function in these notes but you can take a look at the documentation)
 
with iloc you can specify `iloc[row, column]` using the number of row and column as in the following examples

- To select the first column

mydf.iloc[:,0]

- To select the first row

mydf.iloc[0]

- To select the last row

mydf.iloc[-1]

- To select the rows number (not index) 1, 4, 7 and the first and third columns

mydf.iloc[[0,3,6], [0,2]]

#### 3.3.2.- Adding Columns

Adding new variables to a data frame is a rather easy operation (remember that the only requirement is that the number of observations is the same as the number of rows in the data frame).

We can use the single brackets: we write the name of the new column inside of the brackets and assign to it any array.

mydf["gender"] = ["f", "f", "f", "m", "m", "f", "m", "m", "f", "m"]
mydf

usually we may want to add a variable based on the values of another variable. Suppose that we want to add a variable based on the values of *Exper* such that

 - if Exper <= 5, we denote it as "Low"
 - if 5 < Exper <= 10,we denote it as "Mid"
 - if Exper > 10, we denote it as "High"
 
in this case we can use the function `where()` from from the numpy package, which is a sort of if-else condition and just as in those constructions, it can be nested as many times as needed. The structure is `np.where(condition, [x, y])`, such that if the condition is satisfied, it returns *x* and *y* otherwise.

In the following example, since 1 is not greater than 3, it will return "B"

np.where(1 > 3, "A", "B")

Let's use this, together with the single bracket in order to add the new variable (let's denote it as *Code_Exp*)

mydf["Code_Exp"] = np.where(mydf["Exper"] <= 5, "Low",
                            np.where(mydf["Exper"] > 10, "High", "Mid"))
mydf

as mentioned before, we had to nest a second conditional in order to generate the three possible values. This can be generated to any number of conditions.

#### 3.3.3- Sorting Data

In many situations it is important or just interesting to sort the values of the data frame descending or ascending as a function of one of the variables. This can be done with the function `sort_values()` as a method of the data frame itself.

Suppose we want to sort our previous data frame using the variable A

mydf.sort_values(by = ["Age"], ascending = True)

if we want to sort with respect to two different variables we may just add them to the list argument of `by`, for example, if we used `gender` and `Code_Exp` we would do (note that the sorting in the `Code_Exp` is done alphabetically)

mydf.sort_values(by = ["gender", "Code_Exp"], ascending = True)

#### 3.3.4.- Grouping

Pandas is very helpful when we want to find answers from our dataset based on certain conditions of some of the variables. For example, in our data set we may want to find the mean expertise for males and females separatedly. This procedure is knowning as **grouping** and is done with the **`groupby()`** function

mydf.groupby("gender").mean()

In the same sense if we want to keep filter the dataset using this grouping we can use the **`get_group()`** function

mydf.groupby("gender").get_group("f")

this is equivalent to

mydf[mydf.gender == "f"]

## 4.- Input/Output in Pandas

Pandas also allow us to load external data from many different sources: CSV (Comma Separated Values), Excel, HTML, JSON,... into our workspace. All the functions have more or less the same structure `read_type()`. In particular we will mostly use the `read_csv(source, options)` function.

We are going to use this later, once we see how to mount the Google Drive, now let's just explain some of the arguments we can use in the function:

 * `sep`, this is the separator between the columns and, by default it is a comma. However, sometimes we can find it to be a colon or any other character that may be explcitely written here, for example `sep = ";"`
 * `na_values`, makes explicit how the NAs are written in the dataset. Since not all datasets are consistent and sometimes different sources have different encodings, we can find different ways of denoting these values. These must be passed as a list, then we can find `na_values = ["no_info", " ", "."]`

## 5.- Google's Colaboratory

This is the general environment where we will be working on. The main advantage we have here is that we do not have to install anything in our computers and there is no further work on our side to keep everything updated and working. It has some disadvantages but are more technical and do not really concern us.

The Colaboratory environment can be seen as a fork of Jupyter and has the same general structure: they are general text documents with a structure of cells were we can either code or write text

### 5.1.- Code Cells

Here is were we will write our Python codes, then

* Type **Cmd/Ctrl+Enter** to run the cell in place; or
* Press the **Play** button to execute the code.

Note the effect of the `print()` statement.

a = 10
b = 20
print(a)
print(b)

### 5.2.- Text cells

This is a **text cell**. You can **double-click** or **press-enter** to edit this cell. Text cells use markdown syntax. To learn more, see the official [markdown guide](https://colab.research.google.com/notebooks/markdown_guide.ipynb).

You can also add math to text cells using [$\LaTeX$](http://www.latex-project.org/) Just place the statement within a pair of **\$** signs. For example `$\sqrt{3x-1}+(1+x)^2$` becomes the inline equation $\sqrt{3x-1}+(1+x)^2$. If you want to separate this you should create an environment with **\\begin\{equation\}** and **\\end\{equation\}**

\begin{equation}
\sqrt{3x-1}+(1+x)^2
\end{equation}

Some interesting markdown features are:

* Asterisk '*' or dash '-' for bullet points
>- '>-' for indenting a list

1. Numbered list
2. Second item

Include words between double asterisks (or double underscores) for bold, and single asterisks (single underscore) for italic. Example:

- **This is bold**, as well as __this__
- _This is italic_, as well as _this_

Now, a separation line to separate blocks:
***

or

---

To define titles and a Table of Contents:

```
# Main Title
## Subtitle
### Subsubtitle
```

this go up to six levels.

A simple table:

| Column 1 | Column 2 |
|-----------------|-----------------|
|  value 1   | value 2      |

### 5.3.- Connecting Colaboratory to Google Drive

To import and export data, we need to connect colaboratory with our own google drive. This needs to be done every time a session is started.

1. Go to "View" and click on "Table of Contents"
2. Select Files. Navigate one level up, to observe the structure of the machine where your notebook resides. To discover where is your notebook located, type `!pwd` in a code cell. Try `!ls` to get the list of files in your current directory.

3. Now, to connect your google drive under the current directory, copy and execute the following code:

```python
from google.colab import drive
drive.mount('mydrive')
```
You will be ask to enter an authorization code. Once you are done, press `REFRESH` in the Files tab on the left. You should now see a new folder, called `mydrive`, appearing under `content`. You can also use `!ls` to get the updated list. Now **you can access** any file you have in your google drive. This is how we will import files into colaboratory:

- Step 1. Upload the file from your PC to Google Drive
- Step 2. Connect Colaboratory and Google Drive
- Step 3. Navigate to the file inside Google Drive. Right-click on the file, to get the full path. 
- Step 4. Import the file using the appropriate function, typically via:

```python
myData = pd.read_csv("/full path")
```
**Note**: Do not forget to add the first slash "/" in front of the full path.

**Alternative Approach**: You can simply upload the file into the current directory from your local PC. Remember, though, the the next time you connect to this notebook, the file will no longer be there, i.e., you will have to upload it again.

!pwd
!ls

now we can connect to our drive

from google.colab import drive
drive.mount('mydrive')

and then we can read the data set and load it into our workspace

pd.read_csv("/content/sample_data/california_housing_train.csv")