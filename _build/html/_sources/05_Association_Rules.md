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

+++ {"id": "3urCWef6f683"}

# <font color = "0061c7">Association Rules </font>

**Association Rule Mining** is a case of **Unsupervised Learning** where we look for probabilities of joint events. Here we look for frequent patterns, associations or correlations among the items of a set in transaction databases. The idea is to understand customer habits by finding the links between the objects in their “shopping baskets” although this is not the only possible application: cross-marketing, catalog design, fraud detecion or medical treatments are some of the possibilities

In the **Basket Market Data Analysis**, given a dataset of customer transactions we find groups of items which are frequently purchased together. Usually justified with the mythical example of “beers and diapers”. A rule, in this context is an **implication**: A → B, read as “If A then B”, where A is the **antecedent** and B the **consequent** through the computation of different sets of probabilities.

+++ {"id": "iIwnnRLpf69B"}

## <font color = "4587cd"> Apriori algorithm </font>

This algorithm let us simplify the computation of the association rules. The main idea is to use a threshold in different stages by assuming that all subsets of a frequent item set must be frequent and, therefore, that for any infrequent item, all its supersubsets must be infrequent.

Through the algorithm we will compute the different probabilities of all the combinations of different items in all the different transactions. Note that during the algorithm we are not interested on the number of items contained in each
transaction, but only on whether it occurs or not. This implies that at any point of the process we need to turn the dataset into a binary or binary logical set.

+++ {"id": "8l-nQgeBf69G"}

## <font color = "4587cd">Important Metrics</font>

+++ {"id": "ucwQdykVf69J"}

### <font color = "red">Support</font>

The **Support**, T(B), is the probability of a given event, either individual or joint in the set of transactions. The support of the antecedent is known as **coverage**

\begin{equation}
cov(A) = T(A) = P(A)
\end{equation}

The support of a rule is given by the probability of the **intersection**

\begin{equation}
T(A \to B) = P(A \cap B)
\end{equation}

 * It reflects the **popularity** of an item or of a rule
 * We will consider support **thresholds**, implying the minimum proportion to be significant in the set of transactions
 * Sometimes you will see it written as $P(A \cup B)$ which is not formally right

+++ {"id": "gtTi2lt3f69M"}

### <font color = "red">Confidence</font>

The **Confidence**, $C(A \to B)$ is an estimation of the conditional probability

\begin{equation}
C(A \to B) = \frac{T(A \to B)}{T(A)} = \frac{P(A\cap B)}{P(A)} = P(B|A)
\end{equation}

 * We will also consider confidence thresholds as minimum values of likeliness of a rule
 * Sometimes you will see it as $P(A \cup B)/P(A)$, which is formally wrong
 * Since it only considers the popularity of item A, then if both, A and B, are more or less equally likely or the consequent is more likely, this confidence will be very high and does not give a reliable value

+++ {"id": "mj_TkXiOf69Q"}

### <font color = "red">Lift</font>

The **Lift**, $L(A \to B)$ is a measure of the independency of the items in a transaction

\begin{equation}
L(A \to B) = \frac{C(A \to B)}{T(B)} = \frac{P(A \cap B)}{P(A) \cdot P(B)}
\end{equation}

 * It focuses in **less frequent** terms, since the denominator makes the value more sensitive to items with a low coverage.
 * It ranges in $[0,\infty)$, where $\infty$ is probably never obtained
 * If it is 1 it suggest **no association** between the items 
 * If it is greater than 1 it means that item B is likely to be bought if A is bought (**Positive correlation**).
 * If it is smaller than 1 it means that item B is not likely to be bought if A is bought (**Negative correlation**). In short this implies that one item is a substitute of the other.

+++ {"id": "-Slok7Aqf69S"}

### <font color = "red">Leverage</font>

The **Leverage**, $Lev(A \to B)$ is another measure of the independency of the items

\begin{equation}
Lev(A \to B) = T(A \to B) − T(A) \cdot T(B) = P(A \cap B) − P(A) \cdot P(B)
\end{equation}

 * It is the difference between the probability of the rule and the expected probability if the items where independent
 * If **lift** finds strong associations for **less frequent** terms, **leverage** priorizes items with **high support** in the dataset
 * It ranges between [−1, 1], but it can never be greater than the support
 * If it is 0 it suggest **no association** between the items 
 * If it is greater than 0 it means that item B is likely to be bought if A is bought (**Positive correlation**).
 * If it is smaller than 0 it means that item B is not likely to be bought if A is bought (**Negative correlation**).

+++ {"id": "B1e2mslvf69W"}

### <font color = "red">Conviction</font>

The **Conviction**, $L(A \to B)$ is the rate between the expected error rate assuming independency and the observed error rate

\begin{equation}
L(A \to B) = \frac{1 − T(B)}{1 − C(A \to B)} = \frac{P(A) \cdot P(B)}{P(A) − P(A\cap B)}
\end{equation}

 * It is an unbounded quantity: runs in $[0,\infty)$
 * If it is 1, it implies that the measured error rate equals the expected error rate if the items are independent, then we take the items as independent and so, **unrelated**
 * Values $>1$ indicate that the error when we consider the items as independent is greater than the expected error, then the higher the value the greater the strength of the association.

For example, a value of 1.5 implies that the rule $A\to B$ would be incorrect 1.5 times more often if the association between $A$ and $B$ was purely random chance

+++ {"id": "y4HpyXDLf69Y"}

## <font color = "4587cd">Case Study</font>

Let's import the needed libraries

```{code-cell} ipython3
---
executionInfo:
  elapsed: 1634
  status: ok
  timestamp: 1605699521844
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: 4aS-noGmf69a
---
import numpy as np
import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
```

+++ {"id": "3RwMfNrMf69i"}

The **apriori** algorithm as well as the **association rules** are contained in the so-called modules of `mlxtend.frequent_patterns`. Let's see how to find the association rules given a set of transactions.

+++ {"id": "z2bOjzJyaSy3"}

### <font color = "red"> First Approach </font>

In this case we are going to store the dataset in a **multidimensional list** in which each of the arrays corresponds to one of the transactions

```{code-cell} ipython3
:id: QUAwtPeJf69m

transactions = [['Bread','Milk','Butter','Jam','Biscuits'],
                ['Bread', 'Milk', 'Beer', 'Chips'],
                ['Jam', 'Soda', 'Milk'],
                ['Soda', 'Chips', 'Beer'],
                ['Chips', 'Bread', 'Beer', 'Milk'],
                ['Jam', 'Beer', 'Soda', 'Milk']]
```

+++ {"id": "nR9sHHo2f69s"}

now we use the `TransactionEncoder` function. The idea is that we first need to instanciate the object (allocate it in memory) in order to use it later.

```{code-cell} ipython3
:id: UR7uU-1Yf69v

te = TransactionEncoder()
```

+++ {"id": "QGiWqiAif690"}

Now we apply this encoder to our list of transactions by first fitting and then transforming

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 126
executionInfo:
  elapsed: 1516
  status: ok
  timestamp: 1601020339799
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: bu2jwAqXf692
outputId: c0f61694-1e6c-4f28-c710-e2897e31b801
---
te_ary = te.fit(transactions).transform(transactions)
te_ary
```

+++ {"id": "CZHQ5yoVf69-"}

This methods transform the information into a logical set of arrays where *True* implies that the object is in the transaction and *False* otherwise. Now, let's create the dataframe with these encoded transactions

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 237
executionInfo:
  elapsed: 2552
  status: ok
  timestamp: 1601020355660
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: AEiyblGsf69_
outputId: a75d89b9-d7a4-4c3a-9b5e-02088c7219dd
---
dataset = pd.DataFrame(te_ary, columns = te.columns_)
dataset
```

+++ {"id": "_d-B8Igdf6-H"}

It is to this dataframe where we apply the `apriori` algorithm, then

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 206
executionInfo:
  elapsed: 1056
  status: ok
  timestamp: 1601020382248
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: 4FLQxGLcf6-I
outputId: 7de6816f-5c8e-4d21-8466-e152d47c2a06
---
frequent_itemsets = apriori(dataset, min_support = 0.01, use_colnames = True)
frequent_itemsets.head()
```

+++ {"id": "3zN3xQASf6-R"}

which returns the probabilities of all the different intersections of the possible events. Now we find the `association_rules` by imposing a minimum value for one of the metrics, usually **support**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 300
executionInfo:
  elapsed: 1230
  status: ok
  timestamp: 1601020420620
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: TNRn5EFXf6-X
outputId: 1d65019a-d8f5-4814-a185-7519b1933657
---
association_rules(frequent_itemsets, metric = 'support', min_threshold = 0.5)
```

+++ {"id": "E6PZn0d1f6-h"}


Let's interpret one of the rules in this table, for example the one for 

\begin{equation}
\text{Beer} \to \text{Chips}
\end{equation}

has as values

<br>

| Antecedent | Consequent | Support | Confidence | Lift | Leverage | Conviction |
|:----------:|:----------:|:-------:|:----------:|:----:|:--------:|:----------:|
| 0.67 | 0.50 | 0.50 | 0.75 | 1.5 | 0.17 | 2.00 | 

<br>

 * **Support**: The rule appears in 50% of the transactions
 * **Confidence**: The probability that in any transaction someone buys CHIPS if they have already bought BEERS is the 75%
 * **Lift**: Since it is 1.5, i.e. positive, the goods are not independent and it is likely that when customers buy one, they buy the other
 * **Leverage**: Since leverage ranges from -1 to 1, a value of 0.17 indicates a weak positive association between the items
 * **Conviction**: A value of 2.00 means that the rule would return the double of wrong predictions if the association rule was purely due to random chance

+++ {"id": "8F6hbOk4akY5"}

### <font color = "red"> Second Approach </font>

Usually, the transactions do not come as a list of different transactions but in a slightly different form, consider the following dataframe for the same set of transactions as in the previous analysis

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 206
executionInfo:
  elapsed: 977
  status: ok
  timestamp: 1605700667298
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: Am1Dychrf6-j
outputId: 33f067ec-abbf-47a3-b74f-cd53533d7ab5
---
trans_df = pd.DataFrame({'InvoiceNo': [1,1,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,5,5,6,6,6,6],
                         'Description': ['Bread','Milk','Butter','Jam','Biscuits',
                                         'Bread', 'Milk', 'Beer', 'Chips',
                                         'Jam', 'Soda', 'Milk',
                                         'Soda', 'Chips', 'Beer',
                                         'Chips', 'Bread', 'Beer', 'Milk',
                                         'Jam', 'Beer', 'Soda', 'Milk'],
                         'Quantity': [2,3,1,2,3,1,1,5,4,1,5,2,1,2,3,1,3,4,4,2,2,1,1]})
trans_df.head()
```

+++ {"id": "hCdL0fmTf6-p"}

the following code let us take if and turn into a dataframe similar to the previous `dataset`

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 269
executionInfo:
  elapsed: 901
  status: ok
  timestamp: 1605700974218
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: bJJp4560f6-s
outputId: a5619452-8b6c-48ed-b564-cee91212db42
---
basket = (trans_df
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo')
          )
basket
```

+++ {"id": "cH1CslK_f6-z"}

now we need to turn it into a **1/0-matrix**, since we are not actually interested in the number of units bought in the transactions, then 

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 269
executionInfo:
  elapsed: 758
  status: ok
  timestamp: 1605700390215
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: 8WvMhEnaf6-1
outputId: 921e229d-0713-4c4d-adf3-271807b56bf2
---
basket_sets = basket.apply(lambda x: np.where(x >= 1, 1, 0))
basket_sets
```

+++ {"id": "WAQTHhItf6-5"}

Now we can use the **apriori** algorithm as before

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 206
executionInfo:
  elapsed: 1237
  status: ok
  timestamp: 1601020711627
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: 6x-8Dv7Af6-7
outputId: eb97378c-8894-420d-ef4b-5c2f242d42a5
---
frequent_itemsets2 = apriori(basket_sets, min_support = 0.07, use_colnames = True)
frequent_itemsets2.head()
```

+++ {"id": "0Il7VJYGf6_D"}

From where we can find the **association rules** as before

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 300
executionInfo:
  elapsed: 1689
  status: ok
  timestamp: 1601020716448
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: SqGgoFnYf6_F
outputId: 08bdf8e9-1273-43ec-992b-d89cf0e8170f
---
association_rules(frequent_itemsets2, metric = "support", min_threshold = 0.5)
```

+++ {"id": "RjpR2hduf6_J"}

## <font color = "4587cd">Analysis of a real dataset</font>

Let's download from the web a dataset by importing it directly using `pandas` dataframe

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 293
executionInfo:
  elapsed: 40224
  status: ok
  timestamp: 1605701077587
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: GlJs1J8Wf6_N
outputId: 24650fbd-ce27-41bb-d507-6d31a3dbc72f
---
transactions = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
transactions.head()
```

+++ {"id": "pN-PvyTpf6_T"}

And now just to be sure that we always have a copy of the dataset, let's copy it

```{code-cell} ipython3
:id: iY1mcpATf6_V

df_copy = transactions.copy(deep = True)
```

+++ {"id": "mP4AvBBzf6_b"}

Now, this dataset needs to be cleaned (just a bit). The operations we will make are:

 * Remove Initial and Ending spaces in the **Description**
 * Remove the Transactions without invoice number
 * Convert the Invoice Number into a **String**
 * Remove the transactions made by credit card (marked with a "C")

```{code-cell} ipython3
:id: 4KY0I4nAf6_c

transactions['Description'] = transactions['Description'].str.strip()
transactions.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
transactions['InvoiceNo'] = transactions['InvoiceNo'].astype('str')
transactions = transactions[~transactions['InvoiceNo'].str.contains('C')]
```

+++ {"id": "WbHPjOfpf6_g"}

### <font color = "red">Analysis of French basket</font>

Since the dataset is too big, let's focus on the analysis by countries. In particular, let's study what happens in France. Notethat since this data set is given as in the second case in the previous section, we will just use the same steps. Note that the names we are using are the same as before, we do this just because they should belong to different analysis.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 391
executionInfo:
  elapsed: 1075
  status: ok
  timestamp: 1605702924684
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: -NQZcaYuf6_i
outputId: 276a8ac2-92f8-4e89-bad8-5e2fd0983c0b
---
basket = (transactions[transactions['Country'] == "France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets = basket.apply(lambda x: np.where(x >= 1, 1, 0))
basket_sets.drop('POSTAGE', inplace = True, axis = 1)
basket_sets.head()
```

+++ {"id": "piiejOhPf6_q"}

Now we go for the `apriori` algorithm and the association rules

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
executionInfo:
  elapsed: 745
  status: ok
  timestamp: 1605703812518
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: JjzJwvktZ6ch
outputId: 9bba18f5-6a65-4b0b-b611-d693d6cdca4a
---
frequent_itemsets = apriori(basket_sets, min_support = 0.07, use_colnames = True)
frequent_itemsets
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 797
executionInfo:
  elapsed: 724
  status: ok
  timestamp: 1605703668067
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: D8EmjRjAf6_s
outputId: 1bd28f95-5cb5-491d-b524-35bd14474399
---
frequent_itemsets = apriori(basket_sets, min_support = 0.07, use_colnames = True)
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 0)
rules.sort_values(by = ['lift'], axis = 0, ascending = True).head(10)
```

+++ {"id": "TwLb_SU2f6_v"}

See that we have reduced very significantly the **support threshold**. This makes perfect sense because in this case the transactions are less common than in our previous case. This is not really a problem and will only give a significant impact in the values of **lift** and **levarage**.

We might be interested in a certain combination of **association rules**, for example

```{code-cell} ipython3
:id: VnQGwWrVf6_x
:outputId: d463bc11-bddd-45ca-f0d8-a751a9179d57

rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) & 
       (rules['support'] >= 0.1)]
```

+++ {"id": "sAa00Unzf6_1"}

Let's interpret one of the rules in this table, for example the one for RED SPOTTY PAPER PLATES $\to$ RED RETROSPOT PAPER NAPKINS, then

| Antecedent | Consequent | Support | Confidence | Lift | Leverage | Conviction |
|:----------:|:----------:|:-------:|:----------:|:----:|:--------:|:----------:|
| 0.128 | 0.133 | 0.102 | 0.800 | 6.031 | 0.085 | 4.34 | 

 * **Support**: The rule appears in 10.2% of the transactions
 * **Confidence**: The probability that in any transaction someone buys RED RETROSPOT PAPER NAPKINS if they have already bought RED SPOTTY PAPER PLATES is the 80%
 * **Lift**: Since it is 6.03, i.e. positive, the goods are not independent and it is likely that when customers buy one, they buy the other
 * **Leverage**: Since leverage ranges from -1 to 1, a value of 0.09 indicates a weak positive association between the items
 * **Conviction**: A value of 4.34 means that the rule would return 4.34 times more wrong predictions if the association rule was purely due to random chance
 
Now, what is the problem with Lift and Leverage? That since the indiviual events are not very common (12.8% and 13.3%) the Lift will detect stronger associations, while Leverage would be important if the Support is high (10.2%). Then in this case it would be more interesting looking at the Lift.

+++ {"id": "8n4OYkYLf6_2"}

### <font color = "red">Analysis of Spanish basket</font>

Let's repeat the same procedure with Spain (we may choose another
countries, but UK, for example has a big dataset that may lead us into 
memory problems)

```{code-cell} ipython3
---
executionInfo:
  elapsed: 1899
  status: ok
  timestamp: 1605702393842
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: CWK8lqTLf6_3
---
basket2 = (transactions[transactions['Country'] == "Spain"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets2 = basket2.apply(lambda x: np.where(x >= 1, 1, 0))
basket_sets2.drop('POSTAGE', inplace = True, axis = 1)
```

+++ {"id": "FqzNnIL6f6_-"}

And now the rules

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 199
executionInfo:
  elapsed: 1196
  status: ok
  timestamp: 1605702415298
  user:
    displayName: Juan Jose Manjarin
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14GgROm1G9L6BfG7PCIlE0tJxJJ2QITgE4QN52iI2=s64
    userId: 04910883006985787828
  user_tz: -60
id: TwDp7OpQf7AA
outputId: b97e8a80-3c33-48dc-b3d5-8c338759d3cc
---
frequent_itemsets2 = apriori(basket_sets2, min_support = 0.07, use_colnames = True)
rules2 = association_rules(frequent_itemsets2, metric = "lift", min_threshold = 1)
rules2
```

+++ {"id": "yT9pUW4tf7AE"}

Then a possible set of rules we may choose is

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 261
executionInfo:
  elapsed: 922
  status: ok
  timestamp: 1571909327176
  user:
    displayName: Juan Jose Manjarin
    photoUrl: ''
    userId: 04910883006985787828
  user_tz: -120
id: UMuQonHZf7AF
outputId: 4ca850d0-132c-4d02-f457-52f539403283
---
rules2[ (rules2['lift'] >= 3) &
        (rules2['confidence'] >= 0.5) ]
```

+++ {"id": "Ik1vbR2wf7AK"}

**Can you find the same rule as in the previous case? Whichever the answer, interpret the first of the rules in the table above**
