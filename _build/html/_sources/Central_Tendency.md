# Central Tendency Measures

The central tendency measures are values inside the distributions of data around which the other values tend to group. They are usually known as averages, even though that is a clear abuse of language as we will see below. Even more they are *central* only if the distribution satisfies certain properties of symmetry that not all the realistic cases do, so we have to be careful with their use and meaning. We will see only a few of them, the more basic and the ones we will use all along the parametric inference theory. 

## Mean

There are many different quantities associated to the name ``mean'', each applied to a different situation, or much better, they are used depending on what is the relation between the numbers involved. We can find the harmonic, arithmetic, geometric, arithmetic-geometric or root-square means, such that for any set of numbers, the means satisfy the following relation

\begin{equation}
H\leq G\leq AGM\leq A\leq RS 
\end{equation}

In this course we will use extensively the arithmetic mean, but due to their inherent interest, in this chapter, we will also see both the geometric and the harmonic means.

### Arithmetic Mean

This is what we usually know as average and what we will always denote, simply, as mean. If we want to compute the mean of a set of values $x_i$ we wimply add them all and divide by the number of values

\begin{equation}
\bar x=\frac{1}{r}\sum_{i=1}^r x_i
\end{equation}

\noindent in such a way that if $r=n$ we deal with a sample mean and if $r=N$ we work with the population mean. For notational purposes and in order to know which one we are computing, we will always denote the population mean as $\mu$.

Be aware of a very important property of the average: the result will probably not be one numbers we begun with. If we compute the average age in a group of people it is highly probable that none of the subjects is really that age old, and it is not as I heard once on the news at TV that ``half the people have that age''. In cases in which the population is symmetrically distributed, the mean is simply the most common value or the most common would-be value.\\\vspace{0.5cm}

{\small{
    {\bfseries\scshape{Example.}} Suppose we have the following random sample of 10 final exam grades ranging from $0$ to $100$

    \begin{center}
      \begin{tabular}{cccccccccc}
        \toprule[0.1em]
        88 & 51 & 63 & 85 & 79 & 65 & 70 & 73 & 79 & 77\\
        \bottomrule[0.1em]
      \end{tabular}
    \end{center}

    Then we can compute the arithmetic mean as

\begin{equation}
\bar x=\frac{88+\dots +77}{10}=73
\end{equation}

}}\vspace{1cm}

Using the language of frequencies we defined in the previous chapter, we can reformulate our expressions for this mean

\begin{equation}
\bar x=\frac{1}{n}\sum_{i=1}^n n_ix_i=\sum_{i=1}^n\omega_ix_i
\end{equation}

\noindent where we make use of the total frequencies in the first case and of the relative frequency in the second.\\\vspace{0.5cm}

{\small{
    {\bfseries\scshape{Example.}} Let's use the values of the previous example about the scores and let's compute the mean using both: the total and the relative frequencies:

    \begin{equation}
    \bar x=\frac{1}{10}\left( 3\cdot 2+4.5\cdot 1+5\cdot 4+7.5\cdot 2+10\cdot 1\right)=\frac{55.5}{10}=5.55
    \end{equation}

now using the relative frequencies:

    \begin{equation}
    \bar x=\left( 3\cdot 0.2+4.5\cdot 0.1+5\cdot 0.4+7.5\cdot 0.2+10\cdot 0.1\right)=5.55
    \end{equation}

so we obtain the same answer, as should be.
}}\vspace{1cm}

The formula in which we use the relative frequencies can be given another interpretation as the {\sl{weighted arithmetic mean}}, i.e. the mean that can be computed when the values do not contribute with the same weight to the final result. Let's see it\\\vspace{0.5cm}

{\small{
    {\bfseries\scshape{Example.}} Suppose that the evaluation system in a course gives different weights to each of the possible marks to obtain. In the following table we find that system and the marks obtained by one student

 \begin{center}
    \begin{tabular}{clcc}
     \toprule[0.1em]
     A. & Class participation and discussion; exercises in class & $10\%$ & 5  \\
     B. & Group Report and presentation                          & $30\%$ & 8  \\
     C. & Quizzes                                                & $5\%$  & 9  \\
     D. & Mid-term exam                                          & $25\%$ & 7.5\\
     E. & Final exam                                             & $30\%$ & 8  \\\bottomrule[0.1em]
    \end{tabular}
\end{center}

 The first step is take the weights and convert them into relative frequencies:

\begin{equation}
10\%\to 0.10,\quad 30\%\to 0.30,\quad 5\%\to 0.05,\quad 25\%\to 0.25
\end{equation}

\noindent then the mean mark of this student can be computed as

\begin{equation}
\begin{array}{rcl}
\overline x & = & \displaystyle\sum_{i=1}^{5}x_i\omega_i\\[3ex]
                  & = & 5\cdot 0.1+8\cdot 0.3+9\cdot 0.05+7.5\cdot
                  0.25 + 8\cdot 0.3 \\[1.5ex]
                   & \simeq & 7.63
\end{array}
\end{equation}
}}


### Harmonic Mean

When the numbers involved are relatetd to a fixed unit not to an arithmetic operation, we should not use the arithmetic mean but a slight modification known as harmonic mean. It is defined as the arithmetic mean of the reciprocals of the numbers:

\begin{equation}
\bar x_H=\dfrac{r}{\displaystyle\sum_{i=1}^r\dfrac{1}{x_i}}
\end{equation}

\noindent where $r$ can run, as before, over the sample or the population sizes.\\\vspace{0.5cm}

{\small{
    {\bfseries\scshape{Example.}} An investor purchases \EUR{2,000} worth stocks each month. If the values per stock are \EUR{8}, \EUR{8.5} and \EUR{9} those months, compute the mean price he paid for the stocks.

    \begin{equation}
    \bar x_H=\frac{3}{\dfrac{1}{8}+\dfrac{1}{8.5}+\dfrac{1}{9}}=8.48
    \end{equation}
}}\vspace{0.5cm}

In this problem the fixed unit are the \EUR{2,000} she uses which make that she does not buy the same amount of stocks each month, in which case we should use the arithmetic mean. In that case the problem should read somehing like ``an investor purchases $2,000$ stocks per month at a price of \EUR{8}, \EUR{8.5} and \EUR{9}, what is the mean price she pays?'' Whose answer is $8,5$


### Geometric Mean

There is another type of relation that renders the arithmetic mean useless, namely, when the numbers are related geometrically, i.e. by a multiplicative relation. This geometric sequences of numbers appear everytime in science and in economy, from disitegration rates of atoms to birth rates in a country or interest rates in finance. Let's see how they work.

The main idea is that given a value, for example, a deposit in a bank account, the amount after one step depends on that amount and a fraction of it. If $a_0$ is the deposit, then after one step we will have

\begin{equation}
a_1=a_0+ra_0=(1+r)a_0
\end{equation}

\noindent where $r$ is known as the {\emph{growth rate}} (even though it can be negative in some cases). After n-steps we will have

\begin{equation}
a_n=(1+r)a_{n-1}=(1+r)^na_0
\end{equation}

The geometric mean is used to know the mean growth after this n-steps procedure. For example, we would like to know what has been the mean growth of our bank deposit in $5$ years.

The geometric mean of a set, $x_i$ for $i\in\NN$, of numbers is defined as the $n$th-root of the product of those numbers

\begin{equation}
\bar x_g=\sqrt[n]{x_0\cdot x_1\dots x_n}
\end{equation}

\noindent and the mean growth rate of the sequence is

\begin{equation}
\bar r=\bar x_g-1
\end{equation}

{\small{
    {\bfseries\scshape{Example.}} The sales in a shop have grown a $25\%$ in $5$ years, what has been the mean annual growth?

    This problem states that

    \begin{equation}
    a_5=1.25a_0
    \end{equation}

    \noindent whichever the intial sales were (which we do not really need to know). Then $1.25=(1+r)^5$, from where it is easy to see that the annual growth has been $4.6\%$\\\vspace{0.5cm}
}}

Since the geometric mean involves the computation of a root, it is sometimes argued that there is no geometric mean when we deal with negative numbers. But that is plainly wrong. This sort of mean extensively used in some macro- or microeconomic problems, as the mean evolution of the GDP along some years in a country or region. Many times this sort of problems involve negative values of the growth, so a decreasing of the GDP in a given year. Would that mean that we cannot compute a mean growth, even if it is negative, in a period of time? obviously not, but it involves a slight subtlety. Let's work it out through a real example\\\vspace{0.3cm}

{\small{
    {\bfseries\scshape{Example.}} Let's compute the average growth of the GDP In Madrid for a period of seven years, from $2008$ to $2015$, the data from the INE webpage are the following

    \begin{table}[ht]
      \centering
      \caption{INE data for the GDP growth of Madrid from 2008 to 2015}\label{tab:INEMad}
      \vspace{0.3cm}
      \begin{tabular}{lccccccc}
        \toprule[0.1em]
        & 08/09 & 09/10 & 10/11 & 11/12 & 12/13 & 13/14 & 14/15 \\\midrule
        Madrid & -2.4$\%$ & -0.3$\%$ & 0.6$\%$ & -0.9$\%$ & -1.9$\%$ & 1.6$\%$ & 3.4$\%$ \\
        \bottomrule[0.1em]
        \end{tabular}
    \end{table}

    In this case since there is an even number of negative growths, we could simply follow the procedure previously defined. However, let's act as if that were not the case. The idea is that we first transform all the percentages in relative frequencies and the add $1$ to all of them. With these modified data we compute the mean and the substract that $1$ at the end. The new data will be:
    
\begin{center}
  \begin{tabular}{ccc}
    \toprule[0.1em]
    Data & Relative Freqs. & Modified Freqs. \\\midrule
    -2.4  & -0.024 & 0.976 \\
    -0.3 & -0.003 & 0.997 \\
    0.6  & 0.006 & 1.006 \\
    -0.9 & -0.009 & 0.991 \\
    -1.9 & -0.019 & 0.981\\
    1.6 & 0.016 & 1.016 \\
    3.4 & 0.034 & 1.034\\
    \bottomrule[0.1em]
    \end{tabular}
\end{center}

Now, the geometric growth will be

\begin{equation}
\bar x_g=\sqrt[7]{0.976\cdot\dots\cdot 1.034}-1=0,99996-1=-0.00003
\end{equation}

\noindent i.e. a mean decreasing growth of a $-0.003\%$ per year    
}}

## Median

The {\emph{Median}} is the midvalue of the distribution. Then again, it may or may not be an actual value of our set of numbers. However, in this case it depends on the amount of data we have and if it is even or odd. To compute the median, first we rearrange the values from low to high and then compute the location of the mid point then the value.

Suppose that our set has an odd number of elements: $x_n$ with $n=2k+1$, then the mid point is located simply at the position $k+1$ and the median is $x_{k+1}$.

If our set has an even number of elements: $x_n$ with $n=2k$, then the mid point is half-way from $k$ to $k+1$ positions and the median will be $(x_k+x_{k+1})/2$. There are different ways to write this value:

\begin{equation}
{\text{median}}_{{\text{even}}}=x_{k}+\frac{x_{k+1}-x_{k}}{2}=\frac{x_k+x_{k+1}}{2}
 \end{equation}

 \noindent the first one makes evident the idea that will be useful below for percentiles: we add to the lowest value the required percentage of the interval.\\\vspace{0.3cm}

 {\small{
     {\bfseries\scshape{Example.}} Let's compute the median of the set of the exampe above:

     \begin{center}
      \begin{tabular}{cccccccccc}
       \toprule[0.1em]
        88 & 51 & 63 & 85 & 79 & 65 & 70 & 73 & 79 & 77\\
       \bottomrule[0.1em]
      \end{tabular}
     \end{center}

     Since $n=10$, then $k=5$ and the median will be the mid value of the interval between the 5th and 6th positions. Let's rearrange the numbers:

     \begin{center}
      \begin{tabular}{cccccccccc}
       \toprule[0.1em]
        51 & 63 & 65 & 70 & 73 & 77 & 79 & 79 & 85 & 88\\
       \bottomrule[0.1em]
      \end{tabular}
     \end{center}

     \noindent then we see that $x_5=73$ and $x_6=77$ and the median will be

     \begin{equation}
     {\text{median}}=\frac{77+73}{2}=75
     \end{equation}\vspace{0.5cm}


    {\bfseries\scshape{Example.}} Let's compute the median of the set:

     \begin{center}
      \begin{tabular}{ccccccc}
       \toprule[0.1em]
        11 & 11 & 14 & 12 & 17 & 21 & 13 \\
       \bottomrule[0.1em]
      \end{tabular}
     \end{center}

     Since $n=7$, then $2k+1=7$ and $k=3$, then median will be the vale $x_4$. Let's rearrange the numbers:

     \begin{center}
      \begin{tabular}{ccccccc}
       \toprule[0.1em]
        11 & 11 & 12 & 13 & 14 & 17 & 21 \\
       \bottomrule[0.1em]
      \end{tabular}
     \end{center}

     \noindent then we see that the median is $x_4=13$.
}}
 
## Mode

The {\emph{Mode}} is the most repeated value in a set of numbers, i.e. the value with the highest total frequency. From this definition it should be clear that contrary to the mean and the median which, meaningful or not, always exist, there is not always a mode: as soon as all the numbers have a total frequency of $1$, there is no ``most repeated value''.

On the other hand some sets may have more than one. That is why we denote the distribution as unimodal, bimodal, trimodal,...In the examples for the median we can easily see that in the first case there is no mode, since all the numbers appear just once, and the second is unimodal with mode $11$.


## Which CTM should we use?

We have just seen the three main central tendency measures in any distribution, but we should not use indistinctly one or the other since sometimes the use of one or another is missleading or simply does not show up the most common features of a distribution. In table~\ref{tab:MeMeMo} we can see a comparison between the main properties of mean, median and mode in order to see when one is more relevant than the others.

\begin{table}
  \centering
  \caption{Comparative table of the Central Tendency Measures}\label{tab:MeMeMo}
  \vspace{0.3cm}
  \begin{tabulary}{1\textwidth}{lLL}
    \toprule[0.1em]
    & \multicolumn{1}{c}{Pros} & \multicolumn{1}{c}{Cons} \\\midrule[0.1em]
\multirow{5}{*}{{\sc{Mean}}}    & & \\
                                     & {\bf{1.-}} Takes into account all the information & {\bf{1.-}} Only appliable to numerical variables\\
                                     & {\bf{2.-}} Has the best theoretical properties for inference   & {\bf{2.-}} An alteration of the data moves the mean towards it\\
                                     & & {\bf{3.-}} It is very sensitive to the outliers in skewed\\
                                     & & \\
\midrule
\multirow{4}{*}{\sc{Median}} & & \\
                                   & {\bf{1.-}} Good for income measurements & {\bf{1.-}} Not sensitive to the bulk of points\\
                                   & {\bf{2.-}} Not sensitive to outliers & \\
                                   & & \\
\midrule
\multirow{4}{*}{\sc{Mode}}   & & \\
                                   & {\bf{1.-}} Good when only matters the most frequent observation  & {\bf{1.-}} Insensitive to all but the most frequent value\\
                                   & {\bf{2.-}} Can be used for categorical variables & \\
                                   & & \\
\bottomrule[0.1em]
  \end{tabulary}
\end{table}

Later we will see how to define properly some of the notions we are going to use now, but in the discussion of this section an intuitive idea is more than enough. A distribution of numbers is symmetric if there are, more or less the same amount of values to both sides of its midpoint. If that is the case, mean, median and mode will be the same value and then the use of the mean is preferable due to some properties as estimator that we will see in chapter~\ref{InfThe}.

However, if the distribution becomes asymmetric, maybe by the inclusion of new data lying to one side of the previous mean, then this is not an interesting value anymore, since it stops representing a ``central'' value. In that case is more prefereable the use of the median. If, on the other hand there is some new data that modifes the bulk of the numbers but with values on both sides of the median, this will not be modified, but the mean will, then in this case we prefer the use of the mean.

If there are outliers, i.e. data whose value is very different from the bulk of the data, the mean is again a rather useless measure since it will be displaced towards that value and we should use the median or the mode. And between these two, the mode can be more interesting than the median if the distribution is very asymmetric.\\\vspace{0.5cm}


{\small{
    {\bfseries\scshape{Example}}. Let's find the central tendency measures of the demand of coffees at a university cafeteria at lunch time if random sample of $n=12$ intervals of $5$ minutes is
    
\begin{center}
\begin{tabular}{cccccccccccc}
\toprule[0.1em]
60 & 84 & 65 & 67 & 75 & 72 & 80 & 85 & 63 & 82 & 70 & 75 \\
\bottomrule[0.1em]
\end{tabular}
\end{center}

It is not difficult to compute that the arithmetic mean is $73.17$, the median is $73.5$ and the mode is $75$. In this case they do not differ too much and then probably our safest choice is the mean.

I we plot the values in a bar chart it will, probably say nothing. The best way to visualize these sort of data is through the plot of the frequency distribution itself, a plot known as histogram which we will study in a while.
}}
\vspace{1cm}

The values in the example correspond to a rahter symmetric distribution, but what happens in other circumstances? For example, the wages distribution in a company in which the bulk of the people earns way less than the heads of the departments who earn less than the CEO? In that case the distribution should be probabily described through the mode. Let's see a real life example.

\begin{figure}
  \centering
  \includegraphics[width=10cm]{Images/Inc2013.eps}
  \caption{Income distribution in Spain in 2013, {\sc{Source}}: INE, http://www.ine.es}\label{fig:IncD2013}
\end{figure}

In Fig~\ref{fig:IncD2013} we see the income distribution in Spain in year 2013. In the horizontal axis we see the wage in terms of the minimum wage and in the vertical axis we find the percentage of the population with that wage. We inmediately see that it is not a symmetric distribution since there is a long tail to the right of the highest peak, becoming the first example we see of what is known as a right-skewed distribution. In this case, it should be clear that the mean is not the relevant central tendency measure even though all the news we read on newspapers or watch on TV about the wages deal with the ``mean salary''

\begin{figure}
  \centering
  \includegraphics[width=10cm]{Images/Inc2013Line.eps}
  \caption{Central Tendency Measures for the income distribution}\label{fig:IncD2013L}
\end{figure}

In Fig~\ref{fig:IncD2013L} we have the same distribution but with a line chart to allow us to see the locations of all the central tendency measures. As we mentioned before, the mean is displaced towards the outliers of the distribution and the median, being a very asymmetric distribution, is also displaced from the peak. In this case the most relevant number is the mode: the most common salary among the Spanish population, shared by the $32.96\%$ of the population.

Considering that the minimum wage on $2013$ was \EUR{9,034.20} per year, it can be seen that the approximated values for the central tendency measures are:

\begin{equation}
{\text{Mean}}=22,466.35,\qquad{\text{Median}}=19,370.88,\qquad{\text{Mode}}=11.374.38
\end{equation}

\noindent all in Euros. We will see how to compute these values later in section~\ref{HisOji} but we must consider them as good approximations the values found at INE for the exact mean and median are \EUR{22,697.86} and \EUR{19,029.66} respectively.

This shape of the income distribution was first found by W. Pareto at the beginning of 20th century. He gathered data from different ages: from 1454 to his contemporary rental incomes, and simply plotted it. His foundings were really striking and his conclusions even more controversial, to the point that, for example, K. Popper called him the ``theoretician of totalitarism'', not without certain reason, since by his death Italian fascists were really fond of him. Leaving his interpretations for a different forum, the truth is that his distribution has been found once and again all over any income distribution of any country in any historic moment.

The Pareto distribution is an important example of power law that is exremely non-symmetric, but it is not the only one: the Zipf distribution for the appearances of words in a text, the distribution of errors in Hard Drives, the network TCP connections and many other examples around us, important for many different business objectives follow this type of skewed distributions, as for example, the Riemann Zeta distribution related to the Riemann Hypothesis, one the Millenium Prize Problems yet to solve (US $\$$1M is still awaiting for someone to solve it), which is the basis for all the internet security.

Of course there are many other cases of symmetric distributions from pregnancy time to customers in shops or trains arriving to stations, and in all of these cases the mean is the relevant central tendency measure.


## Quartiles, Deciles and Percentiles

In any distribution we can still locate some other interesent points: the deciles, quartiles and percentiles, which will become extremely relevant when we want to compute certain intervals and probabilities later in the course. Independently of the amount of numbers we may have in our distribution we can formally split the whole range in an arbitray number of parts and see what portion of the values are {\emph{under}} a particular value.

As should become our habit, he first step is the ordering of the raw data from low to high. Only then we can proceed with our further splitting. Now, if we split the distribution in forths, we will find the {\emph{quartiles}} as those values where we can find the $25\%$, $50\%$, $75\%$ or $100\%$ of the distribution. The idea is first locate the point, then if we suppose we have $n$ values, then the location of the $k$th quartile is

\begin{equation}
Q_k^{\text{pos}}=\frac{k}{4}(n+1)
\end{equation}

\noindent for $k\in\left\{1,2,3,4\right\}$.

If we split in tenths, the we find the {\emph{Deciles}} as the values where we can find  an integer multiple of the ten percent of the distribution. In this case the location will be

\begin{equation}
D_k^{\text{pos}}=\frac{k}{10}(n+1)
\end{equation}

\noindent for $k\in\mathbb{N}$, $k\subset\left[ 1,10\right]$.

Finally if we split in one hundred parts, we find the {\emph{Percentiles}} which simply give us values up to which we find a percentage of the distribution. We should keep in mind that whenever we compute, later in the course, the value in a distribution that corresponds to a probability we will be computing these percentiles. Now the location is found as

\begin{equation}
P_k^{\text{pos}}=\frac{k}{100}(n+1)
\end{equation}

\noindent for $k\in\mathbb{N}$, $k\subset\left[ 1,100\right]$.

Once we have found the locations we proceed to copmute the value itself, which involving one subtlety is much better if we show how to compute them through examples.\\\vspace{0.5cm}

{\small{
    {\bfseries\scshape{Example.}} Let's compute $P_{65}$, $D_4$ and $Q_3$ in the following set of numbers:

    \begin{center}
    \begin{tabular}{ccccccccccccc}
      \toprule[0.1em]
      15 & 16 & 15 & 18 & 21 & 25 & 19 & 20 & 22 & 23 & 20 & 16 & 17 \\
      \bottomrule[0.1em]
    \end{tabular}
    \end{center}

    First of all, let's reorder the set from low to high:

    \begin{center}
    \begin{tabular}{ccccccccccccc}
      \toprule[0.1em]
      15 & 15 & 16 & 16 & 17 & 18 & 19 & 20 & 20 & 21 & 22 & 23 & 25 \\
      \bottomrule[0.1em]
    \end{tabular}
    \end{center}

    Now let's compute each of the values. In this case we have $13$ values, so se will use $n=13$ in all our three cases. Then, for $P_{65}$:

    \begin{equation}
    P_{65}^{\text{pos}}=\frac{65}{100}(13+1)=9.10
    \end{equation}

    \noindent how do we interpret this value? By saying that the percentil $65$ will be at poistion $9$ plus a $10\%$ of the distance between positions $9$ and $10$. Then since the $9$th number in our set is $20$ 

    \begin{equation}
    P_{65}=20+\frac{10}{100}(21-20)=20.1
    \end{equation}

    Let's compute $D_4$:

    \begin{equation}
    D_4^{\text{pos}}=\frac{4}{10}(13+1)=5.6
    \end{equation}

    \noindent from where, using that the $4$th value is $16$ we find

    \begin{equation}
    D_4=16+\frac{60}{100}(17-16)=16.6
    \end{equation}

    Finally, let's compute the third quartile:

    \begin{equation}
    Q_3^{\text{pos}}=\frac{3}{4}(13+1)=10.5
    \end{equation}

    \noindent then, since the $10$th value is $21$ we find

    \begin{equation}
    Q_3=21+\frac{50}{100}(22-21)=21.5
    \end{equation}

}}\vspace{1cm}

Now, it should be clear that since all these numbers represent values inside our distribution they are going to be related among themselves, then for example:

\begin{equation}
P_{50}=D_{5}=Q_{2}={\text{Median}}
\end{equation}

\noindent and so with all the other combinations. Then if for any reason one finds herself in trouble with, say, computing $Q_1$, compute $P_{25}$ instead.


## Five-Number Summary

The quartiles just defined can be used to give a summary of the distribution known as the {\emph{Five-Number Summary}}, which is simply the chain:

\begin{equation}
{\text{minimum}} < Q_1 < \text{median} < Q_3 < \text{maximum}
\end{equation}

\noindent where the minimum and the maximum represent both extreme values in the set.\\\vspace{0.5cm}

{\small{
    {\bfseries\scshape{Example.}} Let's write down the Five-Number Summary on the example above of the demand of coffees at a university cafeteria at lunch time whose random sample of $n=12$ intervals of $5$ minutes is

    \begin{center}
      \begin{tabular}{cccccccccccc}
        \toprule[0.1em]
        60 & 84 & 65 & 67 & 75 & 72 & 80 & 85 & 63 & 82 & 70 & 75 \\
        \bottomrule[0.1em]
      \end{tabular}
    \end{center}

    Again we reorder the distribution from low to high as

     \begin{center}
      \begin{tabular}{cccccccccccc}
        \toprule[0.1em]
        60 & 63 & 65 & 67 & 70 & 72 & 75 & 75 & 80 & 82 & 84 & 85 \\
        \bottomrule[0.1em]
      \end{tabular}
    \end{center}

     \noindent from where we see that $x_{min}=60$ and $x_{max}=85$, and we know from before that the median is $75.5$. Then we only have to compute $Q_1$ and $Q_3$:

    \begin{equation}
    Q_1^{\text{pos}}=\frac{1}{4}(12+1)=3.25,\qquad Q_3^{\text{pos}}=\frac{3}{4}(12+1)=9.75
    \end{equation}

\noindent and then 

    \begin{equation}
    Q_1=65+\frac{1}{4}(67-65)=65.5,\qquad Q_3=80+\frac{3}{4}(82-80)=81.5
    \end{equation}

    \noindent then the Five-Numer Summary reads:

    \begin{equation}
    \begin{array}{ccccccccc}
      x_{\text{min}} & < & Q_1 & < & \text{median} & < & Q_3 & < & x_{\text{max}} \\[2ex]
      60 & < & 65.5 & < & 73.5 & < & 81.5 & < & 85
    \end{array}
    \end{equation}
}}
