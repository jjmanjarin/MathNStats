# Power of the Test

This case is when we fail to reject $H_0$ but the true hypothesis is $H_1$,
i.e.

\begin{equation}
\beta= P(X\in C^*|H_1 )
\end{equation}

when computing the power of the test we basically find the probability that we are above (or below, depending on the test) the critical value if the true hypothesis is the alternative one, in order to do so, we must choose a value in that critical region and the power of the test will be different depending on that value.

\begin{equation}
H_0:\{ \mu=\mu_0\},\quad H_1:\{ \mu=\mu_1\}
\end{equation}

The interpretation of the power of the test goes along the following lines. Suppose we are performing a test on the effectiveness of a new medicine, in particular let's say we have a control group and a treatment group and then we first want to see if there is significant difference between the way the medicine acts in both groups (the null hypothesis would be that there is not), but now we want to go a bit farther since it would be dangerous if the levels of a blood component are depressed too much, then we may want to know what is the probability that if this drop actually occurs, we may detect it from the data of our sample. This last probability is precisely the power of the test.

The power of the test may be computed in two different ways (results will be different but in the same direction). The first method implies choosing a significance level, say the $5\%$ and then perform the calculations which will not make any explicit use of the sample mean value. The second method implies choosing as significance level the p-value and then we make explicit use of the sample mean value as defining the boundary of the critical region. If we are going to compute the power using R we must understand that it is the first option the one used in the functions, and then we do not need to use our data set as argument of the functions. If we would like to compute the power from the second option, then we have to find the p-value and then use it as argument of the second function.\\

{\sc{Example.}} Find and interpret the power of the test in the example of the ages of students if you want are interested in knowing if you could detect that the true mean age in the group is 21 years old.\\

This means that our hypotheses are

\begin{equation}
H_0:\{\mu=25\},\quad H_1:\{\mu=21\}
\end{equation}

for the computation in R we need the true difference of the means

\begin{equation}
\Delta = \mu_0-\mu_1=25-21=4
\end{equation}

then we have

```{r}
age <- c(18.5,17.6,21.1,17.1,49.0,25,18.2,17.9,18.1,18.0,
         18.9,17.4,25.4,17.4,19.5,17,3,25.3,27.7,24.0)

m <- mean(age)
sdev <- sd(age)
n <- as.numeric(length(age))

power.t.test(n=n,
             delta=4,
             sd=sdev,
             alternative = "t"
             )
```

where the default significance level is $5\%$, however if we want to use the p-value we should use find it and either store it in one variable or pass it directly as an argument

```{r}
slevel <- t.test(age,mu=25,alternative = "t")$p.value
power.t.test(n=n,
             delta=4,
             sig.level=slevel,
             sd=sdev,
             alternative = "t"
             )
```

In both cases we obtain a list of outputs in which we find the element power that returns the power of the test. Then

\begin{equation}
\text{power}_1=0.3131,\quad \text{power}_2=0.2689
\end{equation}

In this case we have used an alternative two-tailed test since that was our previous test. However, we will later see that this is not trully a must and we could use a one-tailed test instead.

This implies that if the true mean age were 21 years old, the probability that we may detect it from our sample would be jsut the $31.31\%$ (or $26.89\%$), a rather low probability.

## Sample Size and Power

Just as in the case of the p-value, the power of the test should reflect the properties of the problem and the severity, in this case, of a type II error. Typically we will require a value for the power of the test between $80\%$ and $90\%$. If we impose a power, the only paramenter that may change is the sample size for a fixed significance level. This in turn implies that we should not use the p-value in this computation: if we could vary the sample size the p-value would change too, then we proceed by considering an $\alpha$ and then imposing a $\beta$ to determine the smaple size. In R we can use the same functions as before but leaving the sample size argument free.\\

{\sc{Example.}} Find the sample size needed to have a power of the test of the $80\%$, $85\%$ and $90\%$ in the previous examples if $\alpha=0.05$.\\

We can directly use

```{r}
power.t.test(delta=4,
             power = 0.80,
             sd=sdev,
             alternative = "t"
             )
```

and the same changing the value of {\emph{power}}. What we obtain is

\begin{equation}
n_{80}=69.72,\quad n_{85}=79.62,\quad n_{80}=93.01
\end{equation}

i.e. a way larger sample than the one we have.\\

The example above let us see that there is a dependency betweeen the sample size and the power of the test for every fixed significance level. This may allow us to make a plot Power vs. n, known as {\bf{power curve}}.\\

{\sc{Example.}} Use R to plot the power curve in the example above.\\

The idea behind the code is that we have to build a function that returns the value of $n$ for a set of values of the power, for this we can create a vector storing the values of the power and for a fixed value of $\alpha=0.05$, create another empty array in which we will store the corresponding sample sizes and then find the values of $n$, which are just the element $n$ in the output list of the test. After this we create an empty plot with the right assignment of axis and then plot the power vs. size line.\\

```{r}
p <- seq(0.10,0.99,0.01)
np <- as.numeric(length(p))

arr.2t <- array(dim = np)
for(i in 1:np){
  pow <- power.t.test(delta = 4,
                      sd=sdev,
                      power = p[i],
                      alternative = "t")
  arr.2t[i] <- ceiling(pow$n)
}

xrange <- range(arr.2t)
yrange <- range(p)
plot(xrange,
      yrange,
      type="n",
      ylab="Power of the test",
      xlab="Sample Size",
      main = "Power")
lines(arr.2t,
      p,
      col="blue",
      lwd=1)
```

The plot we find is the following

\begin{center}
  \includegraphics[scale=0.4]{Images/Power.eps}
\end{center}

{\sc{Going Further.}} Can you make the following plot to see the dependency of the power with the difference $\Delta$? How would you interpret it?

\begin{center}
  \includegraphics[scale=0.5]{Images/PowDelta.eps}
\end{center}

```{r}
%% The code is the following:\\

%% \lstset{language=R}
%% \begin{lstlisting}
%% delta <- seq(-20.1,20.1,1)
%% nd <- as.numeric(length(delta))

%% n <- seq(10,20,2)
%% nn <- as.numeric(length(n))

%% pow.n <- array(dim = c(nn,nd))

%% for(j in 1:nn){
%%   for(i in 1:nd){
%%     pow <- power.t.test(n=n[j],
%%                         delta = delta[i],
%%                         sd=sdev,
%%                         alternative = "t")
%%     pow.n[j,i] <- pow$power
%%   }
%% }

%% xrange <- range(delta)
%% yrange <- range(pow.n)
%% colors <- rainbow(nn)
%% plot(xrange,
%%      yrange,
%%      type="n",
%%      ylab="Power",
%%      xlab="Delta",
%%      main = "Power of the Test")
%% for(i in 1:nn){
%%   lines(delta,
%%         pow.n[i,],
%%         col=colors[i],
%%         lwd=1)
%%   }
%% legend("top",
%%        title = "Sample Size",
%%        as.character(n),
%%        fill = colors, bty="n")
%% \end{lstlisting}
```

## How to Choose the Significance Level

A very important point is that of what significance level we should choose for a study and, of course, since they are not independent, this affects the value of the power.

The key point is that the significance level must reflect the importance of the consequences of a type I error. In the same sense, the election of $\beta$ should reflect the consequences of a type II error, however, since decreasing $\alpha$ increases $\beta$ for the same sample size, we must be careful with this election.

Suppose a pharmaceutical company is developing a new drug and in some previous tests they have found that it makes a great job with the illness it fights, but at the same time it has been detected some unwilling side effects that may result in the sudden death of the patient, then they are considering a new test in order to see if they should extend the studies or not. This scheme can be written as 

\begin{equation}
H_0:\{\text{side effects}\},\quad H_1:\{\text{no side effects}\}
\end{equation}

in this case, if our decision is a type I error, it implies that we may say that the drug does not produce the side effects when it truly does. Since we may want to minimize the probability that we fall into this wrong decision we should choose the smallest possible $\alpha$ (in our previous discussions, the $1\%$). However, if the unwilling side effects are related to headaches, nauseas or dizziness, the company may relax the $\alpha$ to a $10\%$ since it is not a high risk situation.

This kind of situation is what makes us require very small p-values when performing some crucial experiments, for example, in particle physics, quality control or in other real life situations it is usually required that the p-value is located at 5 or 6 standard deviations from the null hypothesis. A value of 6 sigmas means that the probability outside of the null hypothesis in each tail of a two-tailed test is $0.00017\%$ and so, the p-value is $1.7\cdot 10^{-6}$, i.e. if the results were due to chance and we repeat the experiment 3.4 million times we expect that only 1 of them replicates the result.

The way to impose both, a small significance level and a high power of the test is by increasing the sample size. This just will allow us to have balanced and small $\alpha$ and $\beta$. We can see this in the examples above using R as\\

```r
sizes <- c(3,40,100,200,300,400)
ls <- as.numeric(length(sizes))

tab <- matrix(sizes,ncol=1)
v <- array(dim=ls)
for(i in 1:ls){
  pow <- power.t.test(n=tab[i,1],
                      delta = 4,
                      sd=sdev,
                      sig.level = 0.001)
  v[i] <- pow$power
}
tab <- cbind(tab,1-v)
View(tab)
```

and then we find that

\begin{center}
  \begin{tabular}{ccc}
    \toprule[0.1em]
            {\bf{n}} & $\alpha$ &  $\beta$ \\\midrule
            3 & 0.001 & 0.998 \\
            40 & 0.001 & 0.890 \\
            100 & 0.001 & 0.483 \\
            200 & 0.001 & 0..073 \\
            300 & 0.001 & 0.006\\
            400 & 0.001 & 0.0003 \\
            \bottomrule[0.1em]
  \end{tabular}
\end{center}

import numpy as np
import pandas as pd
import scipy.stats as ss
import math

from statsmodels.stats.power import TTestPower # There is no normal distribution here

from google.colab import drive
drive.mount('mydrive')

gifted = pd.read_csv('/content/mydrive/My Drive/IE - Statistics and Data Analysis - DUAL - 2019/DataSets/gifted.csv')
gifted.head(5)

**Using the `count` variable let's answer the following: In a report you have read that the average age at which gifted children are able to count up to 20 is 31 months. However, a new research claims that the children should be denoted as gifted only if this age is actually 30 months. Find the probability that you can detect such an average age from your sample if it were actaully the case. Use a significance level of 1%. What would be the sample size needed in case you want that this probability is 99%. Assume that the population standard deviation is of 3 months.**

gifted['count'].mean()

The decision scheme is

\begin{equation}
H_0:\{\mu \geq 31\},\quad H_1:\{\mu < 31\}
\end{equation}

so it is a left-tailed test. Once we have the alternative value, this becomes

\begin{equation}
H_0:\{\mu = 31\},\quad H_1:\{\mu = 30\}
\end{equation}

# population means
mu0 = 31
mu1 = 30

# population standard deviation
sigma = 3
stdev = gifted['count'].std()

# sample size
n = len(gifted['count'])

# significance level
SL = 0.01

Let's find the **effect size** and the **z-value**

# Effect Size 
delta = mu0 - mu1

# z value
zval = ss.norm.isf(SL)

Now we can find the **beta** and the **power of the test**

# power of the test
power = ss.norm.cdf(delta/(sigma/np.sqrt(n)) - zval)
beta = 1 - power

print('The power of the test is: {:1.8f}\nThe beta is: {:1.8f}'.format(power, beta))

Let's find the **sample size** needed for a power of the test of 99%

power = 0.99

# B value
B = ss.norm.ppf(power)

# new sample sample size
new_n = ((B + zval) * sigma / delta)**2

# print the output
print('The sample size needed is {:1.2f}'.format(np.ceil(new_n)))

analysis = TTestPower()

# find the power
powerTest = analysis.solve_power(effect_size = (mu0 - mu1)/sigma, nobs = n, alpha = SL, alternative = 'larger')
new_n = analysis.solve_power(effect_size = (mu0 - mu1)/sigma, power = power, alpha = SL, alternative = 'larger')

# print the output
print('The probability of a Type II error is {:1.6f}\nThe power of the test is {:1.6f}'.format(1 - powerTest, powerTest))
print('The sample size for a 99% of power is: ', np.ceil(new_n))

**Using the `speak` variable let's answer the following: In a report you have read that the average age at which gifted children are able to speak is 17 months. However, a new research claims that the children should be denoted as gifted only if this age is actually 18.5 months. Find the probability that you can detect such an average age from your sample if it were actaully the case. Use a significance level of 5%. What would be the sample size needed in case you want that this probability is 95%**

gifted['speak'].mean()

The decision scheme is 

\begin{equation}
H_0:\{\mu\leq 17\},\quad H_1:\{\mu > 17\}
\end{equation}

then once we are given an alternative value, this becomes:

\begin{equation}
H_0:\{\mu = 17\},\quad H_1:\{\mu = 18.5\}
\end{equation}

# population means
mu0 = 17
mu1 = 18.5

# sample values
mean = gifted['speak'].mean()
stdev = gifted['speak'].std()
n = len(gifted['speak'])

# significance level
SL = 0.05

# critical t
tcrit = ss.t.isf(SL, n-1)

# effect size
delta = mu0 - mu1

# beta value
beta_val = delta / (stdev/(np.sqrt(n))) + tcrit

# power of the test
beta = ss.t.cdf(beta_val, n-1)
power = 1 - beta

# print the output
print('The probability of a Type II error is {:1.6f}\nThe power of the test is {:1.6f}'.format(beta, power))

The probability that we may detect that the average age at which gifted children speak is 18.5 months is 86.0.8% if that age is actually the true age.

Let's now use the `statsmodels` module to find the value of the power of the test

analysis = TTestPower()
powerTest = analysis.solve_power(effect_size = (mu0 - mu1)/stdev, nobs = n, alpha = SL, alternative = 'smaller')

print('The probability of a Type II error is {:1.6f}\nThe power of the test is {:1.6f}'.format(1 - powerTest, powerTest))

I have used the alternative as smaller because $\mu_1$ is greater than $\mu_0$ and then the argument for the effect size in `power()` is negative. If you want to respect the decision scheme structure you have to ALWAYS use a positive value of the `effect_size` argument

In order to find the sample size needed for a power of 95%, we are going to use the normal approximation (not needed)

A = ss.norm.ppf(0.05)

# critical z
zcrit = ss.norm.isf(SL)

# sample size needed
new_n = ((A - zcrit) * stdev / delta)**2

# print the output
print('The sample size needed is {:1.2f}'.format(new_n))

Let's find the sample size using `statsmodels` and the `solve_power` function

pwr = 0.95
sample_size = analysis.solve_power(effect_size = (mu0 - mu1)/stdev, power = pwr, alpha = SL, alternative = 'smaller')

# print the output
print('The sample size needed is {:1.2f}'.format(sample_size))

**Using the `score` variable let's answer the following: In a report you have read that the average IQ score of gifted children 155. However, you do not fully agree with this value and you claim that it is different to it. On a second approach you want to see what is the probability that you may find that the average IQ score is 160 for a 1% of significance level. What would be the sample size needed in case you want that this probability is 99.99%**

gifted['score'].mean()

# population means
mu0 = 155
mu1 = 160

# significance level
SL = 0.01
pwr = 0.99999

# sample information
n = len(gifted['score'])
stdev = gifted['score'].std()

# critical t
tcrit = ss.t.isf(SL/2, n-1)

# boundaries
delta = mu0 - mu1
std_delta = delta / (stdev/(np.sqrt(n)))

upp = std_delta + tcrit
low = std_delta - tcrit

# power of the test
beta = ss.t.cdf(upp, n-1) - ss.t.cdf(low, n-1)
power = 1 - beta

# print the output
print('The probability of a Type II error is {:1.6f}\nThe power of the test is {:1.6f}'.format(beta, power))

Let's find the power of the test using `statsmodels`.

power_analysis = TTestPower()

powerTest = power_analysis.solve_power(effect_size = (mu1 - mu0)/stdev, nobs = n, alpha = SL, alternative = 'two-sided')

'''
The effect_size must be positive and it is delta/stdev
'''

# Print the output
print('The probability of a Type II error is {:1.6f}\nThe power of the test is {:1.6f}'.format(1-powerTest, powerTest))

Let's find the **sample size**

new_n = power_analysis.solve_power(effect_size = (mu1 - mu0)/stdev, power = pwr, alpha = SL, alternative = 'two-sided')

'''
The effect_size must be positive and it is delta/stdev
'''

# Print the output
print('The sample size needed is', np.ceil(new_n))