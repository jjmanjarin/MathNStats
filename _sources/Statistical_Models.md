# Statistical Models

When we have a complex data set the process of statistical learning allow us to modelize and understand it. Under this umbrella lie many different approaches and we will only deal here with the most basic ones. But before entering into the details, we may need a brief discussion of what is understood about modeling and, most importantly what we can and cannot obtain from the models we build.

We have already mentioned the types of variables that we can find in a data set with respect to their role, among all the possible names we have {\bf{response}}, {\bf{predictors}} and {\bf{confounding}}. Remember that the response are the dependent or endoenous variables and the predictors are also known as independent, exogenous or features, to mention just a few of their names. The way we will always denote them is $Y$ as the response variable and $X$ as the predictors and we must keep in mind the generally $X$ will be a vector of variables, i.e. it will generally make reference to a whole set of independent variables.

In this general context, modeling the behavior of the response in terms of the predictors mean that we say

\begin{equation}
Y = f(X)+u
\end{equation}

In this general form $f(X)$ corresponds to the systematic part of the response and $u$ to the random errors, and the process of estimation or modelling is that of finding a best fit functional form for $f(X)$ which will always be dependent of the data set we have. The main reasons behind estimation are {\bf{prediction}} and {\bf{inference}}.

## Prediction vs. Estimation

We talk about {\bf{prediction}} when we want to use the estimated model

\begin{equation}
\hat Y = \hat f(X)
\end{equation}

to obtain particular values of the response variable. A rather common example may be that of predicting the income of a person in terms of the years of education.

In this sense we must understand that, in general, the estimated value will be different from the measured value for the same $X$ and even more, both of them will be different from the actual true value which we will never know. This allow us to compute the MSE of the model as

\begin{equation}
E[(Y-\hat Y)^2]=[bias(\hat f(X)]^2+Var(\hat f(X))+Var(u)
\end{equation}

which is splitted in an irreducible term defined by the variance of the error term and another reducible written as the MSE of the estimator defined by the model. In the context of Statistical Decision Theory, this expected value is known as the {\bf{expected predition error}} (EPE) and is the expected value of the {\bf{loss function}}.

It is the minimization of this EPE function is what leads to the different models we will see in this chapter and in general all the models try to minimize the reducible part of the function.

If we are not interested in predicting values of the response but on seen the way in which $Y$ is affected by the different $X$ we talk about {\bf{inference}} of the model. It is in this context where all the notions of hypothesis testing come to our rescue: which predictors are associated to the response? Which set of variables can we use to describe the behavior of the response? What is the individual relation between predictors and response?

The estimation processes we will see along this chapter can be described as parametric, in the sense that we will make an explicit assumption on the model for $f(X)$ and then try to estimate it. This is what we do with the linear regression or probability models, where we assume that the functional form of the response as a function of $X$ is that of a linear function

\begin{equation}
Y=\beta_0+\beta_i\,X_i
\end{equation}

The main parametric procedure is that of {\bf{least squares}} that we will shortly view.

Just as a comment, we must keep in mind that this is not the only approach, for example, the {\bf{splines}} are non-parametric since they do not assume any specific functional form. Each procedure has its own pros and cons that are somewhat far from the scope of the course.

There is, however, one potential problem with parametric models and it is that since we do not really know the population model, the estimated model will not follow the actual true values. The idea is that in general they are designed to capture a particular property of the population through a simplication of the true overall behavior.

## Supervised vs. Unsupervised Learning

In our previous discussion we have implicitely assumed that we know the response variables. However, that is not always the case, a for example in a market segmentation study, where we split the population according to some of their properties: age, income, zip code or any other, but we want to understand the spending habits of that population. These last partterns will never be known for us.

If we know the response variable we talk about {\bf{supervised learning}}, in the sense that the response let us see how close to it we are with our model and then, in some sense it is supervising us. However, if we do not know it, we speak of an {\bf{unsupervised learning}}. In this context, the regression and classification linear models we will study are always supervised procedures.

## Regression vs. Classification Models

Another important differentiation in the models is that of regression and classification. In this case we are making reference to the nature of the response variable. If the response variable is quantitative, the model will be a {\bf{regression}}, but if the response is qualitative, the model will be a {\bf{classification}}, then if we want to study how gender affects income, that will be a regression problem. However, if we want to see if we may determine the gender from the income, we will be classifying the population.

## Validation of the Model

The general timeline in the process of building any model is the following

\begin{equation}
\text{Training} \longrightarrow \text{Validation} \longrightarrow \text{Test}
\end{equation}

In this context we must keep in mind that the training and testing must be done on different data sets which receive, accordingly the names of {\bf{training set}} and {\bf{test set}}. In R we can perform a splitting of our sample using the following code\\

```r
set.seed(111)
sampling <- sample.int(n=nrow(mydata.comp),
                       size=floor(0.8*nrow(mydata.comp)),
                       replace = FALSE)

training.set <- mydata[sampling,]
test.set <- mydata[-sampling,]
```

where we have followed a 80-20 rule, so that the $80\%$ of the sample is used for the training of the model and the remaining $20\%$ will be considered as the test set.

The validation procedure is understood as the {\bf{model selection}}, i.e. the procedure in which we estimate the performance of different models by including or excluding variables and through the study of all the potential problems that may arise in the training (outliers, heteroskedasticity,.... On the hand, the testing is understood as the {\bf{model assessment}} where, once found the final model we assess the model by estimating the error on the test data set.

Once we make a difference in the training and test sets, we can see that the idea is that the model built from the training data should be accurate enough once we use it in previously unseen data. If we have, for example, built a model from clinical measurements in a study for the prediction of diabetes or to infer their relation with diabetes, we want to be able to use it with future patients. Mathematically we want a model that gives the lowest MSE for the test set, not for the training set.

In some cases we want to build flexible statistical models so that we decrease the value of the training MSE, but it may occur that the MSE increases with the flexibility. In these cases the model suffers what is known as {\bf{overfitting}} which implies that our model is finding some patterns due to random effects and not to the true behavior, i.e. it is following the white noise of the model. 

## Bias-Variance Trade-off

From the equation for the EPE, we see that the way to minimize it is by choosing a method that produces both, a low bias and a low variance. The variance we are interested on is the amount by which the estimated model would change if we use a different training set, while the bias refers to the error introduced by choosing an approximation to the real life problem.

In flexible models, the variance will increase and the bias will decrease, i.e. we will be closer to the values but in a very unstable way (this is the case of {\bf{K-nearest neighbourhood method}}). On the other hand with restricted models, we obtain models with low variance but large bias, i.e. maybe not close to the points but very stable (this is the case of the {\bf{linear regression}}). It would be an error thinking that a flexible model may overperform another simpler and more restricted, in fact it is not unsual to find the contrary situation, a counterintuitive situation that arises due to the overfitting problem.

It is relatively easy to deal with methods that gives a low value of the bias or of the variance, but not together. In real life it is almost always imposible to find the value of the MSE, bias or variance for the test set but we must always keep in mind that the methods we use deal with a {\bf{bias-variance trade-off}} in which we typically minimize one property or another. 
