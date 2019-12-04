---
layout: post
title: "Predicting If Customer Will Subscribe to a Term Deposit"
date: 2019-11-26
excerpt: "Using Machine Learning to Make Better Marketing Desicion"
tags: [Machine Learning, Predictive Modelling, Logistics Regression, Python]
comments: true
---

## Predicting If Customer Will Subscribe to a Term Deposit

On November 9th to 10th 2019, my team **won the First Place for Best Model Award** at Data Hackathon (a less than 24 data science competition) held by **OCRUG and Merage Analytics Club** at The Paul Merage School of Business. We were a team of 3 people from UCI Master's in Business Analytics Program. Below is the report of our analysis and prediction for the competition.

### Background:
The data is about direct marketing campaigns (phone calls) of a Portuguese banking institution from May 2008 to November 2010. The clients were contacted more than once in order to access if the product (bank term deposit) would be (‘yes’) or not (‘no’) subscribed.

### Goal:
Predict whether the client will subscribe (yes/no) to a term deposit. Key Process Owners can use the output of this prediction to improve the stragegy for the next market campaign.

### Data Dictionary:
<u>**Independent Variable**</u> <br>
**Bank client data** <br>
- Age (numeric) <br>
- Job : type of job <br>
- Marital : marital status <br>
- Education <br>
- Default: has credit in default? (“yes”,“no”) <br>
- Balance: average yearly balance, in euros <br>
- Housing: has housing loan? (“yes”,“no”) <br>
- Loan: has personal loan? “yes”,“no”) <br><br>

**Related with the last contact of the current campaign** <br>
- Contact: contact communication type <br>
- Day: last contact day of the month <br>
- Month: last contact month of year <br>
- Duration: last contact duration, in seconds (numeric) <br><br>

**Other variables** <br>
- Campaign: number of contacts performed during this campaign and for this client <br>
- Pdays: number of days that passed by after the client was last contacted from a previous campaign (-1 means - client was not previously contacted) <br>
- Previous: number of contacts performed before this campaign and for this client <br>
- Poutcome: outcome of the previous marketing campaign <br>

**<u>Dependent Variable (desired target)</u>**<br>
- Deposit - has the client subscribed a term deposit? (“yes”,“no”)


```python
url = 'https://raw.githubusercontent.com/ocrug/hackathon-2019-11/master/data/bank-full.csv'
bank = pd.read_csv(url,sep = ';', encoding='latin1')
```

## Data Cleaning
- The first step we took to clean the data is to check for any duplication, null, or NA values.
- Then, we altered any variables with "yes" and "no" values to binary 1 and 0 to make it easier for analysis and model builing.

Besides the above, the overall dataset is already clean enough. Therefore, we moved forward with the EDA after the above steps.


    duplicated data : 0
    null data : 0
    NA data : 0

```python
bank.replace(('yes', 'no'), (1, 0), inplace=True)
```

## Exploratory Data Analysis (EDA)
In the EDA part, we wanted to visualize and analyze the relationship between the independent variables to each other as well as to the depedent variable (deposit).

- The chart below showed that most of the customer in the dataset were blue collars. However, customers who are retired, students, unemployed, or work as management has higher percentage of subscribing to the term deposit compared to other job titles.

![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_9_0.png)

![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_9_1.png)


- The chart below showed that customers who successfully subscribed from the previous marketing campaign will be most likely to subscribe again. 

![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_11_0.png)



![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_11_1.png)


- The chart below showed that customers who were single were most likely to subscribe as compared to customers who are divorced or married.

![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_13_0.png)



![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_13_1.png)


- The chart below showed that most data were collected in May. However, there were higher percentage of customer subscribring in the month of Feb, Mar, Apr, Oct, Sep, Dec.

![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_15_0.png)



![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_15_1.png)


## Correlation Matrix and Chi Square Test
Once we analyzed the breakdown of the dependent variable on each of these independent variables, we wanted to make sure that this effects are not due to random chance. Therefore, we used chi-squared test to see their significant relationship.
We also wanted to see the correlation between the continuous independent variables to the dependent variable using the correlation matrix.
- The result from the correlation matrix showed that there might be collinearity between pdays and previous variables since both variables technically measured the same thing. Therefore, we will remove one of them from the model.
- **duration** showed very high correlation to the deposit variable. This made sense since we knew that it is impossible to get the duration data before a call is performed. Also, after the end of the call **deposit** result is obviously known. Therefore, this input would be discarded in order to have a realistic predictive model.
- In terms of the Chi-Squared Test, the result was favorable in that all variables showed significant relationship to the dependent variables. This means that all variables are important in predicting the **deposit**.


![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_17_0.png)



```python
pd.set_option('display.float_format', lambda x: '%.8f' % x)
def chisq_of_df_cols(df, c1, c2):
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    return(scs.chi2_contingency(ctsum.fillna(0)))
```


```python
categorical = bank[set(bank.columns)-set(corr.columns)]
variable = []
Chi_Squared_Val = []
P_Value = []

for col in categorical.columns:
    variable.append(col)
    Chi_Squared_Val.append(chisq_of_df_cols(categorical, col,bank.deposit)[0])
    P_Value.append(chisq_of_df_cols(categorical, col,bank.deposit)[1])
```

![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/chi.png)


```python
bank.drop(columns=['duration'],inplace=True)
```

## Outlier Detection
There were outliers in the previous and p-days columns. Therefore, we capped the data to top 90 - 93%.
- previous top 90% value =  2.0
- pdays top 90% value =  185.0
- pdays top 93% value =  6.0



```python
bank[[i for i in list(set(bank.columns) - set(variable)) if i != 'deposit']].describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pdays</th>
      <th>campaign</th>
      <th>balance</th>
      <th>housing</th>
      <th>previous</th>
      <th>day</th>
      <th>default</th>
      <th>age</th>
      <th>loan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45211.00000000</td>
      <td>45211.00000000</td>
      <td>45211.00000000</td>
      <td>45211.00000000</td>
      <td>45211.00000000</td>
      <td>45211.00000000</td>
      <td>45211.00000000</td>
      <td>45211.00000000</td>
      <td>45211.00000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.19782796</td>
      <td>2.76384066</td>
      <td>1362.27205769</td>
      <td>0.55583818</td>
      <td>0.58032337</td>
      <td>15.80641879</td>
      <td>0.01802659</td>
      <td>40.93621021</td>
      <td>0.16022649</td>
    </tr>
    <tr>
      <th>std</th>
      <td>100.12874599</td>
      <td>3.09802088</td>
      <td>3044.76582917</td>
      <td>0.49687781</td>
      <td>2.30344104</td>
      <td>8.32247615</td>
      <td>0.13304894</td>
      <td>10.61876204</td>
      <td>0.36682004</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.00000000</td>
      <td>1.00000000</td>
      <td>-8019.00000000</td>
      <td>0.00000000</td>
      <td>0.00000000</td>
      <td>1.00000000</td>
      <td>0.00000000</td>
      <td>18.00000000</td>
      <td>0.00000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.00000000</td>
      <td>1.00000000</td>
      <td>72.00000000</td>
      <td>0.00000000</td>
      <td>0.00000000</td>
      <td>8.00000000</td>
      <td>0.00000000</td>
      <td>33.00000000</td>
      <td>0.00000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.00000000</td>
      <td>2.00000000</td>
      <td>448.00000000</td>
      <td>1.00000000</td>
      <td>0.00000000</td>
      <td>16.00000000</td>
      <td>0.00000000</td>
      <td>39.00000000</td>
      <td>0.00000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-1.00000000</td>
      <td>3.00000000</td>
      <td>1428.00000000</td>
      <td>1.00000000</td>
      <td>0.00000000</td>
      <td>21.00000000</td>
      <td>0.00000000</td>
      <td>48.00000000</td>
      <td>0.00000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>871.00000000</td>
      <td>63.00000000</td>
      <td>102127.00000000</td>
      <td>1.00000000</td>
      <td>275.00000000</td>
      <td>31.00000000</td>
      <td>1.00000000</td>
      <td>95.00000000</td>
      <td>1.00000000</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's revisit the boxplot of each of these continous independent variables after we capped the outliers.
![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_26_1.png)


## Prepare the Data for Model
For the first model, we will included the important variables and then conducted the backward elimitation methodology to enhance the performance of the model.


```python
#creating one-hot variables
bank_cleaned = bank
for i in variable:
    bank_cleaned = pd.get_dummies(bank_cleaned, columns = [i])
bank_cleaned.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>day</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>deposit</th>
      <th>...</th>
      <th>month_jun</th>
      <th>month_mar</th>
      <th>month_may</th>
      <th>month_nov</th>
      <th>month_oct</th>
      <th>month_sep</th>
      <th>poutcome_failure</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>0</td>
      <td>2143</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>1.00000000</td>
      <td>-1.00000000</td>
      <td>0.00000000</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>0</td>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>1.00000000</td>
      <td>-1.00000000</td>
      <td>0.00000000</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1.00000000</td>
      <td>-1.00000000</td>
      <td>0.00000000</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>0</td>
      <td>1506</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>1.00000000</td>
      <td>-1.00000000</td>
      <td>0.00000000</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1.00000000</td>
      <td>-1.00000000</td>
      <td>0.00000000</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>




```python
bank_cleaned.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45211 entries, 0 to 45210
    Data columns (total 48 columns):
    age                    45211 non-null int64
    default                45211 non-null int64
    balance                45211 non-null int64
    housing                45211 non-null int64
    loan                   45211 non-null int64
    day                    45211 non-null int64
    campaign               45211 non-null float64
    pdays                  45211 non-null float64
    previous               45211 non-null float64
    deposit                45211 non-null int64
    contact_cellular       45211 non-null uint8
    contact_telephone      45211 non-null uint8
    contact_unknown        45211 non-null uint8
    education_primary      45211 non-null uint8
    education_secondary    45211 non-null uint8
    education_tertiary     45211 non-null uint8
    education_unknown      45211 non-null uint8
    job_admin.             45211 non-null uint8
    job_blue-collar        45211 non-null uint8
    job_entrepreneur       45211 non-null uint8
    job_housemaid          45211 non-null uint8
    job_management         45211 non-null uint8
    job_retired            45211 non-null uint8
    job_self-employed      45211 non-null uint8
    job_services           45211 non-null uint8
    job_student            45211 non-null uint8
    job_technician         45211 non-null uint8
    job_unemployed         45211 non-null uint8
    job_unknown            45211 non-null uint8
    marital_divorced       45211 non-null uint8
    marital_married        45211 non-null uint8
    marital_single         45211 non-null uint8
    month_apr              45211 non-null uint8
    month_aug              45211 non-null uint8
    month_dec              45211 non-null uint8
    month_feb              45211 non-null uint8
    month_jan              45211 non-null uint8
    month_jul              45211 non-null uint8
    month_jun              45211 non-null uint8
    month_mar              45211 non-null uint8
    month_may              45211 non-null uint8
    month_nov              45211 non-null uint8
    month_oct              45211 non-null uint8
    month_sep              45211 non-null uint8
    poutcome_failure       45211 non-null uint8
    poutcome_other         45211 non-null uint8
    poutcome_success       45211 non-null uint8
    poutcome_unknown       45211 non-null uint8
    dtypes: float64(3), int64(7), uint8(38)
    memory usage: 5.1 MB


## Split Train and Test Data
To evaluate the model, we used the holdout method and splitted the data into 80% for **model training** and 20% for **model testing/evaluation**. 
Then, we measured the baseline percenteage between each classes in the overall dataset and compare it with the train and test dataset to see if the randomized split was representative enough. The result was favorable, the test and train dataset both had very similar percentage breakdown for the dependent variable.


```python
print(bank_cleaned.deposit.value_counts())
print(bank_cleaned.deposit.value_counts()/(len(bank.deposit)+1))
```

    0    39922
    1     5289
    Name: deposit, dtype: int64
    0   0.88299566
    1   0.11698222
    Name: deposit, dtype: float64


**Now splitting the train and test data**


```python
split = np.random.rand(len(bank_cleaned)) <= .80
train = bank_cleaned[split]
test = bank_cleaned[~split]
```

### Train Data Breakdown
    0   0.88228142
    1   0.11769095
    Name: deposit, dtype: float64



### Test Data Breakdown
```python
test.deposit.value_counts()/(len(test.deposit)+1)
```
    0   0.88576177
    1   0.11412742
    Name: deposit, dtype: float64


## Model 1
- Our first model had a really high accuracy of ~89%. However, the recall was ~18.8%. This indicated that the model was predicting a lot of "no subscription" and it is not representative enough because it was highly skewed to only one class of the predicted variable. It will not work well if we evaluate the model on data with a lot of "yes".
- This happened since we had an **imbalance dataset**. We would like to create a more representative model that can handle more variability. Therefore, we built the next model with a **modified or different sampling method** on the training dataset.


```python
##for dummy vars, we only need N-1
train = train.drop(columns=['job_unknown','marital_divorced','poutcome_other','contact_cellular','education_unknown'])
test = test.drop(columns=['job_unknown','marital_divorced','poutcome_other','contact_cellular','education_unknown'])
```


```python
X_train = train[train.columns[train.columns != 'deposit']]
y_train = train[train.columns[train.columns == 'deposit']]


X_test = test[test.columns[test.columns != 'deposit']]
y_test = test[test.columns[test.columns == 'deposit']]
#X_train
```


```python
##No longer using sklearn due to no summary statistics
#logit = LogisticRegression()
#logit.fit(X_train, y_train)
#y_pred = logit.predict(X_test)
logit_model = sm.Logit(y_train, X_train).fit()
print(logit_model.summary())
```

    Optimization terminated successfully.
             Current function value: 0.300826
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                deposit   No. Observations:                36187
    Model:                          Logit   Df Residuals:                    36145
    Method:                           MLE   Df Model:                           41
    Date:                Sat, 30 Nov 2019   Pseudo R-squ.:                  0.1697
    Time:                        23:21:43   Log-Likelihood:                -10886.
    converged:                       True   LL-Null:                       -13111.
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    =======================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    age                     0.0014      0.002      0.631      0.528      -0.003       0.006
    default                -0.1694      0.164     -1.036      0.300      -0.490       0.151
    balance               1.99e-05   4.96e-06      4.012      0.000    1.02e-05    2.96e-05
    housing                -0.5572      0.043    -12.940      0.000      -0.642      -0.473
    loan                   -0.3957      0.059     -6.664      0.000      -0.512      -0.279
    day                     0.0027      0.002      1.084      0.278      -0.002       0.008
    campaign               -0.1179      0.013     -8.931      0.000      -0.144      -0.092
    pdays                  -0.0021      0.001     -2.734      0.006      -0.004      -0.001
    previous                0.2561      0.075      3.425      0.001       0.110       0.403
    contact_telephone      -0.2439      0.073     -3.345      0.001      -0.387      -0.101
    contact_unknown        -1.3145      0.070    -18.756      0.000      -1.452      -1.177
    education_primary      -0.2344      0.103     -2.284      0.022      -0.435      -0.033
    education_secondary    -0.0713      0.090     -0.790      0.430      -0.248       0.106
    education_tertiary      0.0843      0.095      0.888      0.374      -0.102       0.270
    job_admin.              0.2819      0.236      1.194      0.233      -0.181       0.745
    job_blue-collar         0.1488      0.235      0.633      0.527      -0.312       0.610
    job_entrepreneur        0.0889      0.255      0.349      0.727      -0.411       0.588
    job_housemaid          -0.1039      0.259     -0.401      0.689      -0.612       0.404
    job_management          0.2081      0.234      0.888      0.374      -0.251       0.667
    job_retired             0.6343      0.240      2.643      0.008       0.164       1.105
    job_self-employed       0.2557      0.248      1.031      0.303      -0.230       0.742
    job_services            0.2333      0.239      0.976      0.329      -0.235       0.702
    job_student             0.6144      0.248      2.474      0.013       0.128       1.101
    job_technician          0.1737      0.234      0.741      0.459      -0.286       0.633
    job_unemployed          0.4274      0.248      1.725      0.085      -0.058       0.913
    marital_married        -0.1800      0.058     -3.079      0.002      -0.295      -0.065
    marital_single          0.1339      0.067      2.008      0.045       0.003       0.265
    month_apr              -1.0249      0.330     -3.104      0.002      -1.672      -0.378
    month_aug              -1.8817      0.329     -5.716      0.000      -2.527      -1.236
    month_dec              -0.3764      0.366     -1.027      0.304      -1.094       0.342
    month_feb              -1.4508      0.329     -4.411      0.000      -2.095      -0.806
    month_jan              -2.1788      0.343     -6.351      0.000      -2.851      -1.506
    month_jul              -1.7086      0.330     -5.174      0.000      -2.356      -1.061
    month_jun              -0.8912      0.328     -2.716      0.007      -1.534      -0.248
    month_mar               0.1450      0.343      0.423      0.673      -0.527       0.818
    month_may              -1.4663      0.328     -4.474      0.000      -2.109      -0.824
    month_nov              -1.8762      0.331     -5.670      0.000      -2.525      -1.228
    month_oct              -0.4541      0.338     -1.345      0.179      -1.116       0.208
    month_sep              -0.4072      0.339     -1.200      0.230      -1.072       0.258
    poutcome_failure       -0.3341      0.089     -3.734      0.000      -0.509      -0.159
    poutcome_success        1.9193      0.098     19.516      0.000       1.727       2.112
    poutcome_unknown       -0.1566      0.188     -0.831      0.406      -0.526       0.213
    =======================================================================================



```python
y_pred = logit_model.predict(X_test)
y_predbin= pd.DataFrame(y_pred).values
y_predbin= sklearn.preprocessing.binarize(y_predbin, threshold=0.5, copy=True)
```


```python
cnf_matrix = metrics.confusion_matrix(y_test, y_predbin)

heatMap(pd.DataFrame(cnf_matrix))
print("Accuracy:",metrics.accuracy_score(y_test, y_predbin))
print("Precision:",metrics.precision_score(y_test, y_predbin))
print("Recall:",metrics.recall_score(y_test, y_predbin))
```


![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_42_0.png)


    Accuracy: 0.8922872340425532
    Precision: 0.5923566878980892
    Recall: 0.18058252427184465



```python
cnf_matrix
```




    array([[7866,  128],
           [ 844,  186]])




```python
print('predicted class breakdown',pd.DataFrame(y_predbin)[0].value_counts()/(len(pd.DataFrame(y_predbin)[0])+1),sep=
     '\n')
plt.bar(['0','1'], pd.DataFrame(y_predbin)[0].value_counts(), color = "blue")
plt.title('Predicted Classes')
```

    predicted class breakdown
    0.00000000   0.96509695
    1.00000000   0.03479224
    Name: 0, dtype: float64





    Text(0.5, 1.0, 'Predicted Classes')




![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_44_2.png)



```python
y_pred_proba = y_pred
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="auc="+str(round(auc,2)))
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.title("ROC Curve")
plt.legend(loc=4)
plt.show()
```


![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_45_0.png)


## Model 2 - Using SMOTE and Dropping Insignificant Variables
The method that we used to treat the imbalanced dataset was the SMOTE (Synthetic Minority Oversampling Technique). <br>
<i>"What it does is, it creates synthetic (not duplicate) samples of the minority class. Hence making the minority class equal to the majority class. SMOTE does this by selecting similar records and altering that record one column at a time by a random amount within the difference to the neighbouring records." </i> (Rahim S., 2018)

Next, we also dropped some variables below:
- Date was excluded in the model. Generally speacking, clients decide to subscribe or not to subscribe a bank’s term deposit are not reference to date. 
- **pdays** variable was also excluded since we knew it provided the same information as **previous**.
- Dropping **default** variable as well because it's insignificant

The accuracy on this model dropped significantly to ~71%. However, the recall had improved so much to ~62.8% vs ~18% on the first model. We believed that this model was more representative since it had a better distribution of prediction between the 2 classes (yes/no).


```python
train1 = train.drop(columns=['pdays','default','day','month_apr', 'month_aug', 'month_dec', 'month_feb', 'month_jan',
       'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
       'month_oct', 'month_sep'])

```


```python
test1 = test.drop(columns=['pdays','default','day','month_apr', 'month_aug', 'month_dec', 'month_feb', 'month_jan',
       'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
       'month_oct', 'month_sep'])

X_test1 = test1.loc[:,test1.columns != 'deposit'].values
y_test1 = test1.loc[:, test1.columns == 'deposit'].values

```


```python
train1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>campaign</th>
      <th>previous</th>
      <th>deposit</th>
      <th>contact_telephone</th>
      <th>contact_unknown</th>
      <th>education_primary</th>
      <th>...</th>
      <th>job_self-employed</th>
      <th>job_services</th>
      <th>job_student</th>
      <th>job_technician</th>
      <th>job_unemployed</th>
      <th>marital_married</th>
      <th>marital_single</th>
      <th>poutcome_failure</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>2143</td>
      <td>1</td>
      <td>0</td>
      <td>1.00000000</td>
      <td>0.00000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>1.00000000</td>
      <td>0.00000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1.00000000</td>
      <td>0.00000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>1506</td>
      <td>1</td>
      <td>0</td>
      <td>1.00000000</td>
      <td>0.00000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.00000000</td>
      <td>0.00000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
columns=train1.loc[:,train1.columns != 'deposit'].columns

X= train1.loc[:,train1.columns != 'deposit'].values
y = train1.loc[:, train1.columns == 'deposit'].values


smt = SMOTE()

os_data_X,os_data_y=smt.fit_sample(X, y)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
smoted_train=pd.concat([os_data_X,os_data_y],axis=1)

# we can Check the numbers of our data
print("Length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no deposit data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of deposit data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
```

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/utils/validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)


    Length of oversampled data is  63856
    Number of no subscription in oversampled data 31928
    Number of subscription 31928
    Proportion of no deposit data in oversampled data is  0.5
    Proportion of deposit data in oversampled data is  0.5



```python
os_data_y['y'].value_counts()
```




    1    31928
    0    31928
    Name: y, dtype: int64


```python
logit_model1 = sm.Logit(os_data_y, os_data_X).fit()
print(logit_model1.summary())
```

    Optimization terminated successfully.
             Current function value: 0.568418
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                63856
    Model:                          Logit   Df Residuals:                    63829
    Method:                           MLE   Df Model:                           26
    Date:                Sat, 30 Nov 2019   Pseudo R-squ.:                  0.1799
    Time:                        23:26:45   Log-Likelihood:                -36297.
    converged:                       True   LL-Null:                       -44262.
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    =======================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    age                     0.0010      0.001      0.920      0.357      -0.001       0.003
    balance              2.739e-05   3.08e-06      8.882      0.000    2.13e-05    3.34e-05
    housing                -0.6192      0.021    -30.090      0.000      -0.660      -0.579
    loan                   -0.5089      0.029    -17.669      0.000      -0.565      -0.452
    campaign               -0.1742      0.006    -26.912      0.000      -0.187      -0.162
    previous                0.3661      0.041      8.857      0.000       0.285       0.447
    contact_telephone      -0.2008      0.040     -5.076      0.000      -0.278      -0.123
    contact_unknown        -1.0184      0.026    -39.113      0.000      -1.069      -0.967
    education_primary      -0.1984      0.055     -3.584      0.000      -0.307      -0.090
    education_secondary    -0.0314      0.049     -0.635      0.525      -0.128       0.066
    education_tertiary      0.2294      0.052      4.387      0.000       0.127       0.332
    job_admin.              0.6609      0.092      7.214      0.000       0.481       0.840
    job_blue-collar         0.4505      0.090      4.997      0.000       0.274       0.627
    job_entrepreneur        0.3466      0.104      3.338      0.001       0.143       0.550
    job_housemaid           0.0829      0.108      0.767      0.443      -0.129       0.295
    job_management          0.4444      0.091      4.868      0.000       0.265       0.623
    job_retired             0.9909      0.103      9.650      0.000       0.790       1.192
    job_self-employed       0.4771      0.101      4.703      0.000       0.278       0.676
    job_services            0.4797      0.093      5.157      0.000       0.297       0.662
    job_student             0.9870      0.102      9.636      0.000       0.786       1.188
    job_technician          0.3958      0.091      4.370      0.000       0.218       0.573
    job_unemployed          0.6659      0.102      6.531      0.000       0.466       0.866
    marital_married        -0.1836      0.031     -6.002      0.000      -0.244      -0.124
    marital_single          0.2135      0.034      6.204      0.000       0.146       0.281
    poutcome_failure       -0.4828      0.049     -9.817      0.000      -0.579      -0.386
    poutcome_success        3.1788      0.083     38.215      0.000       3.016       3.342
    poutcome_unknown        0.1025      0.076      1.342      0.179      -0.047       0.252
    =======================================================================================

![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_54_0.png)


    Accuracy: 0.7116578014184397
    Precision: 0.22575017445917656
    Recall: 0.6281553398058253



```python
pd.DataFrame(y_predbin1)[0].value_counts()/(len(pd.DataFrame(y_predbin1)[0])+1)
```




    0.00000000   0.68232687
    1.00000000   0.31756233
    Name: 0, dtype: float64


    Text(0.5, 1.0, 'Predicted Classes')




![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_56_1.png)

![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_57_0.png)


# Model 3
- Based on model 2 result on variable importance, we dropped variables that had p-value > 0.05
- The result indicated that accuracy improved slightly with less than 1%. Therefore we chose Model 3.



```python
train1 = train.drop(columns=['pdays','default','day','month_apr', 'month_aug', 'month_dec', 'month_feb', 'month_jan',
       'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
       'month_oct', 'month_sep','poutcome_unknown','job_housemaid'])
test1 = test.drop(columns=['pdays','default','day','month_apr', 'month_aug', 'month_dec', 'month_feb', 'month_jan',
       'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
       'month_oct', 'month_sep','poutcome_unknown','job_housemaid'])
```

    Optimization terminated successfully.
             Current function value: 0.567444
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                63856
    Model:                          Logit   Df Residuals:                    63831
    Method:                           MLE   Df Model:                           24
    Date:                Sat, 30 Nov 2019   Pseudo R-squ.:                  0.1814
    Time:                        23:35:48   Log-Likelihood:                -36235.
    converged:                       True   LL-Null:                       -44262.
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    =======================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    age                     0.0008      0.001      0.893      0.372      -0.001       0.003
    balance              2.761e-05   3.07e-06      9.007      0.000    2.16e-05    3.36e-05
    housing                -0.6224      0.021    -30.343      0.000      -0.663      -0.582
    loan                   -0.4924      0.029    -17.092      0.000      -0.549      -0.436
    campaign               -0.1678      0.006    -26.260      0.000      -0.180      -0.155
    previous                0.3312      0.023     14.528      0.000       0.286       0.376
    contact_telephone      -0.1974      0.040     -4.976      0.000      -0.275      -0.120
    contact_unknown        -1.0073      0.026    -38.772      0.000      -1.058      -0.956
    education_primary      -0.1047      0.050     -2.095      0.036      -0.203      -0.007
    education_secondary     0.0258      0.044      0.584      0.559      -0.061       0.112
    education_tertiary      0.2607      0.047      5.524      0.000       0.168       0.353
    job_admin.              0.6412      0.054     11.795      0.000       0.535       0.748
    job_blue-collar         0.4388      0.051      8.538      0.000       0.338       0.540
    job_entrepreneur        0.3063      0.073      4.200      0.000       0.163       0.449
    job_management          0.4542      0.054      8.429      0.000       0.349       0.560
    job_retired             0.9639      0.067     14.404      0.000       0.833       1.095
    job_self-employed       0.4232      0.070      6.060      0.000       0.286       0.560
    job_services            0.4812      0.057      8.499      0.000       0.370       0.592
    job_student             0.9895      0.074     13.356      0.000       0.844       1.135
    job_technician          0.3635      0.053      6.911      0.000       0.260       0.467
    job_unemployed          0.6449      0.070      9.177      0.000       0.507       0.783
    marital_married        -0.1517      0.030     -5.117      0.000      -0.210      -0.094
    marital_single          0.2907      0.032      8.995      0.000       0.227       0.354
    poutcome_failure       -0.4990      0.046    -10.792      0.000      -0.590      -0.408
    poutcome_success        3.1733      0.081     39.050      0.000       3.014       3.333
    =======================================================================================


![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_63_0.png)


    Accuracy: 0.7138741134751773
    Precision: 0.22733661278988054
    Recall: 0.6281553398058253


![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_64_1.png)


![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_65_0.png)


# Model 4
- Since we knew there were 2 types of customers in the dataset, one who had **been contacted before** and one who had **not been contacted before**, we wanted to create separate models for these customers since we believed that personalization is key.
- A more similar customers can give you more accurate prediction as well.
- In this case, the variable that we used to separate the model was **pdays** since any value above **-1** indicated customers who had been previously contacted for the bank marketing campaign.
- The result was favorable. using the same exact method as **model 3**, we were able to achieve accuracy of ~77%.


```python
train3 = train[train.pdays != -1]
test3 = test[test.pdays != -1]

train1 = train3.drop(columns=['pdays','default','day','month_apr', 'month_aug', 'month_dec', 'month_feb', 'month_jan',
       'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
       'month_oct', 'month_sep','poutcome_unknown','job_housemaid'])
test1 = test3.drop(columns=['pdays','default','day','month_apr', 'month_aug', 'month_dec', 'month_feb', 'month_jan',
       'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
       'month_oct', 'month_sep','poutcome_unknown','job_housemaid'])

X_test1 = test1.loc[:,test1.columns != 'deposit'].values
y_test1 = test1.loc[:, test1.columns == 'deposit'].values
```

    Length of oversampled data is  10122
    Number of no subscription in oversampled data 5061
    Number of subscription 5061
    Proportion of no deposit data in oversampled data is  0.5
    Proportion of deposit data in oversampled data is  0.5
    1    5061
    0    5061
    Name: y, dtype: int64



    Optimization terminated successfully.
             Current function value: 0.502125
             Iterations 6
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                10122
    Model:                          Logit   Df Residuals:                    10097
    Method:                           MLE   Df Model:                           24
    Date:                Sat, 30 Nov 2019   Pseudo R-squ.:                  0.2756
    Time:                        23:46:37   Log-Likelihood:                -5082.5
    converged:                       True   LL-Null:                       -7016.0
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    =======================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    age                    -0.0001      0.003     -0.046      0.963      -0.005       0.005
    balance              1.595e-05   8.09e-06      1.972      0.049       1e-07    3.18e-05
    housing                -0.9681      0.056    -17.432      0.000      -1.077      -0.859
    loan                   -0.5818      0.087     -6.723      0.000      -0.751      -0.412
    campaign               -0.1535      0.021     -7.246      0.000      -0.195      -0.112
    previous                0.3172      0.054      5.844      0.000       0.211       0.424
    contact_telephone      -0.2155      0.102     -2.104      0.035      -0.416      -0.015
    contact_unknown        -0.5576      0.269     -2.076      0.038      -1.084      -0.031
    education_primary      -0.6053      0.143     -4.242      0.000      -0.885      -0.326
    education_secondary    -0.3852      0.121     -3.184      0.001      -0.622      -0.148
    education_tertiary     -0.1405      0.126     -1.112      0.266      -0.388       0.107
    job_admin.              0.1873      0.153      1.227      0.220      -0.112       0.486
    job_blue-collar        -0.4044      0.150     -2.700      0.007      -0.698      -0.111
    job_entrepreneur       -0.5751      0.228     -2.519      0.012      -1.023      -0.128
    job_management          0.0905      0.152      0.596      0.551      -0.207       0.388
    job_retired             0.2913      0.185      1.573      0.116      -0.072       0.654
    job_self-employed      -0.1098      0.193     -0.570      0.569      -0.488       0.268
    job_services           -0.0418      0.162     -0.257      0.797      -0.360       0.277
    job_student             0.3518      0.181      1.940      0.052      -0.004       0.707
    job_technician         -0.1949      0.151     -1.287      0.198      -0.492       0.102
    job_unemployed          0.5237      0.199      2.628      0.009       0.133       0.914
    marital_married         0.1811      0.084      2.164      0.030       0.017       0.345
    marital_single          0.2831      0.091      3.117      0.002       0.105       0.461
    poutcome_failure       -0.3130      0.062     -5.015      0.000      -0.435      -0.191
    poutcome_success        2.3982      0.082     29.264      0.000       2.238       2.559
    =======================================================================================


![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_71_0.png)


    Accuracy: 0.7750301568154403
    Precision: 0.49353448275862066
    Recall: 0.6239782016348774



![png](https://github.com/jsiwu94/jsiwu94.github.io/blob/master/bank_prediction/output_72_0.png)


## Conclusion and Key Takeaways
- Accuracy is not the only measure of how good the model is
- We should assess the quality of the model and improvise
- Ensure that the data is not imbalanced
- Create segments of similar customers using affinity clustering or k-means 
- Create separate model for each segment to capture different customer behaviour 
- There is no such thing as an average customer because personalisation is the key!

<br>


Analysis and Report Written by : Jennifer Siwu <br>
Reference for SMOTE :
https://medium.com/@saeedAR/smote-and-near-miss-in-python-machine-learning-in-imbalanced-datasets-b7976d9a7a79
