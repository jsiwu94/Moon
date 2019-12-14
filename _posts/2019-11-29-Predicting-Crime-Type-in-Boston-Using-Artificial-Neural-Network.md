---
layout: post
title: "Predicting Crime Type in Boston Using Artificial Neural Network"
date: 2019-11-29
excerpt: "Using Artificial Neural Network to Prevent Crimes"
tags: [Machine Learning, Artificial Neural Networks, Keras, Tensorflow, Crime, Python]
comments: true
---


<script src="https://gist.github.com/jsiwu94/4b674c99cfecc24ecac732bb0a74abbb.js"></script>
**Plot of High Crime Points in Boston (2015 to 2019)**


This project is an extension of my initial Keras & Tensorflow practice in a Kaggle Competition of Predicting Crime in SF. Using a similar approach in ANN, I was ranked within the top 30% in the competition. With that practice, I wanted to take the analysis another step further by trying to apply it to a different state. The detail of my competition result can be found [here](https://www.kaggle.com/jenniferwu94/using-ann-in-tensorflow-to-predict-sf-crime).

## Purpose
We hear stories about crimes happening in Boston everyday. Often times, we only found out about it after it happened.
In this project, I would like to utilize machine learning model to proactively predict certain types of crime that will occur based on a given date, time, day of week, and location. The main target user, The Police Authority, can use the output of this model to proactively lay out their plans to prevent it from happening.

## Project Overview
1. Data Collection and Cleaning <br>
Getting the Boston crime incident data from the official Boston Government website https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system.

2. EDA and Insight Generation <br>
Conducted exploratory data analysis to generate pattern or trends that will be used for our model building.

3. Model Building <br>
Conducted feature engineering based on the EDA result. Fit the training data into Artificial Neural Network model in Keras and Tensorflow. 

4. Assesing Model Performance <br>
Utilizing the Accuracy and Cross Entropy Loss to assess model performance. Conducted the backward elimination methodology for feature selection to improve model performance.


## Part 1
### Data Cleaning 

The dataset has 17 variables and over 470K observations. It contains information regarding crime incident reports from June 2015 to October 2019 with the format below.

        RangeIndex: 435655 entries, 0 to 435654
        Data columns (total 17 columns):
        INCIDENT_NUMBER        435655 non-null object
        OFFENSE_CODE           435655 non-null int64
        OFFENSE_CODE_GROUP     426840 non-null object
        OFFENSE_DESCRIPTION    435655 non-null object
        DISTRICT               433439 non-null object
        REPORTING_AREA         435655 non-null object
        SHOOTING               10562 non-null object
        OCCURRED_ON_DATE       435655 non-null object
        YEAR                   435655 non-null int64
        MONTH                  435655 non-null int64
        DAY_OF_WEEK            435655 non-null object
        HOUR                   435655 non-null int64
        UCR_PART               426730 non-null object
        STREET                 422330 non-null object
        Lat                    399636 non-null float64
        Long                   399636 non-null float64
        Location               435655 non-null object


For data cleaning, the first step I took was by checking for null and Nan's. Through this checked, I removed some observations with null or invalid Latitude and Longitude. This equates to ~3% of the entire observations in the dataset. There were also some observations within the same variable with different data types. For these observations, I altered the data type to match the majority within that variable.

## Part 2
### EDA

I started the EDA by looking at the Number of Crime Frequency pattern across different days of week. Based on the graph below, Friday has been the highest crime day and Sunday has been the lowest for all five years. Based on this, I will make this variable categorical in the model.

<img width="788" alt="DOW" src="https://user-images.githubusercontent.com/54050356/70124675-2bf2ee80-162a-11ea-8444-ea60eba3664f.png">

Based on the below, the Peak Crime hour were at 12pm and between 4pm to 7pm. This trend was seen in all five years.

<img width="813" alt="hr" src="https://user-images.githubusercontent.com/54050356/70124676-2bf2ee80-162a-11ea-8110-cd2c7ca4400e.png">

Below graph shows the top 70% **offense code group** or **crime type** based on the number of times they occured through out the entire 5 years. This is the dependent variable that we will predict in the model.

<img width="643" alt="top2" src="https://user-images.githubusercontent.com/54050356/70124768-5775d900-162a-11ea-82af-72c87081515a.png">

Looking at the crime density, the northern areas seemed to have more crime than the southern Boston areas. Below graph shows the number of crime occured grouped by the district.

<img width="718" alt="district" src="https://user-images.githubusercontent.com/54050356/70124673-2bf2ee80-162a-11ea-8289-c8b03c407d5c.png">

The figures below depicts the top 70% crime offense code group or crime type. It is clearly shown that some crimes happened more often in specific areas while some other occured almost evenly all throughout Boston.
<img width="657" alt="scatterplt" src="https://user-images.githubusercontent.com/54050356/70124812-6a88a900-162a-11ea-9234-fcc750e46aae.png">

## Part 3
### Building The Model

In this section I will talk about the detailed steps I took in building the prediction.
1. Target User
   Police Authority
2. Prediction Method 
   Using Artificial Neural Network in Keras and Tensorflow, I will predict which type of **"Offense Code Group"** or
   **Crime Type** will happen in a specific location and time.
3. Outlier Treatment and Feature Engineering
   Based on the EDA result, I am removing all the observations for the bottom 30% crime incidents. This is to improve my model
   performance and to negate outliers.
   Additionally, I created 2 new variables, to group the months with **season** variable and to grouped the address by
   the **top 10% highest crime addresses**.
4. Feature Selection and Dimension Reduction
   Since the dataset did not have many variables to begin with, I decided to include all of the independent variables to train
   my first model and then conduct the backward elimination method for dimension reduction.
   
## Top 70% Crime Offense Code Group or Crime Type

    {'OFFENSE_CODE_GROUP': {'Drug Violation': 0,
      'Investigate Person': 1,
      'Investigate Property': 2,
      'Larceny': 3,
      'Larceny From Motor Vehicle': 4,
      'Medical Assistance': 5,
      'Motor Vehicle Accident Response': 6,
      'Other': 7,
      'Simple Assault': 8,
      'Towed': 9,
      'Vandalism': 10,
      'Verbal Disputes': 11}}

## Feature Selection

There was already a limited amount of features provided in this dataset. Therefore, I will only drop some features that do not make sense for a more realistic model. Given that, the features that I will drop are :
1. **YEAR**. Using this variable does not make sense in predicting the crime code because it does not matter what year the crime happened in. Instead, we care more about the seasonal, monthly, weekly, or daily, pattern in within each of the year when predicting the future.

2. **SHOOTING**. As mentioned above, the SHOOTING variable tells us if shooting was involved when the crime occured. Therefore, it is impossible to get this data before the crime incident happens. 

## Feature Engineering
To prevent information loss and improve the prediction, I created 2 variables:
1. **Season** to group the months variable.
2. **Top 10% high crime addressess** to group the address variable.
     

## Preparing the Data Format for The Model
```python
##Categorizing the season column using Dummy Vars : the reason is because there is no Hierarchy..
#meaning that, "Fall IS NOT Higher or Better than Summer"

def data_prep(df_clean):
    
    #date time split 2019-10-13 09:28:24
    def parse_time(x):
        DD=datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
        time=DD.hour 
        day=DD.day
        month=DD.month
        year=DD.year
        mins=DD.minute
        return time,day,month,year,mins
    
    parsed = np.array([parse_time(x) for x in df_clean.occured_on_dttm])
    
    df_clean['Dates'] = pd.to_datetime(df_clean['occured_on_dttm'])
    df_clean['WeekOfYear'] = df_clean['Dates'].dt.weekofyear
    df_clean['Hour'] = parsed[:,[0]]
    df_clean['day'] = parsed[:,[1]]
    df_clean['month'] = parsed[:,[2]]
    df_clean['year'] = parsed[:,[3]]
    df_clean['mins'] = parsed[:,[4]]
    
    
        
    #adding season variable
    def get_season(x):
        if x in [5, 6, 7]:
            r = 'summer'
        elif x in [8, 9, 10]:
            r = 'fall'
        elif x in [11, 12, 1]:
            r = 'winter'
        elif x in [2, 3, 4]:
            r = 'spring'
        return r
    
    df_clean['season'] = [get_season(i) for i in crime.MONTH]
        
        
    #grouping street       
    streetcount = df_clean.groupby(['STREET'], as_index=False).agg({"INCIDENT_NUMBER":"count"}).sort_values(by='INCIDENT_NUMBER', ascending=False).head(8)
    streetcount['odds'] = [round(i/(len(df_clean.STREET)+1),3) for i in streetcount.INCIDENT_NUMBER]
    streetcount.drop(['INCIDENT_NUMBER'],axis=1,inplace=True)
    df_clean = pd.merge(left=df_clean,right=streetcount,how='left',left_on='STREET',right_on='STREET')
    
    
    #dummy variables
    df_clean_onehot = pd.get_dummies(df_clean, columns=['season'], prefix = [''])
    s = (len(list(df_clean_onehot.columns))-len(df_clean.season.value_counts()))
    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,s:]], axis=1)

    ##Categorizing the DayOFWeek column using Dummy Vars 
    df_clean_onehot = pd.get_dummies(df_clean, columns=['DAY_OF_WEEK'], prefix = [''])
    l = (len(list(df_clean_onehot.columns))-len(df_clean.DAY_OF_WEEK.value_counts()))
    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,l:]],axis=1)

    ##Categorizing the MONTH column using Dummy Vars : the reason is because there is no Hierarchy..
    #meaning that, "FEB IS NOT Higher or Better than JAN"
    #This insight was shown from the EDA result (forecasting data with trend might be a different case)

    df_clean_onehot = pd.get_dummies(df_clean, columns=['month'], prefix = ['month'])
    n = (len(list(df_clean_onehot.columns))-len(df_clean.MONTH.value_counts()))
    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,n:]],axis=1)

    ##Categorizing the District column using Dummy Vars 

    df_clean_onehot = pd.get_dummies(df_clean, columns=['DISTRICT'], prefix = [''])
    o = (len(list(df_clean_onehot.columns))-len(df_clean.DISTRICT.value_counts()))
    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,o:]],axis=1)

    ##changing the Output Variables to integer
    labels = df_clean['OFFENSE_CODE_GROUP'].astype('category').cat.categories.tolist()
    replace_with_int = {'OFFENSE_CODE_GROUP' : {k: v for k,v in zip(labels,list(range(0,len(labels))))}}
    df_clean.replace(replace_with_int, inplace=True)

    #Normalizing the columns
    def norm_func(i):
        r = (i-min(i))/(max(i)-min(i))
        return(r)

    df_clean['normHour']=norm_func(df_clean.HOUR)
    df_clean['normmins']=norm_func(df_clean.mins)
    df_clean['normdate_day']=norm_func(df_clean.day)
    df_clean['normLat']=norm_func(df_clean.Lat)
    df_clean['normLong']=norm_func(df_clean.Long)
    df_clean['normmonth']=norm_func(df_clean.month)
    df_clean['normyear']=norm_func(df_clean.year)
    df_clean['normWeekOfYear']=norm_func(df_clean.WeekOfYear)



    ##removing the unused columns
    df_clean.drop(columns = [ 'Unnamed: 0', 'INCIDENT_NUMBER', 'OFFENSE_CODE',
       'OFFENSE_DESCRIPTION', 'DISTRICT', 'REPORTING_AREA', 'SHOOTING', 'YEAR',
       'MONTH', 'DAY_OF_WEEK', 'HOUR', 'UCR_PART', 'STREET', 'Lat', 'Long',
       'Location_lat', 'Location_long', 'date_occured', 'occured_on_dttm','Dates',
        'WeekOfYear','Hour','day','month','year','mins','season'], axis = 1,inplace=True)
    return(df_clean)
```

## Splitting the Train & Test Data for Model

Using the holdout method, I randomly split the dataset into 80% for training, and 20% for testing.

```python
split = np.random.rand(len(crime_clean)) < 0.8
train = crime_clean[split]
test = crime_clean[~split]
```
```python
print('entire dataset :',len(crime_clean.OFFENSE_CODE_GROUP.value_counts().index), 'offense codes')
print('train dataset :',len(crime_clean.OFFENSE_CODE_GROUP.value_counts().index), 'offense codes')
print('test dataset :',len(crime_clean.OFFENSE_CODE_GROUP.value_counts().index), 'offense codes')
```

    entire dataset : 12 offense codes
    train dataset : 12 offense codes
    test dataset : 12 offense codes

Below shows the format in which I will fit the data into my model.
```python
train1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 217154 entries, 0 to 217153
    Data columns (total 43 columns):
    OFFENSE_CODE_GROUP    217154 non-null int64
    _fall                 217154 non-null uint8
    _spring               217154 non-null uint8
    _summer               217154 non-null uint8
    _winter               217154 non-null uint8
    _Friday               217154 non-null uint8
    _Monday               217154 non-null uint8
    _Saturday             217154 non-null uint8
    _Sunday               217154 non-null uint8
    _Thursday             217154 non-null uint8
    _Tuesday              217154 non-null uint8
    _Wednesday            217154 non-null uint8
    month_1               217154 non-null uint8
    month_2               217154 non-null uint8
    month_3               217154 non-null uint8
    month_4               217154 non-null uint8
    month_5               217154 non-null uint8
    month_6               217154 non-null uint8
    month_7               217154 non-null uint8
    month_8               217154 non-null uint8
    month_9               217154 non-null uint8
    month_10              217154 non-null uint8
    month_11              217154 non-null uint8
    month_12              217154 non-null uint8
    _A1                   217154 non-null uint8
    _A15                  217154 non-null uint8
    _A7                   217154 non-null uint8
    _B2                   217154 non-null uint8
    _B3                   217154 non-null uint8
    _C11                  217154 non-null uint8
    _C6                   217154 non-null uint8
    _D14                  217154 non-null uint8
    _D4                   217154 non-null uint8
    _E13                  217154 non-null uint8
    _E18                  217154 non-null uint8
    _E5                   217154 non-null uint8
    normHour              217154 non-null float64
    normmins              217154 non-null float64
    normdate_day          217154 non-null float64
    normLat               217154 non-null float64
    normLong              217154 non-null float64
    normmonth             217154 non-null float64
    normWeekOfYear        217154 non-null float64
    dtypes: float64(7), int64(1), uint8(35)
    memory usage: 20.5 MB


## Training the neural network model
### Final Model - Dropped street and Season grouping variables as well as year variables, 2 layers, 20 epoch

```python
model1 = Sequential()
model1.add(Dense(12, input_shape=(38,)))
model1.add(Dense(128, activation='relu', input_dim=512))
model1.add(Dense(90, activation='relu', input_dim=128))
model1.add(Dense(12, activation='softmax', input_dim=90))
model1.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy']
               , optimizer='adam')

# The fit() method - trains the model
model1.fit(Train1Data, Train1Labels, nb_epoch=20)
```

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/ipykernel_launcher.py:10: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      # Remove the CWD from sys.path while we load stuff.


    Epoch 1/20
    217154/217154 [==============================] - 8s 36us/step - loss: 2.3550 - accuracy: 0.1838
    Epoch 2/20
    217154/217154 [==============================] - 9s 40us/step - loss: 2.3194 - accuracy: 0.1915
    Epoch 3/20
    217154/217154 [==============================] - 8s 37us/step - loss: 2.3096 - accuracy: 0.1955 1s - loss: 2.3098  - ETA: 0s - loss: 2.3096 -  - ETA: 0s - loss: 2.309
    Epoch 4/20
    217154/217154 [==============================] - 8s 35us/step - loss: 2.3029 - accuracy: 0.1981
    Epoch 5/20
    217154/217154 [==============================] - 8s 38us/step - loss: 2.2967 - accuracy: 0.2014E
    Epoch 6/20
    217154/217154 [==============================] - 8s 36us/step - loss: 2.2916 - accuracy: 0.2031
    Epoch 7/20
    217154/217154 [==============================] - 9s 40us/step - loss: 2.2867 - accuracy: 0.2061
    Epoch 8/20
    217154/217154 [==============================] - 9s 39us/step - loss: 2.2818 - accuracy: 0.2090
    Epoch 9/20
    217154/217154 [==============================] - 8s 35us/step - loss: 2.2770 - accuracy: 0.2112
    Epoch 10/20
    217154/217154 [==============================] - 7s 33us/step - loss: 2.2732 - accuracy: 0.2128
    Epoch 11/20
    217154/217154 [==============================] - 7s 33us/step - loss: 2.2688 - accuracy: 0.2147
    Epoch 12/20
    217154/217154 [==============================] - 7s 33us/step - loss: 2.2652 - accuracy: 0.2154 5s - loss: 2.2637 - accuracy - E - ETA: 0s - loss: 2.2655 - accuracy:  - ETA: 0s - loss: 2.2655 - accu - ETA: 0s - loss: 2.2654 - accu
    Epoch 13/20
    217154/217154 [==============================] - 8s 37us/step - loss: 2.2612 - accuracy: 0.2178
    Epoch 14/20
    217154/217154 [==============================] - 8s 37us/step - loss: 2.2572 - accuracy: 0.2197
    Epoch 15/20
    217154/217154 [==============================] - 8s 35us/step - loss: 2.2540 - accuracy: 0.2207
    Epoch 16/20
    217154/217154 [==============================] - 8s 38us/step - loss: 2.2501 - accuracy: 0.2223
    Epoch 17/20
    217154/217154 [==============================] - 7s 34us/step - loss: 2.2466 - accuracy: 0.2238
    Epoch 18/20
    217154/217154 [==============================] - 7s 34us/step - loss: 2.2437 - accuracy: 0.2245 0s - los
    Epoch 19/20
    217154/217154 [==============================] - 7s 33us/step - loss: 2.2407 - accuracy: 0.2259
    Epoch 20/20
    217154/217154 [==============================] - 7s 34us/step - loss: 2.2386 - accuracy: 0.2265 0s - loss: 2.2389 - accura





    <keras.callbacks.callbacks.History at 0x14549a650>

## Evaluating on Test Data

    54469/54469 [==============================] - 1s 15us/step
    Test Loss:
    2.2557787604998083
    Test Accuracy:
    0.22216306626796722

## Part 4
### Model Prediction Result
The baseline probability was ~8.3% for each class since there were 12 classes in the dependent variable.
After running 3 different models with backward eliminiatino process, the final model was built based on removing the season, year, and top addresses variables. Based on comparison with the previous models, this has proved to have the best result without overfitting.

To evaluate the model prediction, ROC does not provide a well enough overview on a multiclass classification. Therefore, I decided to visualize it with a confusion matrix instead. The result on the Confusion Matrix showed that our prediction of crime distribution followed the pattern of the actual crime distribution with Motor Vehicle Accident being the highest.

The result for accuracy was ~22.2% and loss was 2.256, both on Test data. The winner of a similar prediction for a Kaggle Competition (Predicting Crime in SF) received a loss of 2.11 and accuracy of 25% predicting 24 classes of crimes using dataset with ~30 years of observations (roughly 7x this project's dataset). Therefore, this prediction is considerably good given the limited amount of data.

### Final Thought and Conclusion
Although this model gives a pretty good estimate on crime prediction, it is only a simple prove of concept (POC). There are definitely a lot of improvements can be made in the future. As time goes by, adding more data would definitely improve the prediction performance. When more data have been collected, a more advanced deep learning method such as RNN (Recurrent Neural Network) can be used for a better prediction. 

![image](https://user-images.githubusercontent.com/54050356/70853266-76f9d680-1e60-11ea-8be8-eeac5482542f.png)


### Model Raw Output
```python
predictions = model1.predict_proba(Test1Data)
predictiondata=pd.DataFrame(data=predictions) 
predictiondata.drop([0], axis =1, inplace = True) #removing the 0th class auto created by TF
predictiondata.rename(columns = {i:j for i,j in zip(predictiondata.columns,labels)}, inplace = True)
predictiondata['MaxProb'] = [np.max(i) for i in predictions]
predictiondata['MaxProbLabel'] = [labels[np.argmax(i)] for i in predictions]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug Violation</th>
      <th>Investigate Person</th>
      <th>Investigate Property</th>
      <th>Larceny</th>
      <th>Larceny From Motor Vehicle</th>
      <th>Medical Assistance</th>
      <th>Motor Vehicle Accident Response</th>
      <th>Other</th>
      <th>Simple Assault</th>
      <th>Towed</th>
      <th>Vandalism</th>
      <th>MaxProb</th>
      <th>MaxProbLabel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9860</th>
      <td>0.010175</td>
      <td>0.000684</td>
      <td>0.964066</td>
      <td>0.005465</td>
      <td>0.003356</td>
      <td>0.008226</td>
      <td>0.002629</td>
      <td>0.002399</td>
      <td>0.000011</td>
      <td>0.001224</td>
      <td>0.000005</td>
      <td>0.964066</td>
      <td>Larceny</td>
    </tr>
    <tr>
      <th>13580</th>
      <td>0.033826</td>
      <td>0.004403</td>
      <td>0.001145</td>
      <td>0.000043</td>
      <td>0.024009</td>
      <td>0.001463</td>
      <td>0.001869</td>
      <td>0.000129</td>
      <td>0.000039</td>
      <td>0.000456</td>
      <td>0.000544</td>
      <td>0.932073</td>
      <td>Drug Violation</td>
    </tr>
    <tr>
      <th>37795</th>
      <td>0.026715</td>
      <td>0.001182</td>
      <td>0.908468</td>
      <td>0.008588</td>
      <td>0.009570</td>
      <td>0.019438</td>
      <td>0.008087</td>
      <td>0.008032</td>
      <td>0.000114</td>
      <td>0.002578</td>
      <td>0.000042</td>
      <td>0.908468</td>
      <td>Larceny</td>
    </tr>
    <tr>
      <th>24645</th>
      <td>0.024893</td>
      <td>0.001246</td>
      <td>0.901784</td>
      <td>0.012499</td>
      <td>0.007624</td>
      <td>0.018414</td>
      <td>0.006875</td>
      <td>0.011354</td>
      <td>0.000050</td>
      <td>0.004500</td>
      <td>0.000047</td>
      <td>0.901784</td>
      <td>Larceny</td>
    </tr>
    <tr>
      <th>26403</th>
      <td>0.050741</td>
      <td>0.015595</td>
      <td>0.002573</td>
      <td>0.000145</td>
      <td>0.033954</td>
      <td>0.004519</td>
      <td>0.005075</td>
      <td>0.000466</td>
      <td>0.000145</td>
      <td>0.000760</td>
      <td>0.001430</td>
      <td>0.884597</td>
      <td>Drug Violation</td>
    </tr>
  </tbody>
</table>
</div>

### Final Prediction Output
<img width="698" alt="output" src="https://user-images.githubusercontent.com/54050356/70124678-2c8b8500-162a-11ea-89c0-8434ce1b4af3.png">

<br>
Analysis and Report Written by : Jennifer Siwu <br>
