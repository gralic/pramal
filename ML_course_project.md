# Practical Machine Learning course project
========================================================
**The goal of the project is to predict the manner in which 6 persons performed a physical exercise,
namely barbell lifts. In the training set, this is the "classe" variable. On one hand, there are 159
other variables, but on the other hand, many of them are always 'NA'. So first of all we discard
the variables that won't help achieve the project goal simply because they are all 'NA' in the test set:**

```r
pmlt = read.csv("pml_training.csv")
pmle = read.csv("pml_testing.csv")
print(c(dim(pmlt), dim(pmle)))
```

```
## [1] 19622   160    20   160
```

```r
for (i in dim(pmle)[2]:1) {
    if (sum(is.na(pmle[, i])) == dim(pmle)[1]) {
        pmle[, i] <- NULL
        pmlt[, i] <- NULL
    }
}
print(c(dim(pmlt), dim(pmle)))
```

```
## [1] 19622    60    20    60
```

**After looking at names of the remaining 60 variables, and first few values,
we can see that 5 out of the first 6 variables are not likely to help predict correctly.
The first of them is simply the test case number, three of them are timestamp parts,
and also a yes/no flag 'new_window'. So those 5 variables were deleted manually:**

```r
print(names(pmlt))
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

```r
pmlt[, 6] = pmle[, 6] = NULL
pmlt[, 5] = pmle[, 5] = NULL
pmlt[, 4] = pmle[, 4] = NULL
pmlt[, 3] = pmle[, 3] = NULL
pmlt[, 1] = pmle[, 1] = NULL
# Thus, we have 54 predictor variables, and one outcome:
print(c(dim(pmlt), dim(pmle)))
```

```
## [1] 19622    55    20    55
```

**Now we set the random seed and define the main function that will take a method name and
a data set on input, split the latter into a training set and a cross validation (CV) set,
80% and 20% accordingly, then build a model on the training set, make predictions for data
records in the CV set, report accuracy on the CV set, and return the model.**

```r
set.seed(13234)
run1set = function(mthd, trainSet) {
    inTrain = createDataPartition(y = trainSet$classe, p = 0.8, list = FALSE)
    training = trainSet[inTrain, ]
    crossval = trainSet[-inTrain, ]
    print(date())
    modFit = train(classe ~ ., data = training, method = mthd, verbose = FALSE)
    pred <- predict(modFit, crossval)
    print(table(pred, crossval$classe))
    print(paste("Accuracy =", 100 * sum(pred == crossval$classe)/length(pred), 
        "%"))
    modFit
}
options(warn = -1)
library(plyr)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(gbm)
```

```
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

**Now if we split the training set into 6 subsets, each corresponding to one person,
making use of the 'user_name' varibale and the fact that the same 6 persons are in
the** goal of the project **testing set,**

```r
pmlt1 = pmlt[pmlt$user_name == "pedro", ]
pmlt2 = pmlt[pmlt$user_name == "eurico", ]
pmlt3 = pmlt[pmlt$user_name == "jeremy", ]
pmlt4 = pmlt[pmlt$user_name == "adelmo", ]
pmlt5 = pmlt[pmlt$user_name == "charles", ]
pmlt6 = pmlt[pmlt$user_name == "carlitos", ]
```

**then our main function is not too slow at training-predicting-reporting on each of
the 6 subsets, in case the method is** 'gbm'**, that is,** Boosting With Trees:

```r
modFit1 = run1set("gbm", pmlt1)
```

```
## [1] "Sun Jun 15 01:06:27 2014"
##     
## pred   A   B   C   D   E
##    A 128   0   0   0   0
##    B   0 101   0   0   0
##    C   0   0  99   0   0
##    D   0   0   0  93   0
##    E   0   0   0   0  99
## [1] "Accuracy = 100 %"
```

```r
modFit2 = run1set("gbm", pmlt2)
```

```
## [1] "Sun Jun 15 01:10:54 2014"
##     
## pred   A   B   C   D   E
##    A 173   0   0   0   0
##    B   0 118   0   0   0
##    C   0   0  97   0   0
##    D   0   0   0 116   0
##    E   0   0   0   0 108
## [1] "Accuracy = 100 %"
```

```r
modFit3 = run1set("gbm", pmlt3)
```

```
## [1] "Sun Jun 15 01:15:55 2014"
##     
## pred   A   B   C   D   E
##    A 235   0   0   0   0
##    B   0  97   0   0   0
##    C   0   0 130   0   0
##    D   0   0   0 104   0
##    E   0   0   0   0 112
## [1] "Accuracy = 100 %"
```

```r
modFit4 = run1set("gbm", pmlt4)
```

```
## [1] "Sun Jun 15 01:21:23 2014"
##     
## pred   A   B   C   D   E
##    A 233   0   0   0   0
##    B   0 155   0   0   0
##    C   0   0 150   0   0
##    D   0   0   0 103   0
##    E   0   0   0   0 137
## [1] "Accuracy = 100 %"
```

```r
modFit5 = run1set("gbm", pmlt5)
```

```
## [1] "Sun Jun 15 01:28:06 2014"
##     
## pred   A   B   C   D   E
##    A 179   0   0   0   0
##    B   0 149   1   0   0
##    C   0   0 106   0   0
##    D   0   0   0 127   0
##    E   0   0   0   1 142
## [1] "Accuracy = 99.7163120567376 %"
```

```r
modFit6 = run1set("gbm", pmlt6)
```

```
## [1] "Sun Jun 15 01:34:20 2014"
##     
## pred   A   B   C   D   E
##    A 166   0   0   0   0
##    B   0 138   0   0   0
##    C   0   0  98   0   0
##    D   0   0   0  97   0
##    E   0   0   0   0 121
## [1] "Accuracy = 100 %"
```

```r
print(date())
```

```
## [1] "Sun Jun 15 01:39:49 2014"
```

**We can see from the six tables above that there are just two incorrect predictions,
out of almost 4000 (19622*0.2), so the average accuracy is very high.**  

**It takes a bit longer if the method is** 'rf', **that is,** Random Forests:

```r
modFit1F = run1set("rf", pmlt1)
```

```
## [1] "Sun Jun 15 01:39:53 2014"
##     
## pred   A   B   C   D   E
##    A 128   0   0   0   0
##    B   0 101   0   0   0
##    C   0   0  99   0   0
##    D   0   0   0  93   0
##    E   0   0   0   0  99
## [1] "Accuracy = 100 %"
```

```r
modFit2F = run1set("rf", pmlt2)
```

```
## [1] "Sun Jun 15 01:45:05 2014"
##     
## pred   A   B   C   D   E
##    A 173   0   0   0   0
##    B   0 118   0   0   0
##    C   0   0  97   0   0
##    D   0   0   0 116   0
##    E   0   0   0   0 108
## [1] "Accuracy = 100 %"
```

```r
modFit3F = run1set("rf", pmlt3)
```

```
## [1] "Sun Jun 15 01:51:33 2014"
##     
## pred   A   B   C   D   E
##    A 235   0   0   0   0
##    B   0  97   0   0   0
##    C   0   0 130   0   0
##    D   0   0   0 104   0
##    E   0   0   0   0 112
## [1] "Accuracy = 100 %"
```

```r
modFit4F = run1set("rf", pmlt4)
```

```
## [1] "Sun Jun 15 01:58:19 2014"
##     
## pred   A   B   C   D   E
##    A 233   0   0   0   0
##    B   0 155   0   0   0
##    C   0   0 150   0   0
##    D   0   0   0 103   0
##    E   0   0   0   0 137
## [1] "Accuracy = 100 %"
```

```r
modFit5F = run1set("rf", pmlt5)
```

```
## [1] "Sun Jun 15 02:07:15 2014"
##     
## pred   A   B   C   D   E
##    A 179   0   0   0   0
##    B   0 149   0   0   0
##    C   0   0 106   1   0
##    D   0   0   1 127   0
##    E   0   0   0   0 142
## [1] "Accuracy = 99.7163120567376 %"
```

```r
modFit6F = run1set("rf", pmlt6)
```

```
## [1] "Sun Jun 15 02:14:42 2014"
##     
## pred   A   B   C   D   E
##    A 166   0   0   0   0
##    B   0 138   0   0   0
##    C   0   0  98   0   0
##    D   0   0   0  97   0
##    E   0   0   0   0 121
## [1] "Accuracy = 100 %"
```

```r
print(date())
```

```
## [1] "Sun Jun 15 02:21:07 2014"
```

**The same number of incorrect predictions as when Boosting With Trees,
two are incorrect, but the good news is that if we make the twenty**
goal of the project **predictions, we observe that** Random Forests
**agree with** Boosting With Trees **in all 20 cases:**

```r
answers = rep("A", 20)
for (i in 1:20) {
    if (pmle$user_name[i] == "pedro") 
        answers[i] = predict(modFit1, pmle[i, ])
    if (pmle$user_name[i] == "eurico") 
        answers[i] = predict(modFit2, pmle[i, ])
    if (pmle$user_name[i] == "jeremy") 
        answers[i] = predict(modFit3, pmle[i, ])
    if (pmle$user_name[i] == "adelmo") 
        answers[i] = predict(modFit4, pmle[i, ])
    if (pmle$user_name[i] == "charles") 
        answers[i] = predict(modFit5, pmle[i, ])
    if (pmle$user_name[i] == "carlitos") 
        answers[i] = predict(modFit6, pmle[i, ])
}
print(answers)
```

```
##  [1] "2" "1" "2" "1" "1" "5" "4" "2" "1" "1" "2" "3" "2" "1" "5" "5" "1"
## [18] "2" "2" "2"
```

```r

answersF = rep("A", 20)
for (i in 1:20) {
    if (pmle$user_name[i] == "pedro") 
        answersF[i] = predict(modFit1F, pmle[i, ])
    if (pmle$user_name[i] == "eurico") 
        answersF[i] = predict(modFit2F, pmle[i, ])
    if (pmle$user_name[i] == "jeremy") 
        answersF[i] = predict(modFit3F, pmle[i, ])
    if (pmle$user_name[i] == "adelmo") 
        answersF[i] = predict(modFit4F, pmle[i, ])
    if (pmle$user_name[i] == "charles") 
        answersF[i] = predict(modFit5F, pmle[i, ])
    if (pmle$user_name[i] == "carlitos") 
        answersF[i] = predict(modFit6F, pmle[i, ])
}
print(c(sum(answers == answersF), length(answersF)))
```

```
## [1] 20 20
```

**However, to better estimate the expected out of sample error, we should not
split the original training set into 6 subsets.  
In this case our main function runs for almost an hour,
but as you might expect the accuracy is significantly lower:**

```r
modFit = run1set("gbm", pmlt)
```

```
## [1] "Sun Jun 15 02:21:11 2014"
##     
## pred    A    B    C    D    E
##    A 1114    7    0    0    0
##    B    2  749   11    2    1
##    C    0    3  670    7    1
##    D    0    0    2  634    7
##    E    0    0    1    0  712
## [1] "Accuracy = 98.8784093805761 %"
```

```r
print(date())
```

```
## [1] "Sun Jun 15 03:07:34 2014"
```

**Here are the 20** goal of the project **predictions:**

```r
print(predict(modFit, pmle))
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

**They are equal to predictions from individual personal models (variables** answers
**and** answersF**, see above), A=1, B=2, ..., E=5.**
