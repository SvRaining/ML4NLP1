# Exercise 1 - Language Identification with sklearn

### Data

We generated the overview data report for both training data and test data using pandas-profiling. The main **distribution properties** are discribed below.

**train_dev_set.tsv**: Consists of 52675 pieces of data, each of which has two variables - tweet and label.

![image-20221018093005311](/Users/apple/Library/Application Support/typora-user-images/image-20221018093005311.png)

There are no missing cells in this dataset, and the total size of the file is 11.5MB.

![image-20221018091158037](/Users/apple/Library/Application Support/typora-user-images/image-20221018091158037.png)

The tweets are string-type comments divided into 69 categories by the languages they are using, the top 3 languages are English, Japanese and Spanish. More about the category distribution can be found in the figure below.

![image-20221018092619418](/Users/apple/Library/Application Support/typora-user-images/image-20221018092619418.png)

**test_set.tsv**: Consists of 13279 pieces of data with the same variables of the training dataset. The file size is 2.9 MB and there are no missing values. Only 60 kinds of languages are showed in this dataset. The category distribution is showed in the figure below. We found it similar to the training dataset.

![image-20221018093957651](/Users/apple/Library/Application Support/typora-user-images/image-20221018093957651.png)

## Part1 - Linear classification

### Step 1 - Generate a Pipeline

We first generated the preprocessing **pipeline**. It contains 3 parts: Dataloader, TweetCleaner and FeatureExtractor.

![image-20221018094609241](/Users/apple/Library/Application Support/typora-user-images/image-20221018094609241.png)

We used TF_IDF vectorizer to extract features. It aims to quantify the importance of a given word relative to other words in the document and in the corpus. We set the **feature space** to 500 and some of the features are showed below.

![image-20221018095149294](/Users/apple/Library/Application Support/typora-user-images/image-20221018095149294.png)

### Step 2 - Train the LR Model

We trained the Logistic Regression model and used GridSearchCV to find the best parameters. The combinations are recorded with predicting accuracy.

| Penalty↓ Solver→ | lbfgs | Newton-cg | sag  | Saga |
| ---------------- | ----- | --------- | ---- | ---- |
| **L2**           | 0.62  | 0.62      | 0.62 | 0.62 |
| **None**         | 0,58  | 0.58      | 0.57 | 0.58 |

As we can see, **the best result** comes from {Penalty=12, Solver=lbfgs}.

By using grid search cross-validation, we were able to create better fitting models by training and testing on all parts of the training dataset.

### Step 3 - Visualize the Results

We calculated the confusion matrix to do error analysis. Part of the matrix is showed below.

![image-20221018101121577](/Users/apple/Library/Application Support/typora-user-images/image-20221018101121577.png)

To be specific, the first line of the matrix is:

![image-20221018101339925](/Users/apple/Library/Application Support/typora-user-images/image-20221018101339925.png)

We also generated the classification report with the precision, reall, f1 score and support.

![image-20221018101518173](/Users/apple/Library/Application Support/typora-user-images/image-20221018101518173.png)

We generated a feature importance table for the top ten features for English, Japanese and Spanish.

![image-20221018101739680](/Users/apple/Library/Application Support/typora-user-images/image-20221018101739680.png)

We noticed there seems to be something wrong with the Japanese features. The TfidfVectorizer cannot do the word tokenization for Japanese properly.  To verify our observation, we also tried nltk library for word tokenizing, but we still found it not working.

## Part 2 - MLP

We played around 5 different sets of hyper parameters, and the results are showned below.

| Parameters                                                   | Accuracy |
| ------------------------------------------------------------ | -------- |
| hidden_layer_sizes = (150,), solver = lbfgs, early_stopping = True | 0.66     |
| hidden_layer_sizes = (150,), solver = Adam, early_stopping = True | 0.68     |
| hidden_layer_sizes = (150,), solver = Adam, early_stopping = False | 0.61     |
| hidden_layer_sizes = (100,), solver = lbfgs, early_stopping = True | 0.59     |
| hidden_layer_sizes = (100,), solver = lbfgs, early_stopping = False | 0.57     |

The results we got are slightly better than the LR model. The reason may be its ability to deal with nonlinear relations.

[Notice: we used the default parameter settings in our sample notebook, so the result may differ from this form. Also, we found it inconvenient to divide this program into two files, so we left it in one file for both lr and mlp:-) ] 
