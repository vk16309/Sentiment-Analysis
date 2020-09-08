#%%
import xlrd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc,confusion_matrix
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn import tree
#%%



#Getting positive and negative words from LoughranMcDonald_sentiwordlist excell file
workbook = xlrd.open_workbook('wordlist.xlsx')
positive_sheet = workbook.sheet_by_name('Positive')
negative_sheet= workbook.sheet_by_name('Negative')
Positive_Word=list()
Negative_Word=list()
for row in range(357):
    Positive_Word.append(positive_sheet.cell(row, 0).value.lower())

for row in range(2361):
    Negative_Word.append(negative_sheet.cell(row,0).value.lower())
    
#adding negation words
Negation = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't", "can't",
          "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt",
          "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't",
          "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "wasnt",
          "werent", "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "without", "wont", "wouldnt", "won't",
          "wouldn't", "rarely", "seldom", "despite", "no", "nobody"]




#%%
def FindScore(words):
    score=0
    for word in words:
        if word.lower() in Positive_Word:
            score+=2
        elif word.lower() in Negative_Word:
            score-=2
    return score

def Normalize(scorelist):
    #lower=min(scorelist)
    #upper=max(scorelist)
    #dif=upper-lower
    for i in range(len(scorelist)):
        #scorelist[i]=2*(scorelist[i]-lower)/dif
        #scorelist[i]-=1
        if scorelist[i]>=2:
            scorelist[i]=3
        elif scorelist[i]>=0:
            scorelist[i]=2
        else:
            scorelist[i]=1


#%%
#Reading  news data from csv file
data = pd.read_csv("stock.csv", encoding= 'unicode_escape')


#Calculating Sentiment score
sentiment_scores=list()
news_list=list()
for news in data['news'].values:
    news_list.append(news)
    words=news.split()
    score=FindScore(words)
    #print(score)
    sentiment_scores.append(score)
    
Normalize(sentiment_scores)
    

#Creating Dataframe
data_clean=pd.DataFrame({"news":news_list,"score":sentiment_scores
        })
    

   
   #%%
#Plotting Dataset    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

plt.scatter(data_clean['news'].values[:50],data_clean['score'].values[:50])
plt.show()
    
   #%% 
    #%%
#Preprocessing of dataset-Tfidf vectorization,stop-word removal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = data_clean['news'].values
vectorizer = TfidfVectorizer(stop_words = stop_words)
X= vectorizer.fit_transform(desc)
word_features = vectorizer.get_feature_names()
#print(word_features)

#%%
#Spliting dataset in 70-30 ratio
X_train, X_test = train_test_split(X, test_size=0.3, random_state=1)
Y_train, Y_test= train_test_split(data_clean['score'].values, test_size=0.3, random_state=1)
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

#%%
#Creating SVM model and training on train data
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, Y_train) 

#Prediction and calculating accuracy
svm_predictions = svm_model_linear.predict(X_test) 
#%%

accuracy_on_training_dataset= svm_model_linear.score(X_train, Y_train) 
accuracy_on_testing_dataset = svm_model_linear.score(X_test, Y_test) 


from sklearn.metrics import classification_report
import seaborn as sn

target_names = ['Buy', 'Hold', 'Sell']
print(classification_report(Y_test, svm_predictions, target_names=target_names))
cm=confusion_matrix(Y_test,svm_predictions)
plt.matshow(cm)


#%%

print("confusion matrix:",cm)
sn.set(font_scale=0.5)
sn.heatmap(cm, annot=True, annot_kws={"size": 16})
plt.show()
print("training data set accuracy for svm:",accuracy_on_training_dataset*100,"%")
print("testing data set accuracy for svm:",accuracy_on_testing_dataset*100,"%")


#%%
#DECISION TREE

from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, Y_train) 
dtree_predictions = dtree_model.predict(X_test) 

training_dt= dtree_model.score(X_train, Y_train) 
testing_dt=dtree_model.score(X_test, Y_test)

target_names = ['Buy', 'Hold', 'Sell']
print(classification_report(Y_test, dtree_predictions, target_names=target_names))
cm=confusion_matrix(Y_test,dtree_predictions)
plt.matshow(cm)

print("confusion matrix:",cm)
sn.set(font_scale=0.5)
sn.heatmap(cm, annot=True, annot_kws={"size": 16})
plt.show()
print("training data set accuracy for decision tree:",training_dt*100,"%")
print("testing data set accuracy for decision tree:",testing_dt*100,"%")
tree.plot_tree(clf)
#%%
#NAIVE BAYES

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train.todense(), Y_train) 
gnb_predictions = gnb.predict(X_test.todense())

training_nb= gnb.score(X_train.todense(), Y_train) 
testing_nb=gnb.score(X_test.todense(), Y_test)

target_names = ['Buy', 'Hold', 'Sell']
print(classification_report(Y_test, gnb_predictions, target_names=target_names))
cm=confusion_matrix(Y_test,gnb_predictions)
plt.matshow(cm)

print("confusion matrix:",cm)
sn.set(font_scale=0.5)
sn.heatmap(cm, annot=True, annot_kws={"size": 16})
plt.show()
print("training data set accuracy for Naive Bayes:",training_nb*100,"%")
print("testing data set accuracy for Naive Bayes:",testing_nb*100,"%")
