import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
#sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#displaying head of the sample
df=pd.read_csv("C:\mini\water_potability.csv")
df.head()
#index
df.columns
#description(count,mean...)
df.describe()
# info of the data
df.info()
#sum of null values
df.isnull().sum()
#heat map for sum values
plt.figure(figsize=(20,10))
sns.heatmap(df.isnull())
#heatmap for correlation between features
plt.figure(figsize=(20,10))
sns.heatmap(df.isnull())
#coutplot for target
import seaborn as sns
sns.countplot(x="Potability",data=df)
#value count of target
df["Potability"].value_counts()
#visualization of outliners
fig, ax=plt.subplots(ncols=5,nrows=2,figsize=(20,10))
ax=ax.flatten()
index=0
for col,values in df.items():
  sns.boxplot(y=col,data=df,ax=ax[index])
  index+=1

#pairplot for data
sns.pairplot(df)
#pie chart for portability
fig=px.pie(df,names="Potability",hole=0.4,template="plotly_dark")
fig.show()
#scatter plot for sulfate vs portability
fig=px.scatter(df,x="ph",y="Sulfate",color="Potability",template="plotly_dark")
fig.show()
#scatter plot for organic_carbon vs potability
fig=px.scatter(df,x="Organic_carbon",y="Sulfate",color="Potability",template="plotly_dark")
fig.show()
#plot bar for features vs percentage of missing values
df.isnull().mean().plot.bar(figsize=(10,6))
plt.xlabel("Features")
plt.ylabel("Percentage of missing values")
#filling of null values
df["ph"]=df["ph"].fillna(df["ph"].mean())
df["Sulfate"]=df["Sulfate"].fillna(df["Sulfate"].mean())
df["Trihalomethanes"]=df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean())
#sum of no ofnull values
df.isnull().sum()
#heatmap for null values
sns.heatmap(df.isnull())
#head for first five values and index tags
df.head()
#seperating features and target from data set
x=df.drop("Potability",axis=1)
y=df["Potability"]
#shape of the data
x.shape,y.shape
#turning all the data into one scale of numbers that can be processed by the model
scaler=StandardScaler()
x=scaler.fit_transform(x)
x
#splitting the data for training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#shape of test and train data
x_train.shape,x_test.shape
#logistic regression
from sklearn.linear_model import LogisticRegression
#iogistic object
model_lr=LogisticRegression()
#trainig model
model_lr.fit(x_train,y_train)
#making prediction
pred_lr=model_lr.predict(x_test)
pred_lr
#accuacy score
accuracy_score_lr=accuracy_score(y_test,pred_lr)
accuracy_score_lr
#decision tree
from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier(max_depth=4)
#train the dt
model_dt.fit(x_train,y_train)
#prediction
pred_dt=model_dt.predict(x_test)
#accuracy score
accuracy_score_dt=accuracy_score(y_test,pred_dt)
accuracy_score_dt
#confusion matrix with heatmap
cm2=confusion_matrix(y_test,pred_dt)

sns.heatmap(cm2/np.sum(cm2))
cm2
#random forest classifier
from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier()
#trainig model
model_rf.fit(x_train,y_train)
#prediction of the model
pred_rf=model_rf.predict(x_test)
#accuracy of the model
accuracy_rf=accuracy_score(y_test,pred_rf)
accuracy_rf
#confusion matrix
cm3=confusion_matrix(y_test,pred_rf)
cm3
#knn algorithm
from sklearn.neighbors import KNeighborsClassifier
# creating a model object
model_knn=KNeighborsClassifier()

model_knn=KNeighborsClassifier(n_neighbors=6)
model_knn.fit(x_train,y_train)
#prediction
pred_knn=model_knn.predict(x_test)
#accuracy score
accuracy_knn=accuracy_score(y_test,pred_knn)
print(accuracy_knn*100)
#svm
from sklearn.svm import SVC
#creating model
model_svm=SVC(kernel='rbf')
#trainig object
model_svm.fit(x_train,y_train)
#predicting
pred_svm=model_svm.predict(x_test)
#accuracy_score
accuracy_svm=accuracy_score(y_test,pred_svm)
print(accuracy_svm*100)
#ada boost classifier
from sklearn.ensemble import AdaBoostClassifier
#creating a object
model_ada=AdaBoostClassifier(n_estimators=200,learning_rate=0.03)
#training the data
model_ada.fit(x_train,y_train)
#prediction
pred_ada=model_ada.predict(x_test)
#accuray
accuracy_ada=accuracy_score(y_test,pred_ada)
accuracy_ada*100
#xgboost
from xgboost import XGBClassifier
#creating model
model_xgb=XGBClassifier(n_estimators=200,learning_rate=0.3)
model_xgb.fit(x_train,y_train)
#prediction
pred_xgb=model_xgb.predict(x_test)
#accuracy
accuracy_xgb=accuracy_score(y_test,pred_xgb)
accuracy_xgb*100
# displaying models accuracy
models=pd.DataFrame({
    "Model" : ["Logistic Regression",
               "Decision Tree",
               "Random Forest",
               "KNN",
               "SVM",
               "AdaBoost",
               "XGBoost"] ,
    "Accuracy Score" : [accuracy_score_lr,accuracy_score_dt,accuracy_rf,
                        accuracy_knn,accuracy_svm,accuracy_ada,accuracy_xgb]
})
models
#sorting
sns.barplot(x="Accuracy Score",y= "Model",data=models)
models.sort_values(by="Accuracy Score")
#making of pickle file
pickle.dump(model_svm,open("model.pkl","wb"))