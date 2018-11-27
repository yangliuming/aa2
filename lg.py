import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
os.chdir( 'E:\\workspace\\IterationCode')
path='./data/1001.csv'
data=pd.read_csv(path,encoding='utf-8')
data01=data.dropna(axis=0, how='any')
data02=data01.iloc[:,1:25]
data03=data01['IF_JZ_FK']
X_train, X_test, y_train, y_test = train_test_split(data02,data03, test_size=0.3)

dt = DictVectorizer(sparse=False)
print(X_train.to_dict(orient="record"))
x_train = dt.fit_transform(X_train.to_dict(orient="record"))
x_test = dt.fit_transform(X_test.to_dict(orient="record"))

# 使用决策树
dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)

dt_predict = dtc.predict(x_test)

print(dtc.score(x_test, y_test))

print(classification_report(y_test, dt_predict, target_names=["died", "survived"]))

# 使用随机森林
rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

rfc_y_predict = rfc.predict(x_test)
rfc_y_predict01=rfc.predict(x_train)

print(rfc.score(x_test, y_test))

print(classification_report(y_test, rfc_y_predict, target_names=["died", "survived"]))