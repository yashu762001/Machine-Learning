from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('train.csv')
del_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
data.drop(del_cols, axis=1)
data = data.fillna(data["Age"].mean())
encoder = LabelEncoder()
data["Sex"] = encoder.fit_transform(data["Sex"])
split = int(0.7*data.shape[0])
train_data = data.loc[:split]
test_data = data.loc[split:]

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
# features represent the set of attributes which we will consider for segregating the data.

sk_tree = DecisionTreeClassifier(max_depth=4)
# max_depth is setted just to increase the efficiency of data since if there is no limit then training data would be overfitted and so the test data would fail not miserably but to a
# larger extent as compared to the case when the max_depth is set.
sk_tree.fit(train_data[features], train_data["Survived"])

y_pred = sk_tree.predict(test_data[features])

print(sk_tree.score(test_data[features], test_data["Survived"]))
#This tells us what is the percentage accuracy in the model we created and the actual data.
