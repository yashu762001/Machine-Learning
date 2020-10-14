import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def entropy(col):
    count = np.unique(col, return_counts=True)
    # count is going to be a tuple storing a list of unique elements and a list of count of every unique element.
    ent = 0.0

    for i in count[1]:
        p = i / col.shape[0]
        ent = ent + (-1.0 * p * np.log2(p))

    return ent


def segregate_data(x_data, column_of_segregation, mean_value_of_column):
    x_left = pd.DataFrame([], columns=x_data.columns)
    x_right = pd.DataFrame([], columns=x_data.columns)

    for i in range(x_data.shape[0]):
        value = x_data[column_of_segregation].loc[i]

        if value >= mean_value_of_column:
            x_left = x_left.append(x_data.iloc[i])
        else:
            x_right = x_right.append(x_data.iloc[i])

    x_left.index = np.arange(x_left.shape[0])
    x_right.index = np.arange(x_right.shape[0])

    return x_left, x_right


def Information_Gain(x_data, fkey, fval, fkey1):
    left, right = segregate_data(x_data, fkey, fval)
    col1 = left[fkey1]
    col2 = right[fkey1]

    entropy_left = entropy(col1)
    entropy_right = entropy(col2)
    a = col1.shape[0]
    b = col2.shape[0]
    if a == 0 or b == 0:
        return "No Segregation Possible Since this column consists of same values. Try With Different Attribute."
    average_entropy_of_children = (a / (a + b)) * entropy_left + (b / (a + b)) * entropy_right
    return entropy(x_data[fkey1]) - average_entropy_of_children


class DecisionTree:
    def __init__(self, max_depth=5, depth=0):
        self.left = None
        self.right = None
        self.depth = depth
        self.max_depth = max_depth
        self.fkey = None
        self.fval = None
        self.target = None

    def train(self, x_train):
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        info_gain = []

        for feature in features:
            gain = Information_Gain(x_train, feature, x_train[feature].mean(), "Survived")
            info_gain.append(gain)
        print(info_gain)
        self.fkey = features[np.argmax(info_gain)]
        # This tells which is the best attribute for segregation.

        self.fval = x_train[self.fkey].mean()

        print("Splitting Training Data", self.fkey, self.fval)

        data_left, data_right = segregate_data(x_train, self.fkey, self.fval)

        if data_left.shape[0] == 0 or data_right.shape[0] == 0:
            if x_train.Survived.mean() >= 0.5:
                self.target = "Will Stay Alive"

            else:
                self.target = "Will Die"

            return

        if self.depth + 1 <= self.max_depth:
            self.left = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)
            self.left.train(data_left)

            self.right = DecisionTree(depth=self.depth + 1, max_depth=self.depth)
            self.right.train(data_right)

        if x_train.Survived.mean() >= 0.5:
            self.target = "Will Stay Alive"

        else:
            self.target = "Will Die"

        return

    def predict(self, test):
        if test[self.fkey].mean() >= self.fval:
            if self.left is None:
                return self.target

            else:
                return self.left.predict(test)

        elif test[self.fkey].mean() < self.fval:
            if self.right is None:
                return self.target

            else:
                return self.right.predict(test)


titanic_data = pd.read_csv('train.csv')
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

del_cols = ["PassengerId", "Name", "Ticket", "Cabin"]

titanic_data = titanic_data.drop(del_cols, axis=1)
titanic_data = titanic_data.fillna(titanic_data["Age"].mean())

encoder = LabelEncoder()

titanic_data["Sex"] = encoder.fit_transform(titanic_data["Sex"])

for i in range(0, titanic_data.shape[0]):
    if titanic_data["Embarked"].iloc[i] == "S":
        titanic_data["Embarked"].iloc[i] = 0

    elif titanic_data["Embarked"].iloc[i] == "C":
        titanic_data["Embarked"].iloc[i] = 1

    else:
        titanic_data["Embarked"].iloc[i] = 2

split_value = int(0.6 * titanic_data.shape[0])
training_data = titanic_data[:split_value]
testing_data = titanic_data[split_value:]

testing_data = testing_data.reset_index(drop=True)

tree = DecisionTree()
tree.train(training_data)

predicted_results = []

for i in range(0, testing_data.shape[0]):
    result = tree.predict(testing_data.iloc[i])
    if result == "Will Stay Alive":
        predicted_results.append(1)
    else:
        predicted_results.append(0)

print(np.mean(predicted_results == testing_data["Survived"]))
