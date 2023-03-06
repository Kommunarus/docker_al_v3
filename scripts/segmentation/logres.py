from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import tree
import json
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from sklearn.preprocessing import StandardScaler


data = []
target = []
with open('dataset_for_logres.txt') as f:
    for row in f.readlines():
        row_d = json.loads(row)
        data.append(row_d['mask'])
        target.append(row_d['reward'])

print(len(data))

X = np.array(data)
X = np.append(X, np.expand_dims(X[:, 0] + X[:, 1], 1), 1) #10
X = np.append(X, np.expand_dims(X[:, 0] + X[:, 1] + X[:, 2], 1), 1) #11
X = np.append(X, np.expand_dims(X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3], 1), 1) #12
X = np.append(X, np.expand_dims(X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] + X[:, 4], 1), 1) #13
X = np.append(X, np.expand_dims(X[:, 9] + X[:, 8], 1), 1) #14
X = np.append(X, np.expand_dims(X[:, 9] + X[:, 8] + X[:, 7], 1), 1) #15
X = np.append(X, np.expand_dims(X[:, 9] + X[:, 8] + X[:, 7] + X[:, 6], 1), 1) #16
X = np.append(X, np.expand_dims(X[:, 9] + X[:, 8] + X[:, 7] + X[:, 6] + X[:, 5], 1), 1) #17
X = np.append(X, np.expand_dims(X[:, 4] + X[:, 5], 1), 1) #18
X = np.append(X, np.expand_dims(X[:, 4] + X[:, 5] + X[:, 6], 1), 1) #19
X = np.append(X, np.expand_dims(X[:, 3] + X[:, 4] + X[:, 5] + X[:, 6], 1), 1) #20
X = np.append(X, np.expand_dims(X[:, 3] + X[:, 4] + X[:, 5] + X[:, 6] + X[:, 7], 1), 1) #21
X = np.append(X, np.expand_dims(X[:, 2] + X[:, 3] + X[:, 4] + X[:, 5] + X[:, 6] + X[:, 7], 1), 1) #22


X = X[:, (0, 9, 21)]



y = np.array(target)
y_bool = (y > 0.07).astype(int)

x_train, x_test, y_train, y_test, y_class_train, y_class_test = train_test_split(X, y, y_bool, test_size=0.2,
                                                                                 random_state=42)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

print('tree')
for depth in [3]:
    print('depth', depth)
    classificator = DecisionTreeClassifier(random_state=2, max_depth=depth, min_samples_leaf=5, criterion='gini',
                                           splitter='best')
    model_class = classificator.fit(x_train, y_class_train)
    pred_class = model_class.predict(x_test)

    dot_data = tree.export_graphviz(model_class, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data)
    png = graph.create_png()
    with open("image.png", "wb") as img:
        img.write(png)

# print(y_class_test.tolist())
# print(pred_class.tolist())

    matrix = confusion_matrix(y_class_test, pred_class)
    print(matrix)
    p = precision_score(y_class_test, pred_class)
    print(p)
    r = recall_score(y_class_test, pred_class)
    print(r)

print('lin')

for c in [2]:
    print(c)
    log_reg = LogisticRegression(random_state=4, penalty='l1', solver='saga', C=c)
    model = log_reg.fit(x_train, y_class_train)
    pred_class = model.predict(x_test)
    print(model.coef_.tolist())
    print(np.argsort(model.coef_).tolist())
    print(np.sort(model.coef_).tolist())

    # print(y_class_test.tolist())
    # print(pred_class.tolist())

    matrix = confusion_matrix(y_class_test, pred_class)
    print(matrix)
    p = precision_score(y_class_test, pred_class)
    print(p)
    r = recall_score(y_class_test, pred_class)
    print(r)

# plt.plot(pred)
# plt.plot(y_test)
# plt.show()
