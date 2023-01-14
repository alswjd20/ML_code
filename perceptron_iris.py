from sklearn import datasets
import numpy as np

LF = '\n'

iris = datasets.load_iris()

# check the target features of the dataset
print (iris['target_names'])
print (['target'])

# see the descriptive features of the sample
print (iris['data'])
print (iris['data'].shape)

X = iris.data[:, [2, 3]]
y = iris.target
print (X, '\n', y)

#----------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=0)

#----------------------------------------------

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print (X_train_std[0], LF, X_train[0])

#----------------------------------------------
"""

from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter_no_change = 40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

"""


import perceptron as p

ppn = p.Perceptron()
ppn.fit(X_train, y_train)

y_pred = ppn.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


#----------------------------------------------

from sklearn.metrics import precision_score, recall_score


print('Precision: %.2f' % precision_score(y_test, y_pred, average = 'macro')) # 정밀도

print('Recall : %.2f' % recall_score(y_test, y_pred, average = 'macro')) # 재현율


#----------------------------------------------

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred)) # 정확도

import cp_utils as cpu
import matplotlib.pyplot as plt

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

cpu.plot_decision_regions(X=X_combined_std, y=y_combined, \
	classifier=ppn, test_idx=range(105,150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

print ('End of Program')