import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)

# Generate train data
#X = 0.3 * np.random.randn(100, 2)
# Generate some abnormal novel observations
#X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
#X = np.r_[X + 2, X - 2, X_outliers]
#print(X,type(X))

# set ([x,y],[x2,y2])
def get_csv_data(data_limit=None):
    fueldata = [];
    i = 0;
    with open('fueldataset2.csv', newline="\r\n") as csvfile:
        spamreader = csv.reader(csvfile, delimiter="\t", quotechar='"')
        for row in spamreader:
            if row and row[0]!='frame_order':
                new_element = [int(row[0]),int(row[2])] #0-frameorder, 1-sum, 2-raw, 3-gpsdatetime
                fueldata.append(new_element)
    if data_limit is None:
        result = fueldata
    else:
        result = fueldata[0::data_limit]
    return np.asarray(result);

X = get_csv_data(10)

# fit the model
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)
#print(X);
#sys.exit()
outliers = [];
for key, val in enumerate(X):
    if y_pred[key] and y_pred[key] == -1:
        outliers.append(val);

#print(type(X),X[:, 0])
#sys.exit()

X = np.array(X)
outliers = np.array(outliers)

plt.title("Local Outlier Factor (LOF)")

a = plt.scatter(X[:, 0], X[:, 1], c='white', edgecolor='k', s=10)
b = plt.scatter(outliers[:, 0], outliers[:, 1], c='red', edgecolor='k', s=10)


plt.axis('tight')
plt.legend([a, b],
           ["normal observations",
            "abnormal observations"],
           loc="upper left")
plt.show()
