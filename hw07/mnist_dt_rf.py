#!/usr/bin/python

####################################################
# module: mnist_digits_random_forest.py
# description: Testing random forest for MNIST
# bugs to vladimir dot kulyukin via canvas
####################################################

'''
Shubham Swami
A02315672

1. Performance report and confusion matrix of top performing DT
precision    recall  f1-score   support

           0       0.91      0.94      0.92       980
           1       0.95      0.95      0.95      1135
           2       0.88      0.84      0.86      1032
           3       0.85      0.86      0.85      1010
           4       0.87      0.86      0.87       982
           5       0.83      0.82      0.82       892
           6       0.87      0.89      0.88       958
           7       0.89      0.89      0.89      1028
           8       0.82      0.80      0.81       974
           9       0.83      0.85      0.84      1009

    accuracy                           0.87     10000
   macro avg       0.87      0.87      0.87     10000
weighted avg       0.87      0.87      0.87     10000

confusion matrix
[[ 918    1    7    5    9   15    7    5    6    7]
 [   0 1083    5    9    2    5   10    5   15    1]
 [  10   11  864   30   14   14   16   32   26   15]
 [   6    4   19  866    6   34    4   11   33   27]
 [   7    4   10    5  847   11   25   18   17   38]
 [  13    8    8   44   10  728   25   12   21   23]
 [  23    5   12    5   17   21  853    2   16    4]
 [   2   12   26   12   10    7    6  920   11   22]
 [  14   10   22   34   22   32   19   13  776   32]
 [  12    4    8   14   39   15   10   21   31  855]]

I got the same accuracy for all the DT's

2. performance and confusion matrix for RF
Number of DT's = 50
Performance and confusion matrix:

              precision    recall  f1-score   support

           0       0.97      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.95      0.97      0.96      1032
           3       0.95      0.96      0.96      1010
           4       0.97      0.97      0.97       982
           5       0.97      0.95      0.96       892
           6       0.97      0.97      0.97       958
           7       0.97      0.95      0.96      1028
           8       0.96      0.95      0.96       974
           9       0.95      0.94      0.94      1009

    accuracy                           0.97     10000
   macro avg       0.96      0.96      0.96     10000
weighted avg       0.97      0.97      0.97     10000

[[ 972    1    1    0    0    2    2    1    1    0]
 [   0 1122    4    2    0    2    2    0    2    1]
 [   4    0  998    5    4    0    6   10    5    0]
 [   1    0   13  969    0    9    0    7    7    4]
 [   1    0    0    0  948    0    5    0    5   23]
 [   4    0    2   16    4  851    7    1    4    3]
 [   7    3    1    0    3    6  933    0    5    0]
 [   1    5   24    1    1    0    0  981    3   12]
 [   5    1    6    7    5    6    3    4  925   12]
 [   6    5    1   15   13    4    2    5    6  952]]

 3.
The accuracy of Random Forest with one DT is low when compared to accuracy of a single DT so for one DT it doesn't make sense.
However, Ensemble learning is a powerful machine learning paradigm which has exhibited apparent advantages in many applications.
By using multiple learners, the generalization ability of an ensemble can be much better than that of a single learner.
The bias-variance decomposition is often used in studying the performance of ensemble methods. It is known that Bagging can significantly reduce the variance,
and therefore it is better to be applied to learners suffered from large variance,e.g., unstable learners such as decision trees.
Boosting can significantly reduce the bias in addition to reducing the variance, and therefore, on weak learners such as decision stumps, Boosting is usually more effective.

4. MNIST CovNet accuracy= 98.7
MNIST DT = 88
MNIST RF = 97
The accuracy of MNIST ConvNet is better than RF and DT
Conclusions:
Use Neural Network for:Images, Audio, Text
If you want to work with tabular data, it is worth checking the Random Forests first because it is easier.
The Random Forests requires less pre processing and the training process is simpler.

'''

from sklearn import tree, metrics
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from mnist_loader import load_data_wrapper

mnist_train_data, mnist_test_data, mnist_valid_data = \
    load_data_wrapper()

mnist_train_data_dc = np.zeros((50000, 784))
mnist_test_data_dc = np.zeros((10000, 784))
mnist_valid_data_dc = np.zeros((10000, 784))

mnist_train_target_dc = None
mnist_test_target_dc = None
mnist_valid_target_dc = None


def reshape_mnist_aux(mnist_data, mnist_data_dc):
    '''auxiliary function to reshape MNIST data for sklearn.'''
    for i in range(len(mnist_data)):
        mnist_data_dc[i] = mnist_data[i][0].reshape((784,))


def reshape_mnist_data():
    '''reshape all MNIST data for sklearn.'''
    global mnist_train_data
    global mnist_train_data_dc
    global mnist_test_data
    global mnist_test_data_dc
    global mnist_valid_data
    global mnist_valid_data_dc
    reshape_mnist_aux(mnist_train_data, mnist_train_data_dc)
    reshape_mnist_aux(mnist_test_data, mnist_test_data_dc)
    reshape_mnist_aux(mnist_valid_data, mnist_valid_data_dc)


def reshape_mnist_target(mnist_data):
    '''reshape MNIST target given data.'''
    return np.array([np.argmax(mnist_data[i][1])
                     for i in range(len(mnist_data))])


def reshape_mnist_target2(mnist_data):
    '''another function for reshaping MNIST target given data.'''
    return np.array([mnist_data[i][1] for i in range(len(mnist_data))])


def prepare_mnist_data():
    '''reshape and prepare MNIST data for sklearn.'''
    global mnist_train_data
    global mnist_test_data
    global mnist_valid_data
    reshape_mnist_data()

    ### make sure that train, test, and valid data are reshaped
    ### correctly.
    for i in range(len(mnist_train_data)):
        assert np.array_equal(mnist_train_data[i][0].reshape((784,)),
                              mnist_train_data_dc[i])

    for i in range(len(mnist_test_data)):
        assert np.array_equal(mnist_test_data[i][0].reshape((784,)),
                              mnist_test_data_dc[i])

    for i in range(len(mnist_valid_data)):
        assert np.array_equal(mnist_valid_data[i][0].reshape((784,)),
                              mnist_valid_data_dc[i])


def prepare_mnist_targets():
    '''reshape and prepare MNIST targets for sklearn.'''
    global mnist_train_target_dc
    global mnist_test_target_dc
    global mnist_valid_target_dc
    mnist_train_target_dc = reshape_mnist_target(mnist_train_data)
    mnist_test_target_dc = reshape_mnist_target2(mnist_test_data)
    mnist_valid_target_dc = reshape_mnist_target2(mnist_valid_data)


def fit_validate_dt():
    clf = tree.DecisionTreeClassifier(random_state=random.randint(0, 1000))
    dtr = clf.fit(mnist_train_data_dc, mnist_train_target_dc)
    valid_preds = dtr.predict(mnist_valid_data_dc)
    print(metrics.classification_report(mnist_valid_target_dc, valid_preds))
    cm1 = confusion_matrix(mnist_valid_target_dc, valid_preds)
    print("confusion matrix ", cm1)
    # test_preds = dtr.predict(mnist_test_data_dc)
    # print(metrics.classification_report(mnist_test_target_dc, test_preds))
    # cm2 = confusion_matrix(mnist_test_target_dc, test_preds)
    # print(cm2)


def fit_validate_dts(num_dts):
    for _ in range(num_dts):
        fit_validate_dt()


def fit_validate_rf(num_dts):
    rs = random.randint(0, 1000)
    clf = RandomForestClassifier(n_estimators=num_dts, random_state=rs)
    rf = clf.fit(mnist_train_data_dc, mnist_train_target_dc)
    valid_preds = rf.predict(mnist_valid_data_dc)
    print(metrics.classification_report(mnist_valid_target_dc, valid_preds))
    cm1 = confusion_matrix(mnist_valid_target_dc, valid_preds)
    print(cm1)


def fit_validate_rfs(low_nt, high_nt):
    for i in range(low_nt, high_nt + 1, 10):
        print(i, ": ")
        fit_validate_rf(i)


## Let's prepare MNIST data for unit tests.
prepare_mnist_data()
prepare_mnist_targets()

'''
if __name__ == '__main__':
    prepare_mnist_data()
    prepare_mnist_targets()
'''
