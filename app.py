pip install numpy pandas matplotlib scikit-learn xgboost
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
import pickle

warnings.filterwarnings('ignore')


class MNISTModel:
    def __init__(self, csv_path):

        try:
            df = pd.read_csv(csv_path)

        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

        
        try:
            self.X = df.iloc[:, 1:]  # independent
            self.y = df.iloc[:, 0]  # dependent
        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')



        # split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,test_size=0.2,random_state=42)

      
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)

        # Placeholders for trained models (matching original global names)
        self.knn_reg = None
        self.naive_reg = None
        self.lr_reg = None
        self.dt_reg = None
        self.rf_reg = None
        self.ada_reg = None
        self.gb_reg = None
        self.xg_reg = None
        self.svm_reg = None

   
    def knn(self):
        try:
            self.knn_reg = KNeighborsClassifier(n_neighbors=5)
            self.knn_reg.fit(self.X_train, self.y_train)
            print(f'KNN Test Accuracy : {accuracy_score(self.y_test, self.knn_reg.predict(self.X_test))}')
        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')


    def nb(self):
        try:
            self.naive_reg = GaussianNB()
            self.naive_reg.fit(self.X_train, self.y_train)
            print(f'Naive Bayes Test Accuracy : {accuracy_score(self.y_test, self.naive_reg.predict(self.X_test))}')
        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def lr(self):
        try:
            self.lr_reg = LogisticRegression()
            self.lr_reg.fit(self.X_train, self.y_train)
            print(f'LogisticRegression Test Accuracy : {accuracy_score(self.y_test, self.lr_reg.predict(self.X_test))}')
        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def dt(self):
        try:
            self.dt_reg = DecisionTreeClassifier(criterion='entropy')
            self.dt_reg.fit(self.X_train, self.y_train)
            print(f'DecisionTreeClassifier Test Accuracy : {accuracy_score(self.y_test, self.dt_reg.predict(self.X_test))}')
        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def rf(self):
        try:
            self.rf_reg = RandomForestClassifier(n_estimators=5, criterion='entropy')
            self.rf_reg.fit(self.X_train, self.y_train)
            print(f'RandomForestClassifier Test Accuracy : {accuracy_score(self.y_test, self.rf_reg.predict(self.X_test))}')
        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def ada(self):
        try:
            t = LogisticRegression()
            self.ada_reg = AdaBoostClassifier(estimator=t, n_estimators=5)
            self.ada_reg.fit(self.X_train, self.y_train)
            print(f'AdaBoostClassifier Test Accuracy : {accuracy_score(self.y_test, self.ada_reg.predict(self.X_test))}')
        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def gb(self):
        try:
            self.gb_reg = GradientBoostingClassifier(n_estimators=5)
            self.gb_reg.fit(self.X_train, self.y_train)
            print(f'GradientBoostingClassifier Test Accuracy : {accuracy_score(self.y_test, self.gb_reg.predict(self.X_test))}')
        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def xgb_(self):
        try:
            self.xg_reg = XGBClassifier()
            self.xg_reg.fit(self.X_train, self.y_train)
            print(f'XGBClassifier Test Accuracy : {accuracy_score(self.y_test, self.xg_reg.predict(self.X_test))}')
        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def svm_c(self):
        try:
            # Fix: SVC is initialized with probability=True to enable predict_proba
            self.svm_reg = SVC(kernel='rbf', probability=True)
            self.svm_reg.fit(self.X_train, self.y_train)
            print(f'SVM Test Accuracy : {accuracy_score(self.y_test, self.svm_reg.predict(self.X_test))}')
        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def _define_svm_param_grid(self):
        """
        Defines the hyperparameter search space for the SVC model.
        """
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
        return param_grid

    def common(self):
        # Call each model method in the same order and print separating lines like original
        print('=========KNN===========')
        self.knn()
        print('=========NB===========')
        self.nb()
        print('=========LR===========')
        self.lr()
        print('=========DT===========')
        self.dt()
        print('=========RF===========')
        self.rf()
        print('=========Adabost===========')
        self.ada()
        print('=========GB===========')
        self.gb()
        print('=========XGB===========')
        self.xgb_()
        print('=========svm===========')
        self.svm_c()

    def compute_roc_and_plot(self):

        try:
          knn_predictions = self.knn_reg.predict_proba(self.X_test)[:, 1]
          naive_predictions = self.naive_reg.predict_proba(self.X_test)[:, 1]
          lr_predictions = self.lr_reg.predict_proba(self.X_test)[:, 1]
          dt_predictions = self.dt_reg.predict_proba(self.X_test)[:, 1]
          rf_predictions = self.rf_reg.predict_proba(self.X_test)[:, 1]
          ada_predictions = self.ada_reg.predict_proba(self.X_test)[:, 1]
          gb_predictions = self.gb_reg.predict_proba(self.X_test)[:, 1]
          xgb_predictions = self.xg_reg.predict_proba(self.X_test)[:, 1]
          svm_predictions = self.svm_reg.predict_proba(self.X_test)[ : , 1]

        except Exception as e:
          ex_type,ex_msg,ex_line = sys.exc_info()
          print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

        # Binarize y_test for ROC curve calculation, focusing on class 1 vs. rest.
        y_test_binary_class_1 = (self.y_test == 1).astype(int)

        knn_fpr, knn_tpr, knn_thre = roc_curve(y_test_binary_class_1, knn_predictions)
        b_fpr, nb_tpr, nb_thre = roc_curve(y_test_binary_class_1, naive_predictions)
        lr_fpr, lr_tpr, lr_thre = roc_curve(y_test_binary_class_1, lr_predictions)
        dt_fpr, dt_tpr, dt_thre = roc_curve(y_test_binary_class_1, dt_predictions)
        rf_fpr, rf_tpr, rf_thre = roc_curve(y_test_binary_class_1, rf_predictions)
        ada_fpr, ada_tpr, ada_thre = roc_curve(y_test_binary_class_1, ada_predictions)
        gb_fpr, gb_tpr, gb_thre = roc_curve(y_test_binary_class_1, gb_predictions)
        xgb_fpr, xgb_tpr, xgb_thre = roc_curve(y_test_binary_class_1, xgb_predictions)
        svm_fpr, svm_tpr, svm_thre = roc_curve(y_test_binary_class_1, svm_predictions)

        plt.figure(figsize=(5, 3))
        plt.plot([0, 1], [0, 1], "k--")

        plt.plot(knn_fpr, knn_tpr, label="KNN")
        plt.plot(b_fpr, nb_tpr, label="NB")
        plt.plot(lr_fpr, lr_tpr, label="LR")
        plt.plot(dt_fpr, dt_tpr, label="DT")
        plt.plot(rf_fpr, rf_tpr, label="RF")
        plt.plot(ada_fpr, ada_tpr, label="ADA")
        plt.plot(gb_fpr, gb_tpr, label="GB")
        plt.plot(xgb_fpr, xgb_tpr, label="XGB")
        plt.plot(svm_fpr, svm_tpr,label ="SVM")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve - ALL Models")
        plt.legend(loc=0)
        plt.show()




if __name__ == "__main__":
    try:

      obj = MNISTModel(csv_path='/content/mnist_train.csv')

    except Exception as e:
      ex_type,ex_msg,ex_line = sys.exc_info()
      print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    # (prints accuracies)
    obj.common()

    # compute ROC curves & plot
    obj.compute_roc_and_plot()

    

    with open('ML_Project.pkl', 'wb') as file:
      pickle.dump(model_data, file)