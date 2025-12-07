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
from sklearn.metrics import roc_curve
import warnings
import pickle

warnings.filterwarnings('ignore')

# ===================== GRID SEARCH FOR SVM (BEST PARAMETERS) =====================

best_params = None   # will hold best params from GridSearchCV

try:
    df_gs = pd.read_csv('/content/mnist_train.csv')   # change path if needed
    df_gs = df_gs.dropna()                            # remove NaNs

    X_gs = df_gs.iloc[:, 1:]  # features
    y_gs = df_gs.iloc[:, 0]   # labels

    X_train_gs, X_test_gs, y_train_gs, y_test_gs = train_test_split(
        X_gs, y_gs, test_size=0.3, random_state=32
    )

    svm = SVC()
    parameters = {
        'kernel': ['rbf', 'poly', 'linear'],
        'C': [1.0, 2.0, 3.0],
        'gamma': ['scale', 'auto'],
        'decision_function_shape': ['ovr', 'ovo'],
        'probability': [False],   # kept same as your code
    }
    grid_model = GridSearchCV(estimator=svm, param_grid=parameters, cv=2, scoring='accuracy')
    result = grid_model.fit(X_train_gs, y_train_gs)
    best_params = result.best_params_
    print("Best SVM parameters from GridSearchCV:", best_params)

except Exception as e:
    ex_type, ex_msg, ex_tb = sys.exc_info()
    print(f'Issue in GridSearchCV block at line {ex_tb.tb_lineno} : due to : {ex_msg}')
    # if error, best_params will stay None


# ============================== MULTI-MODEL CLASS ==============================

class MNISTModel:
    def __init__(self, csv_path):

        try:
            df = pd.read_csv(csv_path)

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

        try:
            self.X = df.iloc[:, 1:]  # independent
            self.y = df.iloc[:, 0]   # dependent
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

        # split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)

        # Placeholders for trained models
        self.knn_reg = None
        self.naive_reg = None
        self.lr_reg = None
        self.dt_reg = None
        self.rf_reg = None
        self.ada_reg = None
        self.gb_reg = None
        self.xg_reg = None
        self.svm_reg = None

        # This will hold tuned SVM in training()/testing()
        self.svm = None

    def knn(self):
        try:
            self.knn_reg = KNeighborsClassifier(n_neighbors=5)
            self.knn_reg.fit(self.X_train, self.y_train)
            print(f'KNN Test Accuracy : {accuracy_score(self.y_test, self.knn_reg.predict(self.X_test))}')
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def nb(self):
        try:
            self.naive_reg = GaussianNB()
            self.naive_reg.fit(self.X_train, self.y_train)
            print(f'Naive Bayes Test Accuracy : {accuracy_score(self.y_test, self.naive_reg.predict(self.X_test))}')
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def lr(self):
        try:
            self.lr_reg = LogisticRegression()
            self.lr_reg.fit(self.X_train, self.y_train)
            print(f'LogisticRegression Test Accuracy : {accuracy_score(self.y_test, self.lr_reg.predict(self.X_test))}')
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def dt(self):
        try:
            self.dt_reg = DecisionTreeClassifier(criterion='entropy')
            self.dt_reg.fit(self.X_train, self.y_train)
            print(f'DecisionTreeClassifier Test Accuracy : {accuracy_score(self.y_test, self.dt_reg.predict(self.X_test))}')
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def rf(self):
        try:
            self.rf_reg = RandomForestClassifier(n_estimators=5, criterion='entropy')
            self.rf_reg.fit(self.X_train, self.y_train)
            print(f'RandomForestClassifier Test Accuracy : {accuracy_score(self.y_test, self.rf_reg.predict(self.X_test))}')
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def ada(self):
        try:
            t = LogisticRegression()
            self.ada_reg = AdaBoostClassifier(estimator=t, n_estimators=5)
            self.ada_reg.fit(self.X_train, self.y_train)
            print(f'AdaBoostClassifier Test Accuracy : {accuracy_score(self.y_test, self.ada_reg.predict(self.X_test))}')
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def gb(self):
        try:
            self.gb_reg = GradientBoostingClassifier(n_estimators=5)
            self.gb_reg.fit(self.X_train, self.y_train)
            print(f'GradientBoostingClassifier Test Accuracy : {accuracy_score(self.y_test, self.gb_reg.predict(self.X_test))}')
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def xgb_(self):
        try:
            self.xg_reg = XGBClassifier()
            self.xg_reg.fit(self.X_train, self.y_train)
            print(f'XGBClassifier Test Accuracy : {accuracy_score(self.y_test, self.xg_reg.predict(self.X_test))}')
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def svm_c(self):
        try:
            # SVC with probability=True to enable predict_proba for ROC curve
            self.svm_reg = SVC(kernel='rbf', probability=True)
            self.svm_reg.fit(self.X_train, self.y_train)
            print(f'SVM Test Accuracy : {accuracy_score(self.y_test, self.svm_reg.predict(self.X_test))}')
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def _define_svm_param_grid(self):
        """
        Defines the hyperparameter search space for the SVC model.
        (You are not using this below, but kept same as your code.)
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
            svm_predictions = self.svm_reg.predict_proba(self.X_test)[:, 1]

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')
            return

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
        plt.plot(svm_fpr, svm_tpr, label="SVM")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve - ALL Models")
        plt.legend(loc=0)
        plt.show()

    # ---------------- TUNED SVM: TRAINING & TESTING ----------------

    def training(self):
        """
        Training tuned SVM using best_params from GridSearchCV.
        Prints train accuracy, classification_report, confusion_matrix.
        """
        try:
            global best_params

            if best_params is None:
                print("best_params is None (GridSearchCV failed or not run). Using default SVC.")
                self.svm = SVC()
            else:
                # Use tuned parameters here
                self.svm = SVC(
                    C=best_params.get('C', 1.0),
                    kernel=best_params.get('kernel', 'rbf'),
                    gamma=best_params.get('gamma', 'scale'),
                    decision_function_shape=best_params.get('decision_function_shape', 'ovr'),
                    probability=best_params.get('probability', False)
                )

            self.svm.fit(self.X_train, self.y_train)

            y_train_pred = self.svm.predict(self.X_train)
            print(f'train accuracy: {accuracy_score(self.y_train, y_train_pred)}')
            print(f'classification report:\n{classification_report(self.y_train, y_train_pred)}')
            print(f'confusion matrix:\n{confusion_matrix(self.y_train, y_train_pred)}')

        except Exception as e:
            ex_type, ex_tb, ex_msg = sys.exc_info()
            print(f'issue is from {ex_tb.tb_lineno} due to {ex_msg}')

    def testing(self):
        """
        Testing tuned SVM on test data.
        Prints test accuracy, classification_report, confusion_matrix.
        """
        try:
            y_test_pred = self.svm.predict(self.X_test)
            print(f'test accuracy: {accuracy_score(self.y_test, y_test_pred)}')
            print(f'classification report:\n{classification_report(self.y_test, y_test_pred)}')
            print(f'confusion matrix:\n{confusion_matrix(self.y_test, y_test_pred)}')
        except Exception as e:
            ex_type, ex_tb, ex_msg = sys.exc_info()
            print(f'issue is from {ex_tb.tb_lineno} due to {ex_msg}')


# ================================ MAIN BLOCK ================================

if __name__ == "__main__":
    try:
        obj = MNISTModel(csv_path='/content/mnist_train.csv')  # change path if needed

        # 1) All models + ROC using base SVM
        obj.common()
        obj.compute_roc_and_plot()

        # 2) Tuned SVM training + testing
        print("\n========= Tuned SVM (Using GridSearchCV best_params) =========")
        obj.training()
        obj.testing()

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')
