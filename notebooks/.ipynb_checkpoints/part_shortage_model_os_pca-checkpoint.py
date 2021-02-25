print('importing packages....')
print('')

#Import packages
import re
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, log_loss, roc_auc_score, roc_curve, classification_report, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import re
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

print('___' * 50)
print('importing data..')
print('___' * 50)
print('')

#Import data

path = r'/data/home/mq814d/poc/data/'

full_data = pd.read_csv(os.path.join(path, 'combo_May212019.csv'))
X_train = pd.read_csv(os.path.join(path, 'X_train_poc.csv'))
y_train = pd.read_csv(os.path.join(path, 'y_train_poc.csv'))
X_valid = pd.read_csv(os.path.join(path, 'X_valid_poc.csv'))
y_valid = pd.read_csv(os.path.join(path, 'y_valid_poc.csv'))
X_test = pd.read_csv(os.path.join(path, 'X_test_poc.csv'))

#Reshape y
c, r = y_train.shape
y_train = y_train.values.reshape(c, )
c, r = y_valid.shape
y_test = y_valid.values.reshape(c, )

#Model definition
models = ['random_forest', 'decision_tree', 'gradient_boosting']

params_dict = {'random_forest':  {"min_samples_split": [3, 5, 10, 15, 20],
              "min_samples_leaf": [30, 30, 40, 50],
              "bootstrap": [True, False],
              "criterion": ['entropy', 'gini'],
              "n_estimators": [50, 100, 150, 200],
              "random_state": [23]},
            'gradient_boosting': {"min_samples_split": [3, 5, 10, 15, 20],
              "min_samples_leaf": [20, 30, 40, 50],
             "n_estimators": [50, 100, 150, 200],
             "random_state": [23]},
            'decision_tree': {"min_samples_split": [3, 5, 10, 15, 20],
              "min_samples_leaf": [20, 30, 40, 50],
              "splitter": ['best', 'random'],
              "criterion": ['entropy', 'gini'],
              "max_depth": [50, 100, 150, 200],
              "random_state": [23]}}

model_dict = {'random_forest': RandomForestClassifier(), 'decision_tree': DecisionTreeClassifier(), 'gradient_boosting': GradientBoostingClassifier()}

fit_model_dict = {}
pred_dict = {}
pred_true_proba_dict = {}
pred_false_proba_dict = {}
test_pred_dict = {}
test_true_proba = {}
test_false_proba = {}

#pca_vals
pca_vals = [2, 5, 10, 15, 20, 30]

for val in pca_vals: 
    #PCA
    pca = PCA(n_components = val)
    X_train_pca = pca.fit_transform(X_train_os)
    X_valid_pca = pca.transform(X_valid)
    X_test_pca = pca.transform(X_test)
    for model in models:
        print('training %s' %(model))
        print('')
        clf = GridSearchCV(model_dict['%s' %(model)], param_grid = params_dict['%s' %(model)], n_jobs = 100, verbose = 10)
        clf.fit(X_train_pca, y_train)

        #Validation preds                  
        preds = clf.predict(X_valid_pca)
        pred_proba = clf.predict_proba(X_valid_pca)
        fit_model_dict['%s' %(model)] = clf
        pred_dict['%s' %(model)] = preds
        pred_true_proba_dict['%s' %(model)] = pred_proba[:,1]
        pred_false_proba_dict['%s' %(model)] = pred_proba[:,0]

        #Test preds
        test_preds = clf.predict(X_test_pca)
        test_pred_proba = clf.predict_proba(X_test_pca)
        fit_model_dict['%s' %(model)] = clf
        test_pred_dict['%s' %(model)] = test_preds
        test_true_proba['%s' %(model)] = test_pred_proba[:,1]
        test_false_proba['%s' %(model)] = test_pred_proba[:,0]

        print('___' * 50)
        print('%s Precision: ' %(model) + str(precision_score(y_valid, preds)))
        print('%s Accuracy: ' %(model) + str(accuracy_score(y_valid, preds)))
        print('%s Recall: ' %(model) + str(recall_score(y_valid, preds)))
        print('%s AUC: ' %(model) + str(roc_auc_score(y_valid, preds)))
        print('___' * 50)
        print('')
        print('___' * 50)
        print('%s completed' %(model))
        print('___' * 50)
        print('')

    #Charts

    print('___' * 50)
    print('Constructing charts')
    print('___' * 50)
    print('')

    path = r'/data/home/mq814d/poc/outputs/pca/pca_%d' %(val)

    for model in models:

        #Print AUC score
        fpr, tpr, _ = roc_curve(y_valid, pred_true_proba_dict['%s' %(model)])
        auc = roc_auc_score(pred_dict['%s' %(model)], y_valid)
        plt.plot(fpr, tpr, label= "%s, auc=" %(model) + str(auc))
        plt.legend(loc=4)
        plt.tight_layout()
        plt.savefig('test.jpg')
        plt.savefig(os.path.join(path, 'auc_curves_combined_poc.jpg'))

    for model in models:

        #Feature importance
        feat_importances = pd.Series(fit_model_dict['%s' %(model)].best_estimator_.feature_importances_, index = X_train_pca.columns)
        plt.clf()
        ax = plt.gca()
        ax.tick_params(axis = 'x', colors = 'black')
        ax.tick_params(axis = 'y', colors = 'black')
        plt.title('Feature Importance', color = 'black')
        feat_importances.nlargest(10).plot(kind ='barh')
        plt.tight_layout()
        plt.savefig('%s.jpg' %(model))
        plt.savefig(os.path.join(path, 'feature_importance_%s_poc.jpg' %(model)))

    for model in models:

        #Confusion matrix
        cm = metrics.confusion_matrix(pred_dict['%s' %(model)], y_valid)
        plt.clf()
        plt.imshow(cm, interpolation ='nearest', cmap = plt.cm.Wistia)
        classNames = ['Negative','Positive']
        ax = plt.gca()
        ax.tick_params(axis = 'x', colors = 'black')
        ax.tick_params(axis = 'y', colors = 'black')
        plt.title('Confusion Matrix', color = 'black')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation = 45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j - 0.25, i, str(s[i][j]) + " = " + str(cm[i][j]))
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'confusion_matrix_%s_poc.jpg' %(model)))

    print('___' * 50)
    print('Charts complete')
    print('___' * 50)
    print('')

    print('___' * 50)
    print('Creating output prediction files')
    print('___' * 50)
    print('')

    #Function to create and save metric report
    def report_to_df(report):
        report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
        report_df = pd.read_csv(StringIO("Classes" + report), sep = ' ', index_col = 0)
        return(report_df)

    #Predictions

    for model in models:

        path = r'/data/home/mq814d/poc/outputs/pca/pca_%d' %(val)

        metrics_ = metrics.classification_report(pred_dict['%s' %(model)], y_valid)
        metrics_df = report_to_df(metrics_)
        metrics_df.to_csv(os.path.join(path, 'metrics_report_%s_poc.csv' %(model)))

        #Validation features to predictions
        valid_features = full_data[full_data.ln.isin(X_valid.ln)]
        valid_features['actual'] = y_valid
        valid_features['prediction'] = pred_dict['%s' %(model)]
        valid_features['prob_no_shortage'] = pred_false_proba_dict['%s' %(model)]
        valid_features['prob_shortage'] = pred_true_proba_dict['%s' %(model)]

        #Test features to predictions
        test_features = full_data[full_data.ln.isin(X_test.ln)]
        test_features['prediction'] = test_pred_dict['%s' %(model)]
        test_features['prob_no_shortage'] = test_false_proba['%s' %(model)]
        test_features['prob_shortage'] = test_true_proba['%s' %(model)]                

        #Sorted preds
        sorted_test = test_features.sort_values('prob_shortage', ascending = False)
        sorted_test['surrog_ind'] = np.arange(len(sorted_test))

        sorted_valid = valid_features.sort_values('prob_shortage', ascending = False)
        sorted_valid['surrog_ind'] = np.arange(len(sorted_valid))

        #Validation Precisions
        prec_score = []
        prec_vals = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500]
        for prec_ in prec_vals:
            prec_score.append(precision_score(sorted_valid.head(prec_).actual, sorted_valid.head(prec_).prediction))
                
        #Test probability chart
        ax = sorted_test.iloc[0:5000].plot.line(x = 'surrog_ind', y = 'prob_shortage', title = 'Test Top 5000 Proabilities of Shortage')
        ax.set_xlabel('Number of Parts')
        ax.set_ylabel('Probability')   
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'test_proba_curve_%s_poc.jpg' %(model)))

        #Validation probability chart
        ax = plt.plot(prec_vals, prec_score , title = 'Validation Precision on True Class')
        ax.set_xlabel('Number of Parts')
        ax.set_ylabel('Precision')   
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'valid_prec_curve_%s_poc.jpg' %(model)))

        #Test labels
        test_features.to_csv(os.path.join(path, 'test_preds_with_features_%s_poc.csv' %(model)))
        valid_features.to_csv(os.path.join(path, 'valid_preds_with_features_%s_poc.csv' %(model)))

        #Model params

        path = r'/data/home/mq814d/poc/outputs/pca/pca_%d' %(val)

        print(fit_model_dict['%s' %(model)].best_estimator_)
        pd.DataFrame(fit_model_dict['%s' %(model)].best_params_, index = [0]).to_csv(os.path.join(path, 'model_params_%s_poc.csv' %(model)))

    print('___' * 50)
    print('script complete..')
    print('___' * 50)
