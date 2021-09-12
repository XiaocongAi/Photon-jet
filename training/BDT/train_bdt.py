#coding=utf-8
#!/usr/bin/env python3

import sys
import os
import random
#import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import ROOT 
import joblib

import seaborn as sns
import pandas as pd
import pandas.core.common as com
from pandas.core.index import Index

from pandas.plotting import scatter_matrix
import imblearn
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
#from sklearn.learning_curve import learning_curve
#from sklearn import grid_search

import enum
print(enum.__file__)  

def usage():
	print ('test usage')
	sys.stdout.write('''
			SYNOPSIS
			./BDT_pre.py signal bkg folder 
			DATE
			10 Jan 2021
			\n''')

def main():

    args = sys.argv[1:]
    if len(args) < 2:
        return usage()

    print ('part1')   

    # get root files and convert them to array

    #9 variables (ATLAS egamma) 
    branch_names = """frac_first,first_lateral_width_eta_w20,first_lateral_width_eta_w3,first_fraction_fside,first_dEs,first_Eratio,second_R_eta,second_R_phi,second_lateral_width_eta_weta2""".split(",")

    
    fin1 = ROOT.TFile(args[0])
    fin2 = ROOT.TFile(args[1])
    output = args[2]
    print("output = ", output)

    tree1 = fin1.Get("fancy_tree")
    signal0 = tree1.AsMatrix(columns=branch_names)
    signal = signal0[:100000,:]

    tree2 = fin2.Get("fancy_tree")
    backgr0 = tree2.AsMatrix(columns=branch_names)
    backgr = backgr0[:100000,:]

    # for sklearn data is usually organised into one 2D array of shape (n_samples * n_features)
    # containing all the data and one array of categories of length n_samples
    X_raw = np.concatenate((signal, backgr))
    y_raw = np.concatenate((np.ones(signal.shape[0]), np.zeros(backgr.shape[0])))
    print(len(signal))

    print ('part2')
    print(len(y_raw[y_raw==1]))
    print(len(y_raw[y_raw==0]))

    #imbalanced learn
    n_sig = len(y_raw[y_raw==1])
    n_bkg = len(y_raw[y_raw==0])
#    sb_ratio = len(y_raw[y_raw==1])/(1.0*len(y_raw[y_raw==0]))
#    if (sb_ratio > 0.2 and sb_ratio < 0.5):
#        smote = SMOTE(ratio=0.5)
#        X, y = smote.fit_sample(X_raw, y_raw)
#        print ('Number of events: ')
#        print ('before: signal: ', len(y_raw[y_raw==1]), ' background: ', len(y_raw[y_raw==0]))
#        print ('after: signal: ', len(y[y==1]), ' background: ', len(y[y==0]))
#    elif (n_sig > 1000 and sb_ratio < 0.2 and sb_ratio > 0.1):
#        smote = SMOTE(ratio=0.2)
#        X, y = smote.fit_sample(X_raw, y_raw)
#        print ('Number of events: ')
#        print ('before: signal: ', len(y_raw[y_raw==1]), ' background: ', len(y_raw[y_raw==0]))
#        print ('after: signal: ', len(y[y==1]), ' background: ', len(y[y==0]))
#    elif (n_sig < 1000 and sb_ratio < 0.2 and sb_ratio > 0.1):
#        smote = SMOTE(ratio=0.4)
#        X, y = smote.fit_sample(X_raw, y_raw)
#        print ('Number of events: ')
#        print ('before: signal: ', len(y_raw[y_raw==1]), ' background: ', len(y_raw[y_raw==0]))
#        print ('after: signal: ', len(y[y==1]), ' background: ', len(y[y==0]))
#    elif (sb_ratio < 0.1 and sb_ratio > 0.05):
#        smote = SMOTE(ratio=0.4)
#        X, y = smote.fit_sample(X_raw, y_raw)
#        print ('Number of events: ')
#        print ('before: signal: ', len(y_raw[y_raw==1]), ' background: ', len(y_raw[y_raw==0]))
#        print ('after: signal: ', len(y[y==1]), ' background: ', len(y[y==0]))
#    elif (sb_ratio < 0.05 and sb_ratio > 0.01):
#        smote = SMOTE(ratio=0.1)
#        X, y = smote.fit_sample(X_raw, y_raw)
#        print ('Number of events: ')
#        print ('before: signal: ', len(y_raw[y_raw==1]), ' background: ', len(y_raw[y_raw==0]))
#        print ('after: signal: ', len(y[y==1]), ' background: ', len(y[y==0]))
#    elif (sb_ratio < 0.01):
#        smote = SMOTE(ratio=0.03)
#        X, y = smote.fit_sample(X_raw, y_raw)
#        print ('Number of events: ')
#        print ('before: signal: ', len(y_raw[y_raw==1]), ' background: ', len(y_raw[y_raw==0]))
#        print ('after: signal: ', len(y[y==1]), ' background: ', len(y[y==0]))
#    else:
#        X = X_raw
#        y = y_raw
#        print ('Number of events: ')
#        print ('signal: ', len(y[y==1]), ' background: ', len(y[y==0]))


    """
    Training Part
    """
    # Train and test
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.30, random_state=3443)
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=344)

    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=200, min_samples_split=10)

    bdt = AdaBoostClassifier(dt, algorithm='SAMME', n_estimators=100, learning_rate=1.0)
    bdt.fit(X_train, y_train)
    print('tune hyperparameters')
#    run_grid_search("sklearnClassification", bdt, X_train, y_train)

    importances = bdt.feature_importances_
    f = open('bdt_results_9var/'+output+'/output_importance.txt', 'w')
    f.write("%-25s%-15s\n"%('Variable Name','Output Importance'))
    for i in range (len(branch_names)):
        f.write("%-25s%-15s\n"%(branch_names[i], importances[i]))
        print("%-25s%-15s\n"%(branch_names[i], importances[i]), f)
    f.close() 

    y_predicted = bdt.predict(X_train)
    print (classification_report(y_train, y_predicted, target_names=["background", "signal"]))
    print ("Area under ROC curve: %.4f"%(roc_auc_score(y_train, bdt.decision_function(X_train))))

    y_predicted = bdt.predict(X_test)
    print (classification_report(y_test, y_predicted, target_names=["background", "signal"]))
    print ("Area under ROC curve: %.4f"%(roc_auc_score(y_test, bdt.decision_function(X_test))))

    #y_pred = bdt.predict_proba(X_train)[:,1]
    #print(y_pred) 
    #res=roc_auc_score(y_train, y_pred)
    #print(res)

    precision, recall, th = precision_recall_curve(y_test, y_predicted)
    plt.plot(th, precision[1:], label="Precision",linewidth=5) 
    plt.plot(th, recall[1:], label="Recall",linewidth=5) 
    plt.title('Precision and recall for different threshold values') 
    plt.xlabel('Threshold') 
    plt.ylabel('Precision/Recall') 
    plt.legend() 
    plt.show()
  

    decisions1 = bdt.decision_function(X_train)
    decisions2 = bdt.decision_function(X_test)

    filepath = 'ROC'

    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    # Compute ROC curve and area under the curve
    fpr1, tpr1, thresholds1 = roc_curve(y_train, decisions1)
    fpr2, tpr2, thresholds2 = roc_curve(y_test, decisions2)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    fig=plt.figure(figsize=(8,6))
    fig.patch.set_color('white')
    plt.plot(fpr1, tpr1, lw=1.2, label='train:ROC (area = %0.4f)'%(roc_auc1), color="r")
    plt.plot(fpr2, tpr2, lw=1.2, label='test: ROC (area = %0.4f)'%(roc_auc2), color="b")
    plt.plot([0,1], [0,1], '--', color=(0.6, 0.6, 0.6), label = 'Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating  characteristic')
    plt.legend(loc = "lower right")
    plt.grid()
    plt.savefig("./bdt_results_9var/"+output+"/"+filepath+"/ROC.png")
#    plt.show()

    compare_train_test(bdt, X_train, y_train, X_test, y_test, output, filepath)
#    plot_learning_curve("./bdt_results/", bdt, X_train, y_train)
    plot_correlations("./bdt_results_9var/"+output+"/", branch_names, signal, backgr)
    plot_inputs("./bdt_results_9var/"+output+"/", branch_names, signal, None, backgr, None)

    joblib.dump(bdt, './bdt_results_9var/'+output+'/'+filepath+'/bdt_model.pkl')

# Comparing train and test results
def compare_train_test(clf, X_train, y_train, X_test, y_test, output, savepath, bins=50):

    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)
    fig=plt.figure(figsize=(8,5.5))
    fig.patch.set_color('white')
    plt.hist(decisions[0], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density = True, label='Signal (train)')
    plt.hist(decisions[1], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density = True, label='Background (train)')
    
    hist, bins = np.histogram(decisions[2], bins=bins, range=low_high, density=True)
    scale = len(decisions[2])/sum(hist)
    err = np.sqrt(hist*scale)/scale

    width = (bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='Signal (test)')

    hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, density=True)
    scale = len(decisions[2])/sum(hist)
    err = np.sqrt(hist*scale)/scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='Background (test)')
  
    plt.xlabel("BDT score")
    plt.ylabel("Normalized Unit")
    plt.legend(loc='best')
    plt.savefig("./bdt_results_9var/"+output+"/"+savepath+"/BDTscore.png")
#    plt.show()

def run_grid_search(outdir, bdt, x, y):
	logging.info('starting hyper-parameter optimization')
	param_grid = {"n_estimators": [50,100,800,1000], 'learning_rate': [0.01,0.1,0.5]}

	clf = grid_search.GridSearchCV(bdt, param_grid, cv=CV, scoring='roc_auc', n_jobs=NJOBS, verbosity=2)
	clf.fit(x, y)

	out = '\nHyper-parameter optimization\n'
	out += '============================\n\n'
	out += 'Best estimator: {}\n'.format(clf.best_estimator_)
	out += '\nFull Scores\n'
	out += '-----------\n\n'
	for params, mean_score, scores in clf.grid_scores_:
		out += u'{:0.4f} (±{:0.4f}) for {}\n'.format(mean_score, scores.std(), params)
	with codecs.open(os.path.join(outdir, "log-hyper-parameters.txt"), "w", encoding="utf8") as fd:
		fd.write(out)

def plot_inputs(outdir, vars, sig, sig_w, bkg, bkg_w):
    for n, var in enumerate(vars):
        _, bins = np.histogram(np.concatenate((sig[:, n], bkg[:, n])), bins=40)
        sns.distplot(bkg[:, n], hist_kws={'weights': bkg_w}, bins=bins, kde=False, norm_hist=True, label='background')
        sns.distplot(sig[:, n], hist_kws={'weights': sig_w}, bins=bins, kde=False, norm_hist=True, label='signal')
        plt.title(var)
        plt.legend()
        plt.savefig(os.path.join(outdir, 'input_{}.png'.format(var)))
        plt.savefig(os.path.join(outdir, 'input_{}.png'.format(var)))
        plt.close()
def plot_correlations(outdir, vars, sig, bkg):
    for data, label in ((sig, "Signal"), (bkg, "Background")):
        d = pd.DataFrame(data, columns=vars)
        sns.heatmap(d.corr(), annot=True, fmt=".2f", linewidth=.5)
        plt.title(label + " Correlations")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'correlations_{}.png'.format(label.lower())))
        plt.savefig(os.path.join(outdir, 'correlations_{}.png'.format(label.lower())))
        plt.close()

def plot_learning_curve(outdir, bdt, x, y):
	logging.info("creating learning curve")
	train_sizes, train_scores, test_scores = learning_curve(bdt,
								x,
								y,
		                                                cv=ShuffleSplit(len(x),
		                                                n_iter=100,
		                                                test_size=1.0 / CV),
		                                            	n_jobs=NJOBS,
								verbosity=2,
		                                            	train_sizes=np.linspace(.1, 1., 7),
		                                            	scoring='roc_auc')
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.fill_between(train_sizes,
		     train_scores_mean - train_scores_std,
		     train_scores_mean + train_scores_std,
		     alpha=.2, color='r')
	plt.fill_between(train_sizes,
		     test_scores_mean - test_scores_std,
		     test_scores_mean + test_scores_std,
		     alpha=.2, color='g')
	plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
	plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

	plt.xlabel("Sample size")
	plt.ylabel("Score (ROC area)")

	plt.legend()
	plt.savefig(os.path.join(outdir, 'learning-curve.png'))
	plt.savefig(os.path.join(outdir, 'learning-curve.pdf'))
	plt.close()

if __name__ == '__main__':
	print('start')
	main()
