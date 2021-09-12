import sys
import os
import random
#import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import ROOT 
import joblib
import ctypes

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
#from sklearn.learning_curve import learning_curve
#from sklearn import grid_search


ROOT.gStyle.SetPadLeftMargin(0.15);
ROOT.gStyle.SetPadRightMargin(0.05);
ROOT.gStyle.SetPadTopMargin(0.1);
ROOT.gStyle.SetPadBottomMargin(0.15);
ROOT.gStyle.SetTitleSize(0.04, "xy");
ROOT.gStyle.SetLabelSize(0.04, "xy");
ROOT.gStyle.SetTitleOffset(1.2,"x");
ROOT.gStyle.SetTitleOffset(1.2, "y");
#ROOT.gStyle.SetNdivisions(505, "y");


def usage():
	print ('test usage')
	sys.stdout.write('''
			SYNOPSIS
			./BDT_pre.py tr_sig tr_sig0, tr_bkg0 tr_bkg1 test_sig test_bkg0 test_bkg1 
			AUTHOR
			Yanxi Gu <GUYANXI@ustc.edu>
			DATE
			10 Jan 2021
			\n''')


def findBin(bins, var, ibin):
    for i in range(len(bins)-1):
        if var>=bins[i] and var<bins[i+1]:
           ibin.append(i)



def main():

    args = sys.argv[1:]
    if len(args) < 3:
        return usage()

    print ('part1')   

    # get root files and convert them to array
    branch_names = """frac_first,first_lateral_width_eta_w20,first_lateral_width_eta_w3,first_fraction_fside,first_dEs,first_Eratio,second_R_eta,second_R_phi,second_lateral_width_eta_weta2""".split(",")
    branch_ene = """total_e""".split(",")

    fin1 = ROOT.TFile(args[0])
    fin2 = ROOT.TFile(args[1])
    fin3 = ROOT.TFile(args[2])
    fin4 = ROOT.TFile(args[3])
    fin5 = ROOT.TFile(args[4])
    fin6 = ROOT.TFile(args[5])
  
    output = args[6]

    # ########### Train samples #############
    train_nEvents = 70000
    train_tree1 = fin1.Get("fancy_tree")
    train_sig0 = train_tree1.AsMatrix(columns=branch_names)
    train_sig0_ene = train_tree1.AsMatrix(columns=branch_ene)
    train_signal0 = train_sig0[:np.int(train_nEvents),:]
    train_signal0_ene = train_sig0_ene[:np.int(train_nEvents),:]
    
    train_tree2 = fin2.Get("fancy_tree")
    train_backgr0 = train_tree2.AsMatrix(columns=branch_names)
    train_backgr0_ene = train_tree2.AsMatrix(columns=branch_ene)
    train_background0 = train_backgr0[:np.int(train_nEvents),:]
    train_background0_ene = train_backgr0_ene[:np.int(train_nEvents),:]
    
    train_tree3 = fin3.Get("fancy_tree")
    train_backgr1 = train_tree3.AsMatrix(columns=branch_names)
    train_backgr1_ene = train_tree3.AsMatrix(columns=branch_ene)
    train_background1 = train_backgr1[:np.int(train_nEvents),:]
    train_background1_ene = train_backgr1_ene[:np.int(train_nEvents),:]

    # ########### Test samples #############
    test_nEvents = 30000
    test_tree1 = fin4.Get("fancy_tree")
    test_sig0 = test_tree1.AsMatrix(columns=branch_names)
    test_sig0_ene = test_tree1.AsMatrix(columns=branch_ene)
    test_signal0 = test_sig0[:np.int(test_nEvents),:]
    test_signal0_ene = test_sig0_ene[:np.int(test_nEvents),:]

    test_tree3 = fin5.Get("fancy_tree")
    test_backgr0 = test_tree3.AsMatrix(columns=branch_names)
    test_backgr0_ene = test_tree3.AsMatrix(columns=branch_ene)
    test_background0 = test_backgr0[:np.int(test_nEvents),:]
    test_background0_ene = test_backgr0_ene[:np.int(test_nEvents),:]

    test_tree4 = fin6.Get("fancy_tree")
    test_backgr1 = test_tree4.AsMatrix(columns=branch_names)
    test_backgr1_ene = test_tree4.AsMatrix(columns=branch_ene)
    test_background1 = test_backgr1[:np.int(test_nEvents),:]
    test_background1_ene = test_backgr1_ene[:np.int(test_nEvents),:]

    

    # for sklearn data is usually organised into one 2D array of shape (n_samples * n_features)
    # containing all the data and one array of categories of length n_samples
    train_X_raw = np.concatenate((train_signal0, train_background0, train_background1))
    train_X_raw_ene = np.concatenate((train_signal0_ene, train_background0_ene, train_background1_ene))
    test_X_raw = np.concatenate((test_signal0, test_background0, test_background1))
    test_X_raw_ene = np.concatenate((test_signal0_ene, test_background0_ene, test_background1_ene))
    
    train_y_raw = np.concatenate((np.ones(train_signal0.shape[0]), np.zeros(train_background0.shape[0]),np.ones(train_background1.shape[0])+1))
    test_y_raw = np.concatenate((np.ones(test_signal0.shape[0]), np.zeros(test_background0.shape[0]), np.ones(test_background1.shape[0])+1))
    #train_y_raw = np.concatenate((np.zeros(train_signal0.shape[0]), np.ones(train_background0.shape[0]),np.ones(train_background1.shape[0])+1))
    #test_y_raw = np.concatenate((np.zeros(test_signal0.shape[0]), np.ones(test_background0.shape[0]), np.ones(test_background1.shape[0])+1))
    
    print(len(train_signal0))
    print(len(test_signal0))

    print ('part2')
    print(len(test_y_raw[test_y_raw==1]))
    print(len(test_y_raw[test_y_raw==0]))
    print(len(test_y_raw[test_y_raw==2]))


    """
    Training Part
    """
    # Train and test
    #X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.30, random_state=3443)
    X_train = train_X_raw
    
    X_test = test_X_raw
    X_test_ene = test_X_raw_ene
    #X_test_comb = list(zip(X_test, X_test_ene))
    #print("X_test_comb", X_test_comb)
    
    y_train = train_y_raw
    y_test = test_y_raw

    ###################################################################

    #category the X_test into a few sub arrays
    #ene_bins = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 250]
    ene_bins = [0, 100, 150, 200, 300]
    #sigEff = ROOT.TEfficiency("Signal efficiency", "", len(ene_bins)-1, ctypes.c_void_p(ene_bins.ctypes.data)) 
    #bkg0Eff = ROOT.TEfficiency("Bkg0 efficiency", "", len(ene_bins)-1, ctypes.c_void_p(ene_bins.ctypes.data)) 
    #bkg1Eff = ROOT.TEfficiency("Bkg1 efficiency", "", len(ene_bins)-1, ctypes.c_void_p(ene_bins.ctypes.data)) 
    #sigEff = ROOT.TEfficiency("Signal efficiency", "", 12, 19, 271) 
    #bkg0Eff = ROOT.TEfficiency("Bkg0 efficiency", "", 12, 19, 271) 
    #bkg1Eff = ROOT.TEfficiency("Bkg1 efficiency", "", 12, 19, 271) 
    sigEff = ROOT.TEfficiency("Signal efficiency", "", 10, 40, 250) 
    bkg0Eff = ROOT.TEfficiency("Bkg0 efficiency", "", 10, 40, 250) 
    bkg1Eff = ROOT.TEfficiency("Bkg1 efficiency", "", 10, 40, 250) 

    ene = []
    for i in range(len(ene_bins)-1):
        ene.append((ene_bins[i]+ene_bins[i+1])/2)

    ###################################################################
#    X_train_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_train_sig_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_train_bkg0_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_train_bkg1_binned = [[], [], [], [], [], [], [], [], [], []]
#    y_train_binned = [[], [], [], [], [], [], [], [], [], []]
#   
#    X_test_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_test_sig_binned = [[], [], [], [], [], [], [], [], [], []]
#    y_test_binned = [[], [], [], [], [], [], [], [], [], []]
#    
#    X_test_bkg0_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_test_bkg1_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_test_ene_binned = [[], [], [], [], [], [], [], [], [], []]
#    
#    X_test_sig_ene_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_test_bkg0_ene_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_test_bkg1_ene_binned = [[], [], [], [], [], [], [], [], [], []]

    X_train_binned = [[], [], [], []]
    X_train_sig_binned = [[], [], [], []]
    X_train_bkg0_binned = [[], [], [], []]
    X_train_bkg1_binned = [[], [], [], []]
    y_train_binned = [[], [], [], []] 

    X_test_binned = [[], [], [], []] 
    X_test_sig_binned = [[], [], [], []] 
    y_test_binned = [[], [], [], []] 

    X_test_bkg0_binned = [[], [], [], []] 
    X_test_bkg1_binned = [[], [], [], []] 
    X_test_ene_binned = [[], [], [], []] 

    X_test_sig_ene_binned = [[], [], [], []] 
    X_test_bkg0_ene_binned = [[], [], [], []] 
    X_test_bkg1_ene_binned = [[], [], [], []] 


    # Categorize the train_X
    for i in range (len(train_X_raw)):
        ibin=[]
        #Find which bin this belongs
        findBin(ene_bins, train_X_raw_ene[i][0]/1000., ibin)
        if len(ibin)>0:
            y_train_binned[ibin[0]].append(train_y_raw[i])

            X_raw_col = []
            for j in range (len(train_X_raw[i])):
                X_raw_col.append(train_X_raw[i][j])

            X_train_binned[ibin[0]].append(X_raw_col)
            #print("X_raw_col = ", X_raw_col, "label", train_y_raw[i]) 
            if train_y_raw[i] ==1 :
                X_train_sig_binned[ibin[0]].append(X_raw_col)
            elif train_y_raw[i] == 0:
                X_train_bkg0_binned[ibin[0]].append(X_raw_col)
            elif train_y_raw[i] == 2:
                X_train_bkg1_binned[ibin[0]].append(X_raw_col)

    # Categorize the test_X
    for i in range (len(test_X_raw)):
        ibin=[]
        findBin(ene_bins, test_X_raw_ene[i][0]/1000., ibin)
        if len(ibin)>0:
            y_test_binned[ibin[0]].append(test_y_raw[i])
           
            #Copy
            X_raw_col = []
            for j in range (len(test_X_raw[i])):
                X_raw_col.append(test_X_raw[i][j])
            X_raw_ene_col = []
            for j in range (len(test_X_raw_ene[i])):
                X_raw_ene_col.append(test_X_raw_ene[i][j])

            X_test_binned[ibin[0]].append(X_raw_col)
            X_test_ene_binned[ibin[0]].append(X_raw_ene_col)
           
            if test_y_raw[i] ==1 :
                X_test_sig_binned[ibin[0]].append(X_raw_col)
                X_test_sig_ene_binned[ibin[0]].append(X_raw_ene_col)
            elif test_y_raw[i] == 0:
                X_test_bkg0_binned[ibin[0]].append(X_raw_col)
                X_test_bkg0_ene_binned[ibin[0]].append(X_raw_ene_col)
            elif test_y_raw[i] == 2:
                X_test_bkg1_binned[ibin[0]].append(X_raw_col)
                X_test_bkg1_ene_binned[ibin[0]].append(X_raw_ene_col)

    ###################################################################


    bdt = GradientBoostingClassifier(max_depth=5, min_samples_leaf=200, min_samples_split=10,  n_estimators=100, learning_rate=1.0)
    bdt.fit(X_train, y_train)

    print("Accuracy score (training): {0:.3f}".format(bdt.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(bdt.score(X_test, y_test)))

    importances = bdt.feature_importances_
    f = open('gbdt_results_9var/'+output+'/output_importance.txt', 'w')
    f.write("%-25s%-15s\n"%('Variable Name','Output Importance'))
    for i in range (len(branch_names)):
        f.write("%-25s%-15s\n"%(branch_names[i], importances[i]))
        print("%-25s%-15s\n"%(branch_names[i], importances[i]), file=f)
    f.close() 

    y_predicted = bdt.predict(X_train)
#    print (classification_report(y_train, y_predicted, target_names=["background", "signal","background1"]))
#    print ("Area under ROC curve: %.4f"%(roc_auc_score(y_train, bdt.decision_function(X_train),multi_class="ovr")))

    y_predicted = bdt.predict(X_test)
#    print (classification_report(y_test, y_predicted, target_names=["background", "signal","background1"]))
#    print ("Area under ROC curve: %.4f"%(roc_auc_score(y_test, bdt.decision_function(X_test),multi_class="ovo")))

    #The dimension of decisions1 is equal to the number of labels (one-vs-all for each label)
    # The colomn i is the score for distinguishing the label i vs. others
    decisions1 = bdt.decision_function(X_train)
    decisions2 = bdt.decision_function(X_test)
    #print("decisions 1 size ", decisions1.shape[1]) 


    filepath = 'ROC'
    compare_train_test(bdt, X_train, y_train, X_test, y_test, output, filepath)
    plot_correlations("./gbdt_results_9var/"+output+"/", branch_names, train_signal0, train_background0)
    plot_inputs("./gbdt_results_9var/"+output+"/", branch_names, train_signal0, None, train_background0, None, train_background1,None)
    joblib.dump(bdt, './gbdt_results_9var/'+output+'/'+filepath+'/bdt_model.pkl')


    #Get 0 or 1 indicating the first/second/third cololum has label equaling to 0/1/2 or not
    y_train = label_binarize(y_train, classes = [0, 1, 2])
    y_test = label_binarize(y_test, classes = [0, 1, 2])
    y_predicted = label_binarize(y_predicted, classes = [0, 1, 2])
    n_classes = y_test.shape[1] 
    print("n_classes=", y_test.shape[1])

    # Compute ROC curve and area under the curve
    for i in range(n_classes):
        fpr1, tpr1, thresholds1 = roc_curve(y_train[:, i], decisions1[:, i])
        fpr2, tpr2, thresholds2 = roc_curve(y_test[:, i], decisions2[:, i])
        roc_auc1 = auc(fpr1, tpr1)
        roc_auc2 = auc(fpr2, tpr2)
        #print("fpr2 =", fpr2)
        #print("tpr2 =", tpr2)
        #print("thresholds2 =", thresholds2)
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
        plt.savefig('./gbdt_results_9var/'+output+'/'+filepath+'/ROC_%i.png'%(i))


    # Compute recall
#    for i in range(3):
#        precision, recall, th = precision_recall_curve(y_test[:, i],
#                                                       y_predicted[:, i])
#        print("th dim =", th.ndim)
#        fig=plt.figure(figsize=(8,6))
#        fig.patch.set_color('white')
#        plt.plot(th, precision[1:], label="Precision",linewidth=5)
#        plt.plot(th, recall[1:], label="Recall",linewidth=5)
#        plt.title('Precision and recall for different threshold values')
#        plt.xlabel('Threshold')
#        plt.ylabel('Precision/Recall')
#        plt.legend()
#        plt.show()
#        plt.grid()
#        plt.savefig('./gbdt_results_9var/'+output+'/'+filepath+'/PRC_%i.png'%(i))



########################Train for each energy bin############################

    threshold=0
    for i in range (len(X_train_binned)):
        print("bin ", i, " ene = ", ene[i])
        bdt_ = GradientBoostingClassifier(max_depth=5, min_samples_leaf=200, min_samples_split=10,  n_estimators=100, learning_rate=1.0)
        #print("X_train_binned[i] = ", X_train_binned[i])
        #print("y_train_binned[i]", y_train_binned[i])
        bdt_.fit(X_train_binned[i], y_train_binned[i])  

        print("Accuracy score (training): {0:.3f}".format(bdt_.score(X_train_binned[i], y_train_binned[i])))
        print("Accuracy score (test): {0:.3f}".format(bdt_.score(X_test_binned[i], y_test_binned[i])))
        compare_train_test(bdt_, np.array(X_train_binned[i]), np.array(y_train_binned[i]), np.array(X_test_binned[i]), np.array(y_test_binned[i]), output, filepath, "_bin%i"%(i))
#        #y_testbinned[:, 1] can tell us it's signal or not 
#        y_testbinned = label_binarize(y_test_binned[i], classes = [0, 1, 2])
#        decision_binned = bdt.decision_function(X_test_binned[i])
#        fpr, tpr, thresholds = roc_curve(y_testbinned[:, 1], decision_binned[:, 1])
#        for ii in range(len(thresholds)-1):
#            if thresholds[ii]>-0.1 and thresholds[ii+1]<=-0.1 :
#                print("th = ", thresholds[ii], thresholds[ii+1], "eff = ", tpr[ii+1], "fake = ", fpr[ii+1]) 
#
        #The samples for the sig, bkg0, bkg1
        decision_sig = bdt_.decision_function(X_test_sig_binned[i])
        decision_bkg0 = bdt_.decision_function(X_test_bkg0_binned[i])
        decision_bkg1 = bdt_.decision_function(X_test_bkg1_binned[i])
#        print("decision_sig size ", len(decision_sig))
#        print("decision_bkg0 size ", len(decision_bkg0))
#        print("decision_bkg1 size ", len(decision_bkg1))
#        print("decision_sig ", decision_sig)
#        print("decision_bkg0 size ", len(decision_bkg0))
#        print("decision_bkg1 size ", len(decision_bkg1))
#
#        # 1 denotes whether it's signal
        dsig_ = decision_sig[:, 1]
        dbkg0_ = decision_bkg0[:, 1] 
        dbkg1_ = decision_bkg1[:, 1] 
        
        nSigAll = 0 
        nBkg0All = 0 
        nBkg1All = 0 
        nSigPassed = 0 
        nBkg0Passed = 0 
        nBkg1Passed = 0 
        for j in range(len(dsig_)):
            nSigAll +=1
            if dsig_[j] >= threshold :
                sigEff.Fill(True, X_test_sig_ene_binned[i][j][0]/1000.)
                nSigPassed +=1
            else:
                sigEff.Fill(False, X_test_sig_ene_binned[i][j][0]/1000.)
#            
        for j in range(len(dbkg0_)):
            nBkg0All +=1
            if dbkg0_[j] >= threshold :
                bkg0Eff.Fill(True, X_test_bkg0_ene_binned[i][j][0]/1000.)
                nBkg0Passed +=1
            else:
                bkg0Eff.Fill(False, X_test_bkg0_ene_binned[i][j][0]/1000.)
#
        for j in range(len(dbkg1_)):
            nBkg1All +=1
            if dbkg1_[j] >= threshold :
                bkg1Eff.Fill(True, X_test_bkg1_ene_binned[i][j][0]/1000.)
                nBkg1Passed +=1
            else:
                bkg1Eff.Fill(False, X_test_bkg1_ene_binned[i][j][0]/1000.)

        print("(nSigAll, nBkg0All, nBkg1All) = ", nSigAll, nBkg0All, nBkg1All)
        print("(nSigPassed, nBkg0Passed, nBkg1Passed) = ", nSigPassed, nBkg0Passed, nBkg1Passed)

    #Get the score for test sig, bkg0 and bkg1 samples, respectively
#    decision_signal0 = bdt.decision_function(test_signal0)
#    decision_background0 = bdt.decision_function(test_background0)
#    decision_background1 = bdt.decision_function(test_background1)
#    dsig = decision_signal0[:, 1]
#    dbkg0 = decision_background0[:, 1]
#    dbkg1 = decision_background1[:, 1]
#    for j in range(len(dsig)):
#        if dsig[j] >= threshold :
#            sigEff.Fill(True, test_signal0_ene[j][0]/1000.)
#        else:
#            sigEff.Fill(False, test_signal0_ene[j][0]/1000.)
#    for j in range(len(dbkg0)):
#        if dbkg0[j] >= threshold :
#            bkg0Eff.Fill(True, test_background0_ene[j][0]/1000.)
#        else:
#            bkg0Eff.Fill(False, test_background0_ene[j][0]/1000.)
#    for j in range(len(dbkg1)):
#        if dbkg1[j] >= threshold :
#            bkg1Eff.Fill(True, test_background1_ene[j][0]/1000.)
#        else:
#            bkg1Eff.Fill(False, test_background1_ene[j][0]/1000.)
 

    sigEff.SetFillStyle(3004);
    sigEff.SetFillColor(ROOT.kRed);
    sigEff.SetMarkerColor(ROOT.kRed);
    sigEff.SetLineColor(ROOT.kRed);
    sigEff.SetMarkerStyle(20);
   
    bkg0Eff.SetFillStyle(3005);
    bkg0Eff.SetFillColor(ROOT.kBlue);
    bkg0Eff.SetMarkerColor(ROOT.kBlue);
    bkg0Eff.SetLineColor(ROOT.kBlue);
    bkg0Eff.SetMarkerStyle(20);
   
    bkg1Eff.SetFillStyle(3005);
    bkg1Eff.SetFillColor(ROOT.kGreen);
    bkg1Eff.SetMarkerColor(ROOT.kGreen);
    bkg1Eff.SetLineColor(ROOT.kGreen);
    bkg1Eff.SetMarkerStyle(20);
   
    sigEff.SetTitle(";E_{T} [GeV];Efficiency"); 

    canvas = ROOT.TCanvas("c1", "", 700, 600) 
    canvas.SetFillStyle(1001);
    canvas.cd() 
    #sigEff.Draw("A4") 
    #bkg0Eff.Draw("same4") 
    #bkg1Eff.Draw("same4")
    sigEff.Draw("") 
    bkg0Eff.Draw("same") 
    bkg1Eff.Draw("same")
   
    ROOT.gPad.Update();
    graph = sigEff.GetPaintedGraph();
    graph.SetMinimum(0);
    graph.SetMaximum(1.3);
    #graph.GetYaxis().SetTitleOffset(1.5) 
    #graph.GetXaxis().SetTitleOffset(1.2) 
    ROOT.gPad.Update();

    leg = ROOT.TLegend(0.7, 0.75, 0.95, 0.9);
    if "axion1" in output:
        leg.AddEntry(sigEff, "a#rightarrow #gamma #gamma","APL");
    elif "axion2" in output: 
        leg.AddEntry(sigEff, "a#rightarrow #pi^{0} #pi^{0} #pi^{0}","APL");
    elif "scalar1" in output: 
        leg.AddEntry(sigEff, "s#rightarrow #pi^{0} #pi^{0}","APL");
    leg.AddEntry(bkg0Eff, "#gamma","APL");
    leg.AddEntry(bkg1Eff, "#pi^{0}","APL");
    leg.SetLineStyle(0);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.Draw();

    canvas.Show()
    #canvas.SaveAs("Eff.pdf")
    canvas.SaveAs("./gbdt_results_9var/"+output+"/"+filepath+"/Eff.png")
    

    #fig=plt.figure(figsize=(8,6))
    #fig.patch.set_color('white')
    #plt.plot(ene, sigEff, lw=1.2, label='Signal', color="r")
    #plt.plot(ene, bkg0Eff, lw=1.2, label='Bkg0', color="b")
    #plt.plot(ene, bkg1Eff, lw=1.2, label='Bkg1', color="b")
    #plt.xlabel('E_{T} [GeV]')
    #plt.ylabel('Efficiency')
    #plt.legend(loc = "lower left")
    #plt.grid()







# Comparing train and test results for signal vs. rest
def compare_train_test(clf, X_train, y_train, X_test, y_test, output, savepath, label="", bins=50):

    decisions = []
    print("label = ", label)
    for X,y in ((X_train, y_train), (X_test, y_test)):
        print("X=", X)
        print("y=", y)
        #d1 = clf.decision_function(X[y==1]).ravel()
        #d2 = clf.decision_function(X[y<0.5]).ravel()
        #d3 = clf.decision_function(X[y>1.5]).ravel()
        d1 = clf.decision_function(X[y==1])
        d2 = clf.decision_function(X[y<0.5])
        d3 = clf.decision_function(X[y>1.5])
        #d1 = clf.decision_function(X[y==0])
        #d2 = clf.decision_function(X[y==1])
        #d3 = clf.decision_function(X[y==2])
        decisions += [d1[:, 1], d2[:, 1], d3[:, 1]]
        #decisions += [d1[:, 0], d2[:, 0], d3[:, 0]]

    #low = min(np.min(d) for d in decisions)
    #high = max(np.max(d) for d in decisions)
    low = -15 
    high = 15 
    low_high = (low, high)
    fig=plt.figure(figsize=(8,5.5))
    fig.patch.set_color('white')
    plt.hist(decisions[0], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density = True, label='Signal (train)')
    plt.hist(decisions[1], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density = True, label='Background0 (train)')
    plt.hist(decisions[2], color='g', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density = True, label='background1 (train)')
 
    sigScore_pass = decisions[0]>=0
    sigScore_failed = decisions[0]<0
    print("Signal with score>=0", sigScore_pass.sum()) 
    print("Signal with score<0", sigScore_failed.sum()) 

    hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, density=True)
    scale = len(decisions[3])/sum(hist)
    err = np.sqrt(hist*scale)/scale

    width = (bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='Signal (test)')

    hist, bins = np.histogram(decisions[4], bins=bins, range=low_high, density=True)
    scale = len(decisions[4])/sum(hist)
    err = np.sqrt(hist*scale)/scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='Background0 (test)')

    hist, bins = np.histogram(decisions[5], bins=bins, range=low_high, density=True)
    scale = len(decisions[5])/sum(hist)
    err = np.sqrt(hist*scale)/scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='g', label='Background1 (test)')
  
  
    plt.xlabel("BDT score")
    plt.ylabel("Normalized Unit")
    plt.legend(loc='best')
    plt.savefig("./gbdt_results_9var/"+output+"/"+savepath+"/BDTscore"+label+".png")
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

def plot_inputs(outdir, vars, sig, sig_w, bkg, bkg_w, bkg2, bkg2_w):
    for n, var in enumerate(vars):
        _, bins = np.histogram(np.concatenate((sig[:, n], bkg[:, n],bkg2[:, n])), bins=40)
        sns.distplot(sig[:, n], hist_kws={'weights': sig_w}, bins=bins, kde=False, norm_hist=True, label='signal')
        sns.distplot(bkg[:, n], hist_kws={'weights': bkg_w}, bins=bins, kde=False, norm_hist=True, label='background0')
        sns.distplot(bkg2[:, n], hist_kws={'weights': bkg2_w}, bins=bins, kde=False, norm_hist=True, label='Background1')
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
