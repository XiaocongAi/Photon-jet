#--------------------------------------------------------------------------------
# Usage example:
# python splitROOT.py old_file new_file1  new_file2 0.7 "fancy_tree" 
# ---------------------------------------------------------------------------------

from ROOT import TFile, gDirectory, TTreeReader, TTree
import numpy as np

import sys

# Import the input parameters
arg_list = sys.argv
old_filename = arg_list[1]
new_filename1 = arg_list[2]
new_filename2 = arg_list[3]
entries_file1 = arg_list[4]
tree_name = arg_list[5]


input_file = TFile(old_filename )
input_tree = input_file.Get(tree_name)


totalEntries = input_tree.GetEntries()
print("totalEntries=", totalEntries)

newfile1 = TFile(new_filename1, "recreate")
newtree1 = input_tree.CloneTree(0)
start_counter=0
for je in xrange(totalEntries):
    input_tree.GetEntry( start_counter + je )
    if je< np.int(entries_file1) :
        newtree1.Fill()
newtree1.Write()
newfile1.Close()


newfile2 = TFile(new_filename2, "recreate")
newtree2 = input_tree.CloneTree(0)
start_counter=0
for je in xrange(totalEntries):
    input_tree.GetEntry( start_counter + je )
    if je>= np.int(entries_file1) :
        newtree2.Fill()
newtree2.Write()
newfile2.Close()

