from ROOT import *
#import ROOT
import numpy as np

nbins0x = 4; 
nbins1x = 4;
nbins2x = 16;
nbins3x = 16;

nbins0y = 16;
nbins1y = 128;
nbins2y = 16;
nbins3y = 8;

lvl0 = nbins0x * nbins0y;
lvl1 = nbins1x * nbins1y;
lvl2 = nbins2x * nbins2y;
lvl3 = nbins3x * nbins3y;

#The cell index (0-503) is column major
# 5 * *
# 4 * *
# 3 * *
# 2 * * *
# 1 * * *
# 0 * * * 
#   0 1 2

# x(phi), y(eta), z
#Layer1 : 3*96
#Layer2 : 12*12 
#Layer3 : 12*6 
#Total: 504
#cell with index 504, 505, 506 are for overflow

#
#Function to get the cell index
#
def get_z(myindex):
    if myindex == 960:
        return 0
    elif myindex == 961:
        return 1
    elif myindex == 962:
        return 2
    elif myindex == 963:
        return 3 
    elif (myindex >= lvl0 + lvl1 + lvl2):
        return 3 
    elif (myindex >= lvl0 + lvl1):
        return 2 
    elif (myindex >= lvl0):
        return 1 
    else:
        return 0
    pass

def get_y(myindex,zbin):
    if (myindex >=960):
        return -1
    elif (zbin==0):
        return myindex % nbins0y
    elif (zbin==1):
        return (myindex - lvl0) % nbins1y
    elif (zbin==2):
        return (myindex - lvl0 - lvl1) % nbins2y
    else:
        return (myindex - lvl0 - lvl1 - lvl2) % nbins3y
    pass

def get_x(myindex,ybin,zbin):
    if (myindex >=960):
        return -1
    elif (zbin==0):
        return (myindex - ybin)/nbins0y
    elif (zbin==1):
        return (myindex - lvl0 - ybin)/nbins1y
    elif (zbin==2):
        return (myindex - lvl0 - lvl1 - ybin)/nbins2y
    else:
        return (myindex - lvl0 - lvl1 - lvl2 - ybin)/nbins3y
    pass


def setHistStyle(hist,color,marker):
    hist.SetMarkerStyle(marker);
    hist.SetMarkerSize(1.0);
#    hist.SetLineWidth(1);
    hist.SetLineColor(color);
    hist.SetMarkerColor(color);

def setYRange(hists,yMinScale = 0.5, yMaxScale = 2.0):
    if( len(hists) < 1):
        return

    min = hists[0].GetBinContent(1);
    max = hists[0].GetBinContent(1);

    #Find the min and max
    for hist in hists:
        binmin = hist.GetMinimumBin();
        binmax = hist.GetMaximumBin();
        binMinContent = hist.GetBinContent(binmin);
        binMaxContent = hist.GetBinContent(binmax);
        if (binMinContent < min):
            min = binMinContent
        if (binMaxContent > max):
            max = binMaxContent;
    for hist in hists:
        hist.GetYaxis().SetRangeUser(min*yMinScale, max* yMaxScale);



gStyle.SetOptFit(0011);
gStyle.SetOptStat(0000);
gStyle.SetPadLeftMargin(0.12);
gStyle.SetPadRightMargin(0.05);
gStyle.SetPadTopMargin(0.1);
gStyle.SetPadBottomMargin(0.15);

histDict = {}

inputDir="/nfs/dust/atlas/user/xiaocong/photonJet/generation/output/08-04-21-20/root/"

inputFiles = [ "axion1_40-250GeV_100k.root", "axion2_40-250GeV_100k.root", "scalar1_40-250GeV_100k.root", "gamma_40-250GeV_100k.root", "pi0_40-250GeV_100k.root" ]
#inputFiles = [ "gamma_40-250GeV_100k.root", "pi0_40-250GeV_100k.root" ]

outputDir="/nfs/dust/atlas/user/xiaocong/photonJet/generation/plots_test/"

#suffix="axion1_"
#suffix="electron_"
#suffix="100GeV_"
suffix="40-250GeV_"
#suffix="1TeV_"

Fraction_in_thirdlayer = []
Fraction_not_in = []
PreSam_lateral = []
Front_lateral = []
Middle_lateral = []
Back_lateral = []
PreSam_lateral_width = []
Front_lateral_width = []
Middle_lateral_width = []
Back_lateral_width = []
Shower_Depth = []
Shower_Depth_width = []

legs = []
for i in range(12):
    legs.append(TLegend(.45,.65,.7,.9))
    legs[i].SetBorderSize(0)
    legs[i].SetFillColor(0)
    legs[i].SetFillStyle(0)
    legs[i].SetTextFont(42)
    legs[i].SetTextSize(0.035)
colors = [ 1, 2, 4, 6, 8, 46, 30, 41, 9 ] 
markers = [20, 21, 22, 23, 24, 25, 26 ]

tfile_open = [];

for iFile in range(len(inputFiles)):
        inputFile= inputFiles[iFile]
	outputFile = inputFile.replace(".root", "")
        print("outputFile = ", outputFile)

        tfile_open.append(TFile(inputDir + inputFile))
        

	#pion-10GeV-1k.root
	#electrons-10GeV-5k.root 
	#photons-10GeV-1k.root
	mytree = tfile_open[iFile].Get("fancy_tree")

	zsegmentation = TH1F("","",4,np.array([1464.148, 1500, 1590.684, 1928.113, 1970.292]))
	zsegmentation.GetXaxis().SetTitle("Layer");
	zsegmentation.GetYaxis().SetTitle("Energy [GeV]");
	setHistStyle(zsegmentation,colors[iFile],markers[iFile])

	#PreSampling layer: 4x16 (phixeta)
	sampling0_eta = TH2F("","",4,-0.1*2,0.1*2, 16,-0.025*8, 0.025*8)
	sampling0_eta.GetXaxis().SetTitle("Phi");
	sampling0_eta.GetYaxis().SetTitle("Eta");
	#Layer1: 4x128 (phixeta)
	sampling1_eta = TH2F("","",4,-0.098*2,0.098*2, 128,-0.003125*64, 0.003125*64)
	sampling1_eta.GetXaxis().SetTitle("Phi");
	sampling1_eta.GetYaxis().SetTitle("Eta");
        if ("pi0" in inputFile) or ("gamma" in inputFile):	
        	sampling1_eta.GetYaxis().SetRangeUser(-0.02, 0.02);
        else:
        	sampling1_eta.GetYaxis().SetRangeUser(-0.05, 0.05);
	#Layer2: 16x16 (phixeta)
	sampling2_eta = TH2F("","",16,-0.0245*8, 0.0245*8, 16,-0.025*8, 0.025*8)
	sampling2_eta.GetXaxis().SetTitle("Phi");
	sampling2_eta.GetYaxis().SetTitle("Eta");
	#Layer3: 16x8 (phixeta)
	sampling3_eta = TH2F("","",16,-0.0245*8, 0.0245*8, 8,-0.05*4, 0.05*4)
	sampling3_eta.GetXaxis().SetTitle("Phi");
	sampling3_eta.GetYaxis().SetTitle("Eta");

        #Plots will be superimposed on them
        #Fraction_in_thirdlayer.append(TH1F("Fraction_in_thirdlayer_{0}".format(outputFile),"Fraction_in_thirdlayer_{0}".format(outputFile),100,0,0.01))
        Fraction_in_thirdlayer.append(TH1F("Fraction_in_thirdlayer_{0}".format(outputFile),"",100,0,1))
        Fraction_in_thirdlayer[iFile].GetXaxis().SetTitle("Fraction");
        Fraction_in_thirdlayer[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(Fraction_in_thirdlayer[iFile],colors[iFile],markers[iFile])
        Fraction_not_in.append(TH1F("Fraction_not_in_{0}".format(outputFile),"",100,0,0.01))
        Fraction_not_in[iFile].GetXaxis().SetTitle("Fraction");
        Fraction_not_in[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(Fraction_not_in[iFile],colors[iFile],markers[iFile])
            
        PreSam_lateral.append(TH1F("PreSampling_lateral_{0}".format(outputFile),"",30,-0.2,0.2))
        PreSam_lateral[iFile].GetXaxis().SetTitle("Energy weighted size");
        PreSam_lateral[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(PreSam_lateral[iFile],colors[iFile],markers[iFile])
        PreSam_lateral_width.append(TH1F("PreSampling_lateral_width_{0}".format(outputFile),"",50,0,0.2))
        PreSam_lateral_width[iFile].GetXaxis().SetTitle("Sigma of energy weighted size");
        PreSam_lateral_width[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(PreSam_lateral_width[iFile],colors[iFile],markers[iFile])

        Front_lateral.append(TH1F("Front_lateral_{0}".format(outputFile),"",30,-0.2,0.2))
        Front_lateral[iFile].GetXaxis().SetTitle("Energy weighted size");
        Front_lateral[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(Front_lateral[iFile],colors[iFile],markers[iFile])
        Front_lateral_width.append(TH1F("Front_lateral_width_{0}".format(outputFile),"",25,0,0.1))
        Front_lateral_width[iFile].GetXaxis().SetTitle("Sigma of energy weighted size");
        Front_lateral_width[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(Front_lateral_width[iFile],colors[iFile],markers[iFile])

        Middle_lateral.append(TH1F("Middle_lateral_{0}".format(outputFile),"",120,-0.2,0.2))
        Middle_lateral[iFile].GetXaxis().SetTitle("Energy weighted size");
        Middle_lateral[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(Middle_lateral[iFile],colors[iFile],markers[iFile])
        Middle_lateral_width.append(TH1F("Middle_lateral_width_{0}".format(outputFile),"",100,0,0.05))
        Middle_lateral_width[iFile].GetXaxis().SetTitle("Sigma of energy weighted size");
        Middle_lateral_width[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(Middle_lateral_width[iFile],colors[iFile],markers[iFile])

        Back_lateral.append(TH1F("Back_lateral_{0}".format(outputFile),"",120,-0.2,0.2))
        Back_lateral[iFile].GetXaxis().SetTitle("Energy weighted size");
        Back_lateral[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(Back_lateral[iFile],colors[iFile],markers[iFile])
        Back_lateral_width.append(TH1F("Back_lateral_width_{0}".format(outputFile),"",200,0,0.1))
        Back_lateral_width[iFile].GetXaxis().SetTitle("Sigma of energy weighted size");
        Back_lateral_width[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(Back_lateral_width[iFile],colors[iFile],markers[iFile])
            
        Shower_Depth.append(TH1F("Shower_Depth_{0}".format(outputFile),"",100,0,2.5))
        Shower_Depth[iFile].GetXaxis().SetTitle("Energy weighted depth");
        Shower_Depth[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(Shower_Depth[iFile],colors[iFile],markers[iFile])
        Shower_Depth_width.append(TH1F("Shower_Depth_width_{0}".format(outputFile),"",100,0,1.0))
        Shower_Depth_width[iFile].GetXaxis().SetTitle("Sigma of energy weighted depth");
        Shower_Depth_width[iFile].GetYaxis().SetTitle("Entries");
        setHistStyle(Shower_Depth_width[iFile],colors[iFile],markers[iFile])
   
        # Add legend entries
        legs[0].AddEntry(Fraction_in_thirdlayer[iFile], outputFile, "l") 
        legs[1].AddEntry(Fraction_not_in[iFile], outputFile, "l") 
        legs[2].AddEntry(PreSam_lateral[iFile], outputFile, "l") 
        legs[3].AddEntry(Front_lateral[iFile], outputFile, "l") 
        legs[4].AddEntry(Middle_lateral[iFile], outputFile, "l") 
        legs[5].AddEntry(Back_lateral[iFile], outputFile, "l") 
        legs[6].AddEntry(PreSam_lateral_width[iFile], outputFile, "l") 
        legs[7].AddEntry(Front_lateral_width[iFile], outputFile, "l") 
        legs[8].AddEntry(Middle_lateral_width[iFile], outputFile, "l") 
        legs[9].AddEntry(Back_lateral_width[iFile], outputFile, "l") 
        legs[10].AddEntry(Shower_Depth[iFile], outputFile, "l") 
        legs[11].AddEntry(Shower_Depth_width[iFile], outputFile, "l") 
	
	#Loop over all events
        for i in range(min(100,mytree.GetEntries())):
        #for i in range(mytree.GetEntries()):
	    mytree.GetEntry(i)
	    if (i%100==0):
	        print i,mytree.GetEntries()
	        pass
	    y = "energy"
	    exec("%s = %s" % (y,"mytree.cell_0"))
	    total_energy = 0.
	    pre_energy = 0.
	    front_energy = 0.
	    middle_energy = 0.
	    back_energy = 0.
	    not_in = 0.
	    # Energy weighted depth
	    lateral_depth = 0.
	    lateral_depth2 = 0.
	    # Energy weighted X in first, second and third layers
	    pre_layer_X = 0.
	    pre_layer_X2 = 0.
	    first_layer_X = 0.
	    first_layer_X2 = 0.
	    second_layer_X = 0.
	    second_layer_X2 = 0.
	    third_layer_X = 0.
	    third_layer_X2 = 0.
	    #Loop over all cells in the event
	    for j in range(964):
	        exec("%s = %s" % (y,"mytree.cell_"+str(j)))
	        xbin = get_x(j,get_y(j,get_z(j)),get_z(j))
	        ybin = get_y(j,get_z(j))
	        zbin = get_z(j)
	        zsegmentation.Fill(zsegmentation.GetXaxis().GetBinCenter(zbin+1),energy)
	        zvalue = zsegmentation.GetXaxis().GetBinCenter(zbin+1)
	        yvalue = 0.;
	        xvalue = 0.;
	        total_energy+=energy
	        lateral_depth+=zbin*energy
	        lateral_depth2+=zbin*zbin*energy
	        if (xbin < 0 or ybin < 0):
	            not_in+=energy
	            pass
	        if (zbin==0):
	            sampling0_eta.Fill(sampling0_eta.GetXaxis().GetBinCenter(xbin+1),sampling0_eta.GetYaxis().GetBinCenter(ybin+1),energy)
	            xvalue = sampling0_eta.GetXaxis().GetBinCenter(xbin+1)
	            yvalue = sampling0_eta.GetYaxis().GetBinCenter(ybin+1)
	            pre_layer_X += xvalue*energy
	            pre_layer_X2 += xvalue*xvalue*energy
	            pre_energy+=energy
	        elif (zbin==1):
	            sampling1_eta.Fill(sampling1_eta.GetXaxis().GetBinCenter(xbin+1),sampling1_eta.GetYaxis().GetBinCenter(ybin+1),energy)
	            xvalue = sampling1_eta.GetXaxis().GetBinCenter(xbin+1)
	            yvalue = sampling1_eta.GetYaxis().GetBinCenter(ybin+1)
	            first_layer_X += xvalue*energy
	            first_layer_X2 += xvalue*xvalue*energy
	            front_energy+=energy
	        elif (zbin==2):
	            sampling2_eta.Fill(sampling2_eta.GetXaxis().GetBinCenter(xbin+1),sampling2_eta.GetYaxis().GetBinCenter(ybin+1),energy)
	            xvalue = sampling2_eta.GetXaxis().GetBinCenter(xbin+1)
	            yvalue = sampling2_eta.GetYaxis().GetBinCenter(ybin+1)
	            second_layer_X += xvalue*energy
	            second_layer_X2 += xvalue*xvalue*energy
	            middle_energy+=energy
	        else:
	            sampling3_eta.Fill(sampling3_eta.GetXaxis().GetBinCenter(xbin+1),sampling3_eta.GetYaxis().GetBinCenter(ybin+1),energy)
	            xvalue = sampling3_eta.GetXaxis().GetBinCenter(xbin+1)
	            yvalue = sampling3_eta.GetYaxis().GetBinCenter(ybin+1)
	            third_layer_X += xvalue*energy
	            third_layer_X2 += xvalue*xvalue*energy
	            back_energy+=energy
	            pass
	        pass
	    Fraction_in_thirdlayer[iFile].Fill(back_energy/total_energy)
	    Fraction_not_in[iFile].Fill(not_in/total_energy)
	    Shower_Depth[iFile].Fill(lateral_depth/total_energy)

	    if (pre_energy > 0):
	        PreSam_lateral[iFile].Fill(pre_layer_X/pre_energy)
                if pre_layer_X2/pre_energy - (pre_layer_X/pre_energy)**2 > 0:
			PreSam_lateral_width[iFile].Fill((pre_layer_X2/pre_energy - (pre_layer_X/pre_energy)**2)**0.5)
		pass
	    if (front_energy > 0):
	        Front_lateral[iFile].Fill(first_layer_X/front_energy)
	        if first_layer_X2/front_energy - (first_layer_X/front_energy)**2 > 0 : 
			Front_lateral_width[iFile].Fill((first_layer_X2/front_energy - (first_layer_X/front_energy)**2)**0.5)
	        pass
	    if (middle_energy > 0):
	        Middle_lateral[iFile].Fill(second_layer_X/middle_energy)
		if second_layer_X2/middle_energy - (second_layer_X/middle_energy)**2 > 0 : 
		        Middle_lateral_width[iFile].Fill(((second_layer_X2/middle_energy) - (second_layer_X/middle_energy)**2)**0.5)
	        pass
	    if (back_energy > 0):
	        Back_lateral[iFile].Fill(third_layer_X/back_energy)
                if third_layer_X2/back_energy - (third_layer_X/back_energy)**2 > 0 :	
		        Back_lateral_width[iFile].Fill(((third_layer_X2/back_energy) - (third_layer_X/back_energy)**2)**0.5)
	        pass
            if lateral_depth2/total_energy - (lateral_depth/total_energy)**2 > 0 : 
		    Shower_Depth_width[iFile].Fill((lateral_depth2/total_energy - (lateral_depth/total_energy)**2)**0.5)
	    pass
	
        try :
            hdict =  histDict[outputFile]
        except :
            histDict[outputFile] = []
            hdict = histDict[outputFile]
 
        histDict[outputFile] = [ Fraction_in_thirdlayer[iFile], Fraction_not_in[iFile],  PreSam_lateral[iFile], Front_lateral[iFile], Middle_lateral[iFile],  Back_lateral[iFile], PreSam_lateral_width[iFile], Front_lateral_width[iFile], Middle_lateral_width[iFile], Back_lateral_width[iFile], Shower_Depth[iFile], Shower_Depth_width[iFile] ] 
        
        c1Name = "c1_{0}".format(outputFile)
	
        # canvas1
        print(c1Name) 
	c1 = TCanvas(c1Name,"c1",1600,1000)
	c1.Divide(2,2)
	c1.cd(1);
	zsegmentation.Draw()
	c1.cd(2);
	gPad.SetLogz()
	sampling1_eta.Draw("colz")
	c1.cd(3);
	gPad.SetLogz()
	sampling2_eta.Draw("colz")
	c1.cd(4);
	gPad.SetLogz()
	sampling3_eta.Draw("colz")
#	for i in range(1,sampling3_eta.GetNbinsX()+1):
#	    for j in range(1,sampling3_eta.GetNbinsY()+1):
#	        print i,j,sampling3_eta.GetBinContent(i,j)
#	        pass
#	    pass
	c1.Print(outputDir + outputFile + "_tot_zprofile_and_xy.pdf")
        

#print(histDict)

setYRange(Fraction_in_thirdlayer, 0.5, 2.0)
setYRange(Fraction_not_in, 0.5, 2.0)

setYRange(PreSam_lateral, 0.5, 2.0)
setYRange(Front_lateral, 0.5, 2.0)
setYRange(Middle_lateral, 0.5, 2.0)
setYRange(Back_lateral, 0.5, 2.0)
setYRange(PreSam_lateral_width, 0.5, 2.0)
setYRange(Front_lateral_width, 0.5, 2.0)
setYRange(Middle_lateral_width, 0.5, 2.0)
setYRange(Back_lateral_width, 0.5, 2.0) 
setYRange(Shower_Depth, 0.5, 2.0)
setYRange(Shower_Depth_width, 0.5, 2.0)


# canvas2
c2 = TCanvas("c2_all","c2",800,400)
c2.Divide(2,1)
c2.cd(1)
for i in range(len(Fraction_in_thirdlayer)):
    Fraction_in_thirdlayer[i].Draw("same")
legs[0].Draw()
c2.cd(2)
for hist in Fraction_not_in:
    hist.Draw("same")
legs[1].Draw()

#canvas3
c3 = TCanvas("c3_all","c3",1600,400)
c3.Divide(4,1)
c3.cd(1)
for hist in PreSam_lateral:
   hist.Draw("same")
legs[2].Draw()
c3.cd(2)
for hist in Front_lateral:
   hist.Draw("same")
legs[3].Draw()
c3.cd(3)
for hist in Middle_lateral:
   hist.Draw("same")
legs[4].Draw()
c3.cd(4)
for hist in Back_lateral:
   hist.Draw("same")
legs[5].Draw()

# canvas4
c4 = TCanvas("c4_all","c4",1600,400)
c4.Divide(4,1)
c4.cd(1)
for hist in PreSam_lateral_width:
   hist.Draw("same")
legs[6].Draw()
c4.cd(2)
for hist in Front_lateral_width:
   hist.Draw("same")
legs[7].Draw()
c4.cd(3)
for hist in Middle_lateral_width:
   hist.Draw("same")
legs[8].Draw()
c4.cd(4)
for hist in Back_lateral_width:
   hist.Draw("same")
legs[9].Draw()


# canvas5
c5 = TCanvas("c5_all","c5",800,400)
c5.Divide(2,1)
c5.cd(1)
for hist in Shower_Depth:
    hist.Draw("same")
legs[10].Draw()
c5.cd(2)
for hist in Shower_Depth_width:
    hist.Draw("same")
legs[11].Draw()


c2.Print(outputDir + suffix + "Fraction.pdf")
c3.Print(outputDir + suffix + "Lateral.pdf")
c4.Print(outputDir + suffix + "Lateral_width.pdf")
c5.Print(outputDir + suffix + "Shower_Depth.pdf")
