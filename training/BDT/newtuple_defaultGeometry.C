int nbins1x=3;
int nbins2x=12;
int nbins3x=12;
int nbins1y=96;
int nbins2y=12;
int nbins3y=6;
int lvl1=nbins1x*nbins1y;
int lvl2=nbins2x*nbins2y;
int lvl3=nbins3x*nbins3y;

int get_z(int index){
        if(index==504) return 0;
        else if(index==505) return 1;
        else if(index==506) return 2;
        else if(index>=lvl1+lvl2) return 2;
        else if(index>=lvl1) return 1;
        else return 0;
        }

int get_y(int index, int zbin){
        if(index>=504) return -1;
        else if(zbin==0) return index % nbins1y;
        else if(zbin==1) return index % nbins2y;
        else return index % nbins3y;
        }

int get_x(int index, int ybin, int zbin){
        if(index>=504) return -1;
        else if(zbin==0) return (index-ybin) / nbins1y;
        else if(zbin==1) return (index-lvl1-ybin) / nbins2y;
        else return (index-lvl1-lvl2-ybin) / nbins3y;
        }

int newntuple(TString inname="axion1_100GeV_20k.root", TString output_prefix="new"){
        TString treename="fancy_tree";
        TString outname=output_prefix+"_"+inname;
        TFile* fin=new TFile(inname);
        TTree* tin=(TTree*)fin->Get(treename);
        TFile* fout=new TFile(outname,"recreate");
        TTree* tout=new TTree(treename,treename);
        TH2D* sampling1=new TH2D("","",3,-240,240,int(480/5),-240,240);
        TH2D* sampling2=new TH2D("","",int(480/40),-240,240,int(480/40),-240,240);
        TH2D* sampling3=new TH2D("","",int(480/40),-240,240,int(480/80),-240,240);
        double cells[507];
        float total_e(0.),firstlayer_e(0.),secondlayer_e(0.),thirdlayer_e(0.),lateral_depth(0.),lateral_depth2(0.),firstlayer_x(0.),firstlayer_x2(0.),secondlayer_x(0.),secondlayer_x2(0.);
        float frac_first(0.),frac_second(0.),frac_third(0.);
        float shower_depth_gamma(0.),second_lateral_width_gamma(0.),first_lateral_width_gamma(0.),shower_depth_width_gamma(0.);

        tout->Branch("total_e",&total_e,"total_e/F");
        tout->Branch("firstlayer_e",&firstlayer_e,"firstlayer_e/F");
        tout->Branch("secondlayer_e",&secondlayer_e,"secondlayer_e/F");
        tout->Branch("thirdlayer_e",&thirdlayer_e,"thirdlayer_e/F");
        tout->Branch("lateral_depth",&lateral_depth,"lateral_depth/F");
        tout->Branch("lateral_depth2",&lateral_depth2,"lateral_depth2/F");
        tout->Branch("firstlayer_x",&firstlayer_x,"firstlayer_x/F");
        tout->Branch("firstlayer_x2",&firstlayer_x2,"firstlayer_x2/F");
        tout->Branch("secondlayer_x",&secondlayer_x,"secondlayer_x/F");
        tout->Branch("secondlayer_x2",&secondlayer_x2,"secondlayer_x2/F");
        tout->Branch("frac_first",&frac_first,"frac_first/F");
        tout->Branch("frac_second",&frac_second,"frac_second/F");
        tout->Branch("frac_third",&frac_third,"frac_third/F");
        tout->Branch("shower_depth_gamma",&shower_depth_gamma,"shower_depth_gamma/F");
        tout->Branch("second_lateral_width_gamma",&second_lateral_width_gamma,"second_lateral_width_gamma/F");
        tout->Branch("first_lateral_width_gamma",&first_lateral_width_gamma,"first_lateral_width_gamma/F");
        tout->Branch("shower_depth_width_gamma",&shower_depth_width_gamma,"shower_depth_width_gamma/F");
        for(int icell=0;icell<507;++icell){
            TString name="cell_";
            name+=icell;
            tin->SetBranchAddress(name,&(cells[icell]));
            }
        for(int ievt=0;ievt<tin->GetEntries();++ievt){
            total_e=0.;
            firstlayer_e=0;
            secondlayer_e=0.;
            thirdlayer_e=0.;
            lateral_depth=0.;
            lateral_depth2=0.;
            firstlayer_x=0.;
            firstlayer_x2=0.;
            secondlayer_x=0.;
            secondlayer_x2=0.;
            tin->GetEntry(ievt);
            for(int icell=0;icell<504;++icell){
                int zbin=get_z(icell);
                int ybin=get_y(icell,zbin);
                int xbin=get_x(icell,ybin,zbin);
                lateral_depth+=zbin * cells[icell];
                lateral_depth2+=zbin*zbin*cells[icell];
                double xvalue=0.;
                double yvalue=0.;
                total_e+=cells[icell];
                if(zbin==2) {
                    thirdlayer_e+=cells[icell];
                    sampling3->Fill(sampling3->GetXaxis()->GetBinCenter(xbin+1),sampling3->GetYaxis()->GetBinCenter(ybin+1),cells[icell]);
                    xvalue=sampling3->GetXaxis()->GetBinCenter(xbin+1);
                    yvalue=sampling3->GetYaxis()->GetBinCenter(ybin+1);

                    }
                if(zbin==1) {
                    secondlayer_e+=cells[icell];
                    sampling2->Fill(sampling2->GetXaxis()->GetBinCenter(xbin+1),sampling2->GetYaxis()->GetBinCenter(ybin+1),cells[icell]);
                    xvalue=sampling2->GetXaxis()->GetBinCenter(xbin+1);
                    yvalue=sampling2->GetYaxis()->GetBinCenter(ybin+1);
                    secondlayer_x+=xvalue*cells[icell];
                    secondlayer_x2+=xvalue*xvalue*cells[icell];
                    }
                if(zbin==0) {
                    firstlayer_e+=cells[icell];
                    sampling1->Fill(sampling1->GetXaxis()->GetBinCenter(xbin+1),sampling1->GetYaxis()->GetBinCenter(ybin+1),cells[icell]);
                    xvalue=sampling1->GetXaxis()->GetBinCenter(xbin+1);
                    yvalue=sampling1->GetYaxis()->GetBinCenter(ybin+1);
                    firstlayer_x+=xvalue*cells[icell];
                    firstlayer_x2+=xvalue*xvalue*cells[icell];
                    }
                }
            frac_first=firstlayer_e/total_e;
            frac_second=secondlayer_e/total_e;
            frac_third=thirdlayer_e/total_e;
            shower_depth_gamma=lateral_depth/total_e;
            second_lateral_width_gamma=pow(secondlayer_x2/secondlayer_e-pow((secondlayer_x/secondlayer_e),2),0.5);
            first_lateral_width_gamma=pow(firstlayer_x2/firstlayer_e-pow((firstlayer_x/firstlayer_e),2),0.5);
            shower_depth_width_gamma=pow(lateral_depth2/total_e-pow((lateral_depth/total_e),2),0.5);
            tout->Fill();

            }

        tout->Write();
        fout->Close();
        return 0;
        }
