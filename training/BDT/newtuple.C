#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>


int nbins0x = 4;  
int nbins1x = 4;
int nbins2x = 16; 
int nbins3x = 16; 
int nbins0y = 16; 
int nbins1y = 128;
int nbins2y = 16; 
int nbins3y = 8;
int lvl0 = nbins0x * nbins0y;
int lvl1 = nbins1x * nbins1y;
int lvl2 = nbins2x * nbins2y;
int lvl3 = nbins3x * nbins3y;


int get_z(int myindex){
    if (myindex == 960)
        return 0;
    else if (myindex == 961)
        return 1;
    else if (myindex == 962)
        return 2;
    else if (myindex == 963)
        return 3;
    else if (myindex >= lvl0 + lvl1 + lvl2)
        return 3;
    else if (myindex >= lvl0 + lvl1)
        return 2;
    else if (myindex >= lvl0)
        return 1;
    else
        return 0;
    }

int get_y(int myindex,int zbin){
    if (myindex >=960)
        return -1;
    else if (zbin==0)
        return myindex % nbins0y;
    else if (zbin==1)
        return (myindex - lvl0) % nbins1y;
    else if (zbin==2)
        return (myindex - lvl0 - lvl1) % nbins2y;
    else
        return (myindex - lvl0 - lvl1 - lvl2) % nbins3y;
    } 


int get_x(int myindex, int ybin, int zbin){
    if (myindex >=960)
        return -1;
    else if (zbin==0)
        return (myindex - ybin)/nbins0y;
    else if (zbin==1)
        return (myindex - lvl0 - ybin)/nbins1y;
    else if (zbin==2)
        return (myindex - lvl0 - lvl1 - ybin)/nbins2y;
    else
        return (myindex - lvl0 - lvl1 - lvl2 - ybin)/nbins3y;
    } 

struct Cell{
  Cell(double eta_, double phi_, double energy_, int etaBin_, int phiBin_, int ilayer_):eta(eta_), phi(phi_), energy(energy_), etaBin(etaBin_), phiBin(phiBin_), ilayer(ilayer_){};
 
  double eta = 0;
  double phi = 0;
  double energy = 0;
  int etaBin = 0;
  int phiBin = 0;
  int ilayer =0;
};

double getRatio(const std::vector<Cell>& cells, int window_eta_central, int window_phi_central, int window_eta_all, int window_phi_all){
   // Note this assumes the input cells are already sorted
   const auto& maxCell = cells.front();
   //const auto& maxE = maxCell.energy;
   const auto& maxEtaBin = maxCell.etaBin;
   const auto& maxPhiBin = maxCell.phiBin;
  
   double de =0, ne =0; 
   // Get the denominator
   for(int etaBin = maxEtaBin - (window_eta_central-1)/2; etaBin <= maxEtaBin + (window_eta_central-1)/2; ++etaBin){
       for(int phiBin = maxPhiBin - (window_phi_central-1)/2; phiBin <= maxPhiBin + (window_phi_central-1)/2; ++phiBin){
         auto it = std::find_if(cells.begin(), cells.end(), [&](const Cell& cell){return (cell.etaBin == etaBin and cell.phiBin == phiBin);});
         if(it!=cells.end()){
           de += (*it).energy; 
         }
      } 
   }

   // Get the numerator 
   for(int etaBin = maxEtaBin - (window_eta_all-1)/2; etaBin <= maxEtaBin + (window_eta_all-1)/2; ++etaBin){
       for(int phiBin = maxPhiBin - (window_phi_all-1)/2; phiBin <= maxPhiBin + (window_phi_all-1)/2; ++phiBin){
         auto it = std::find_if(cells.begin(), cells.end(), [&](const Cell& cell){return (cell.etaBin == etaBin and cell.phiBin == phiBin);});
         if(it!=cells.end()){
           ne += (*it).energy; 
         }   
      }   
   } 

   // The ratio 
   return de/ne; 
}

double getWidthAngle(const std::vector<Cell>& cells, int window_eta, int window_phi, bool etaDirection = true){
   // Note this assumes the input cells are already sorted
   const auto& maxCell = cells.front();
   const auto& maxEtaBin = maxCell.etaBin;
   const auto& maxPhiBin = maxCell.phiBin;

   double total_e = 0, energy_weighted_eta =0 , energy_weighted_eta2 =0, energy_weighted_phi =0, energy_weighted_phi2 =0; 
   for(const auto& cell : cells){
     if(cell.etaBin >= (maxEtaBin - (window_eta-1)/2) and cell.etaBin <= (maxEtaBin + (window_eta-1)/2) ){
        if(cell.phiBin >= (maxPhiBin - (window_phi-1)/2) and cell.phiBin <= (maxPhiBin + (window_phi-1)/2) ){
           total_e += cell.energy;  
           energy_weighted_eta += cell.energy*cell.eta;  
           energy_weighted_phi += cell.energy*cell.phi;  
           energy_weighted_eta2 += cell.energy*std::pow(cell.eta,2);  
           energy_weighted_phi2 += cell.energy*std::pow(cell.phi,2);  
        } 
     } 
   } 

   double eta_width = std::sqrt(energy_weighted_eta2/total_e - std::pow(energy_weighted_eta/total_e, 2));
   double phi_width = std::sqrt(energy_weighted_phi2/total_e - std::pow(energy_weighted_phi/total_e, 2));

   if(etaDirection){
     return eta_width;
   }

   return phi_width;
}

// window_eta=3 or 20
double getWidthIndex(const std::vector<Cell>& cells, int window_eta){
   // Note this assumes the input cells are already sorted
   const auto& maxCell = cells.front();
   const auto& maxEtaBin = maxCell.etaBin;
   const auto& maxPhiBin = maxCell.phiBin;
 
   double total_e=0, energy_weighted_eta_index2=0; 
   for(const auto& cell : cells){
     if(cell.etaBin >= (maxEtaBin - (window_eta-1)/2) and cell.etaBin <= (maxEtaBin + (window_eta-1)/2) ){
        if(cell.phiBin == maxPhiBin ){
            total_e += cell.energy;      
            energy_weighted_eta_index2 += cell.energy * std::pow((cell.etaBin - maxEtaBin), 2);      
        } 
     } 
   } 
   
   return std::sqrt(energy_weighted_eta_index2/total_e); 
}

double getFractionSide(const std::vector<Cell>& cells, int window_eta_central, int window_eta_all){

   // Note this assumes the input cells are already sorted
   const auto& maxCell = cells.front();
   const auto& maxEtaBin = maxCell.etaBin;
   const auto& maxPhiBin = maxCell.phiBin;

   double e_central = 0, e_all = 0;
   for(const auto& cell : cells){
     // Only look at the same phi row 
     if(cell.phiBin == maxPhiBin ){
        if(cell.etaBin >= (maxEtaBin - (window_eta_central-1)/2) and cell.etaBin <= (maxEtaBin + (window_eta_central-1)/2) ){
          e_central+= cell.energy; 
        } 
        if(cell.etaBin >= (maxEtaBin - (window_eta_all-1)/2) and cell.etaBin <= (maxEtaBin + (window_eta_all-1)/2) ){
          e_all+= cell.energy; 
        } 
     } 
   } 
  
   return (e_all-e_central)/e_central; 
}


double getERatio(const std::vector<Cell>& cells){
  const auto& maxEnergy = cells.front().energy;    
  const auto& smaxEnergy = cells[1].energy;    
 
  //if(cells.front().phiBin != cells[1].phiBin){
  //  std::cout<<"WARNING: the maximum cell and second maximum cell have different phi bins: " << std::abs(cells.front().phiBin - cells[1].phiBin) << std::endl;
  //} 
 
  return (maxEnergy - smaxEnergy)/(maxEnergy + smaxEnergy);
}


double getDeltaE(const std::vector<Cell>& cells){
  const auto& maxEnergy = cells.front().energy;    
  const auto& maxEtaBin = cells.front().etaBin;    
  const auto& maxPhiBin = cells.front().phiBin;    
  const auto& smaxEnergy = cells[1].energy;    
  const auto& smaxEtaBin = cells[1].etaBin;    
  const auto& smaxPhiBin = cells[1].phiBin;    

  int sEtaBin =(maxEtaBin>smaxEtaBin)? smaxEtaBin : maxEtaBin;
  int eEtaBin =(maxEtaBin>smaxEtaBin)? maxEtaBin : smaxEtaBin;
  int sPhiBin =(maxPhiBin>smaxPhiBin)? smaxPhiBin : maxPhiBin;
  int ePhiBin =(maxPhiBin>smaxPhiBin)? maxPhiBin : smaxPhiBin;
  
  //std::cout<<"Searching minimum in eta = [ " << sEtaBin <<", " << eEtaBin <<"]" << ", phi = [ " << sPhiBin <<", " << ePhiBin <<"]" << std::endl; 

  double minEnergyBetween = std::numeric_limits<double>::max();
  bool found = false;
  for(const auto& cell : cells){
    if(cell.etaBin >= sEtaBin and cell.etaBin <= eEtaBin and cell.phiBin >= sPhiBin and cell.phiBin <= ePhiBin){
      if ( cell.energy < minEnergyBetween){
        minEnergyBetween = cell.energy;
        found = true;
      }        
    }
  }
 
  if(not found){
    std::cout<<"Failed to find the minimum cell between maximum and second maximum" << std::endl;
  }
 
  return (smaxEnergy - minEnergyBetween);

}



int newtuple(TString path = "../../../generation/output/08-04-21-20/root/", TString inname="electron_40-250GeV_100k.root", TString output_prefix="new"){
        TString treename="fancy_tree";
        TString outname=output_prefix+"_"+inname;
        TFile* fin=new TFile(path + inname);
        TTree* tin=(TTree*)fin->Get(treename);
        TFile* fout=new TFile(outname,"recreate");
        TTree* tout=new TTree(treename,treename);
        // 2D hist in units of rad
        TH2D* sampling0=new TH2D("","",4,-0.1*2,0.1*2, 16,-0.025*8, 0.025*8);
        TH2D* sampling1=new TH2D("","",4,-0.098*2,0.098*2, 128,-0.003125*64, 0.003125*64);
        TH2D* sampling2=new TH2D("","",16,-0.0245*8, 0.0245*8, 16,-0.025*8, 0.025*8);
        TH2D* sampling3=new TH2D("","",16,-0.0245*8, 0.0245*8, 8,-0.05*4, 0.05*4);
        double cells[964];
      
         // Collect the Cell info for each layer
         std::vector<Cell> prelayer_cells;
         std::vector<Cell> firstlayer_cells;
         std::vector<Cell> secondlayer_cells;
         std::vector<Cell> thirdlayer_cells;                
 
        // The total energy and energy fraction 
        float total_e(0.), prelayer_e(0.), firstlayer_e(0.),secondlayer_e(0.),thirdlayer_e(0.);
        // Depth-weighted total energy, ld 
        float depth_weighted_total_e(0.),depth_weighted_total_e2(0.);
        float prelayer_x(0), prelayer_x2(0.), firstlayer_x(0.),firstlayer_x2(0.),secondlayer_x(0.),secondlayer_x2(0.), thirdlayer_x(0.),thirdlayer_x2(0.);
        float prelayer_y(0), prelayer_y2(0.), firstlayer_y(0.),firstlayer_y2(0.),secondlayer_y(0.),secondlayer_y2(0.), thirdlayer_y(0.),thirdlayer_y2(0.);
        float prelayer_cell(0), prelayer_cell2(0.), firstlayer_cell(0.),firstlayer_cell2(0.),secondlayer_cell(0.),secondlayer_cell2(0.), thirdlayer_cell(0.),thirdlayer_cell2(0.);
        float frac_pre(0.), frac_first(0.),frac_second(0.), frac_third(0.);
        float shower_depth(0.),shower_depth_width(0.);
        // This is similar to ATLAS Wphi2
        float pre_lateral_width_x(0.), first_lateral_width_x(0.), second_lateral_width_x(0.), third_lateral_width_x(0.);
        // This is similar to ATLAS Weta2
        float pre_lateral_width_y(0.), first_lateral_width_y(0.), second_lateral_width_y(0.), third_lateral_width_y(0.);
        float pre_lateral_width_cell(0.), first_lateral_width_cell(0.), second_lateral_width_cell(0.), third_lateral_width_cell(0.);

        float second_R_eta(0.), second_R_phi(0.), second_lateral_width_eta_weta2(0.), first_lateral_width_eta_w3(0.), first_lateral_width_eta_w20(0.), first_fraction_fside(0.), first_dEs(0.);
        float pre_Eratio(0.), first_Eratio(0.), second_Eratio(0.), third_Eratio(0.);
   
        // Deepest layer that recoerds non-zero energy 
        int d(0);

        tout->Branch("total_e",&total_e,"total_e/F");
        tout->Branch("prelayer_e",&prelayer_e,"prelayer_e/F");
        tout->Branch("firstlayer_e",&firstlayer_e,"firstlayer_e/F");
        tout->Branch("secondlayer_e",&secondlayer_e,"secondlayer_e/F");
        tout->Branch("thirdlayer_e",&thirdlayer_e,"thirdlayer_e/F");
        tout->Branch("depth_weighted_total_e",&depth_weighted_total_e,"depth_weighted_total_e/F");
        tout->Branch("depth_weighted_total_e2",&depth_weighted_total_e2,"depth_weighted_total_e2/F");
        tout->Branch("prelayer_x",&prelayer_x,"prelayer_x/F");
        tout->Branch("prelayer_x2",&prelayer_x2,"prelayer_x2/F");
        tout->Branch("firstlayer_x",&firstlayer_x,"firstlayer_x/F");
        tout->Branch("firstlayer_x2",&firstlayer_x2,"firstlayer_x2/F");
        tout->Branch("secondlayer_x",&secondlayer_x,"secondlayer_x/F");
        tout->Branch("secondlayer_x2",&secondlayer_x2,"secondlayer_x2/F");
        tout->Branch("thirdlayer_x",&thirdlayer_x,"thirdlayer_x/F");
        tout->Branch("thirdlayer_x2",&thirdlayer_x2,"thirdlayer_x2/F");
        tout->Branch("prelayer_y",&prelayer_y,"prelayer_y/F");
        tout->Branch("prelayer_y2",&prelayer_y2,"prelayer_y2/F");
        tout->Branch("firstlayer_y",&firstlayer_y,"firstlayer_y/F");
        tout->Branch("firstlayer_y2",&firstlayer_y2,"firstlayer_y2/F");
        tout->Branch("secondlayer_y",&secondlayer_y,"secondlayer_y/F");
        tout->Branch("secondlayer_y2",&secondlayer_y2,"secondlayer_y2/F");
        tout->Branch("thirdlayer_y",&thirdlayer_y,"thirdlayer_y/F");
        tout->Branch("thirdlayer_y2",&thirdlayer_y2,"thirdlayer_y2/F");
        tout->Branch("frac_pre",&frac_pre,"frac_pre/F");
        tout->Branch("frac_first",&frac_first,"frac_first/F");
        tout->Branch("frac_second",&frac_second,"frac_second/F");
        tout->Branch("frac_third",&frac_third,"frac_third/F");
        tout->Branch("depth_weighted_total_e",&depth_weighted_total_e,"depth_weighted_total_e/F");
        tout->Branch("shower_depth",&shower_depth,"shower_depth/F");
        tout->Branch("shower_depth_width",&shower_depth_width,"shower_depth_width/F");
        tout->Branch("pre_lateral_width_x",&pre_lateral_width_x,"pre_lateral_width_x/F");
        tout->Branch("first_lateral_width_x",&first_lateral_width_x,"first_lateral_width_x/F");
        tout->Branch("second_lateral_width_x",&second_lateral_width_x,"second_lateral_width_x/F");
        tout->Branch("third_lateral_width_x",&third_lateral_width_x,"third_lateral_width_x/F");
        tout->Branch("pre_lateral_width_y",&pre_lateral_width_y,"pre_lateral_width_y/F");
        tout->Branch("first_lateral_width_y",&first_lateral_width_y,"first_lateral_width_y/F");
        tout->Branch("second_lateral_width_y",&second_lateral_width_y,"second_lateral_width_y/F");
        tout->Branch("third_lateral_width_y",&third_lateral_width_y,"third_lateral_width_y/F");
        tout->Branch("pre_lateral_width_cell",&pre_lateral_width_cell,"pre_lateral_width_cell/F");
        tout->Branch("first_lateral_width_cell",&first_lateral_width_cell,"first_lateral_width_cell/F");
        tout->Branch("second_lateral_width_cell",&second_lateral_width_cell,"second_lateral_width_cell/F");
        tout->Branch("third_lateral_width_cell",&third_lateral_width_cell,"third_lateral_width_cell/F");
        
        tout->Branch("pre_Eratio",&pre_Eratio,"pre_Eratio/F");
        tout->Branch("first_Eratio",&first_Eratio,"first_Eratio/F");
        tout->Branch("second_Eratio",&second_Eratio,"second_Eratio/F");
        tout->Branch("third_Eratio",&third_Eratio,"third_Eratio/F");
        
        tout->Branch("second_R_eta",&second_R_eta,"second_R_eta/F");
        tout->Branch("second_R_phi",&second_R_phi,"second_R_phi/F");
        tout->Branch("second_lateral_width_eta_weta2",&second_lateral_width_eta_weta2,"second_lateral_width_eta_weta2/F");
      
        tout->Branch("first_lateral_width_eta_w3",&first_lateral_width_eta_w3, "first_lateral_width_eta_w3/F");
        tout->Branch("first_lateral_width_eta_w20",&first_lateral_width_eta_w20,"first_lateral_width_eta_w20/F");
        tout->Branch("first_fraction_fside",&first_fraction_fside,"first_fraction_fside/F");
        tout->Branch("first_dEs",&first_dEs,"first_dEs/F");
       
        tout->Branch("d",&d,"d/I");
        
        for(int icell=0;icell<964;++icell){
            TString name="cell_";
            name+=icell;
            tin->SetBranchAddress(name,&(cells[icell]));
        }
        // Loop over all the events
        for(int ievt=0;ievt<100;++ievt){
        //for(int ievt=0;ievt<tin->GetEntries();++ievt){
            prelayer_cells.clear();
            firstlayer_cells.clear();
            secondlayer_cells.clear();
            thirdlayer_cells.clear();
            total_e=0.;
            prelayer_e=0;
            firstlayer_e=0;
            secondlayer_e=0.;
            thirdlayer_e=0.;
            depth_weighted_total_e=0.;
            depth_weighted_total_e2=0.;
            
            prelayer_x=0.;
            prelayer_x2=0.;
            firstlayer_x=0.;
            firstlayer_x2=0.;
            secondlayer_x=0.;
            secondlayer_x2=0.;
            thirdlayer_x=0.;
            thirdlayer_x2=0.;
         
            prelayer_y=0.;
            prelayer_y2=0.;
            firstlayer_y=0.;
            firstlayer_y2=0.;
            secondlayer_y=0.;
            secondlayer_y2=0.;
            thirdlayer_y=0.;
            thirdlayer_y2=0.;
            
            prelayer_cell=0.;
            prelayer_cell2=0.;
            firstlayer_cell=0.;
            firstlayer_cell2=0.;
            secondlayer_cell=0.;
            secondlayer_cell2=0.;
            thirdlayer_cell=0.;
            thirdlayer_cell2=0.;
    
            d=0;
       
            tin->GetEntry(ievt);
            // Loop over all the cells
            for(int icell=0;icell<964;++icell){
                int zbin=get_z(icell);
                int ybin=get_y(icell,zbin);
                int xbin=get_x(icell,ybin,zbin);
                //std::cout<<"xbin =  " << xbin <<", ybin = "<< ybin <<", zbin = "<< zbin << std::endl;
 
                depth_weighted_total_e+=zbin * cells[icell];
                depth_weighted_total_e2+=zbin*zbin*cells[icell];
                double xvalue=0.;
                double yvalue=0.;
                total_e+=cells[icell];
                if(zbin==3) {
                    thirdlayer_e+=cells[icell];
                    sampling3->Fill(sampling3->GetXaxis()->GetBinCenter(xbin+1),sampling3->GetYaxis()->GetBinCenter(ybin+1),cells[icell]);
                    xvalue=sampling3->GetXaxis()->GetBinCenter(xbin+1);
                    yvalue=sampling3->GetYaxis()->GetBinCenter(ybin+1);
                    thirdlayer_x+= xvalue*cells[icell];
                    thirdlayer_x2+=xvalue*xvalue*cells[icell];
                    thirdlayer_y+=yvalue*cells[icell];
                    thirdlayer_y2+=yvalue*yvalue*cells[icell];
                    thirdlayer_cell+=icell*cells[icell];
                    thirdlayer_cell2+=icell*icell*cells[icell];
                    thirdlayer_cells.push_back(Cell(xvalue, yvalue, cells[icell], xbin, ybin, zbin));
                }
                if(zbin==2) {
                    secondlayer_e+=cells[icell];
                    sampling2->Fill(sampling2->GetXaxis()->GetBinCenter(xbin+1),sampling2->GetYaxis()->GetBinCenter(ybin+1),cells[icell]);
                    xvalue=sampling2->GetXaxis()->GetBinCenter(xbin+1);
                    yvalue=sampling2->GetYaxis()->GetBinCenter(ybin+1);
                    secondlayer_x+=xvalue*cells[icell];
                    secondlayer_x2+=xvalue*xvalue*cells[icell];
                    secondlayer_y+=yvalue*cells[icell];
                    secondlayer_y2+=yvalue*yvalue*cells[icell];
                    secondlayer_cell+=icell*cells[icell];
                    secondlayer_cell2+=icell*icell*cells[icell];
                    secondlayer_cells.push_back(Cell(xvalue, yvalue, cells[icell], xbin, ybin, zbin));
                }
                if(zbin==1) {
                    firstlayer_e+=cells[icell];
                    sampling1->Fill(sampling1->GetXaxis()->GetBinCenter(xbin+1),sampling1->GetYaxis()->GetBinCenter(ybin+1),cells[icell]);
                    xvalue=sampling1->GetXaxis()->GetBinCenter(xbin+1);
                    yvalue=sampling1->GetYaxis()->GetBinCenter(ybin+1);
                    firstlayer_x+=xvalue*cells[icell];
                    firstlayer_x2+=xvalue*xvalue*cells[icell];
                    firstlayer_y+=yvalue*cells[icell];
                    firstlayer_y2+=yvalue*yvalue*cells[icell];
                    firstlayer_cell+=icell*cells[icell];
                    firstlayer_cell2+=icell*icell*cells[icell];
                    firstlayer_cells.push_back(Cell(xvalue, yvalue, cells[icell], xbin, ybin, zbin));
                }
                if(zbin==0) {
                    prelayer_e+=cells[icell];
                    sampling0->Fill(sampling0->GetXaxis()->GetBinCenter(xbin+1),sampling0->GetYaxis()->GetBinCenter(ybin+1),cells[icell]);
                    xvalue=sampling0->GetXaxis()->GetBinCenter(xbin+1);
                    yvalue=sampling0->GetYaxis()->GetBinCenter(ybin+1);
                    prelayer_x+=xvalue*cells[icell];
                    prelayer_x2+=xvalue*xvalue*cells[icell];
                    prelayer_y+=yvalue*cells[icell];
                    prelayer_y2+=yvalue*yvalue*cells[icell];
                    prelayer_cell+=icell*cells[icell];
                    prelayer_cell2+=icell*icell*cells[icell];
                    prelayer_cells.push_back(Cell(xvalue, yvalue, cells[icell], xbin, ybin, zbin));
                }
 
                if(cells[icell]>0){
                  if(zbin>d){
                     d=zbin;
                  }
                }
            } // end of all cells
       
            // sort the cells according to energy on each layer
            std::sort(prelayer_cells.begin(), prelayer_cells.end(), [](const Cell& lhs, const Cell& rhs){return lhs.energy > rhs.energy;});
            std::sort(firstlayer_cells.begin(), firstlayer_cells.end(), [](const Cell& lhs, const Cell& rhs){return lhs.energy > rhs.energy;});
            std::sort(secondlayer_cells.begin(), secondlayer_cells.end(), [](const Cell& lhs, const Cell& rhs){return lhs.energy > rhs.energy;});
            std::sort(thirdlayer_cells.begin(), thirdlayer_cells.end(), [](const Cell& lhs, const Cell& rhs){return lhs.energy > rhs.energy;});
 
            frac_pre=prelayer_e/total_e;
            frac_first=firstlayer_e/total_e;
            frac_second=secondlayer_e/total_e;
            frac_third=thirdlayer_e/total_e;
         
            // Sd = ld/Etot
            shower_depth=depth_weighted_total_e/total_e;
            // Sigma_Sd
            shower_depth_width=pow(depth_weighted_total_e2/total_e-pow((depth_weighted_total_e/total_e),2),0.5);
      
            pre_lateral_width_x=pow(prelayer_x2/prelayer_e-pow((prelayer_x/prelayer_e),2),0.5);
            first_lateral_width_x=pow(firstlayer_x2/firstlayer_e-pow((firstlayer_x/firstlayer_e),2),0.5);
            second_lateral_width_x=pow(secondlayer_x2/secondlayer_e-pow((secondlayer_x/secondlayer_e),2),0.5);
            third_lateral_width_x=pow(thirdlayer_x2/thirdlayer_e-pow((thirdlayer_x/thirdlayer_e),2),0.5);
            pre_lateral_width_y=pow(prelayer_y2/prelayer_e-pow((prelayer_y/prelayer_e),2),0.5);
            first_lateral_width_y=pow(firstlayer_y2/firstlayer_e-pow((firstlayer_y/firstlayer_e),2),0.5);
            second_lateral_width_y=pow(secondlayer_y2/secondlayer_e-pow((secondlayer_y/secondlayer_e),2),0.5);
            third_lateral_width_y=pow(thirdlayer_y2/thirdlayer_e-pow((thirdlayer_y/thirdlayer_e),2),0.5);
 
            pre_lateral_width_cell=pow(prelayer_cell2/prelayer_e-pow((prelayer_cell/prelayer_e),2),0.5);
            first_lateral_width_cell=pow(firstlayer_cell2/firstlayer_e-pow((firstlayer_cell/firstlayer_e),2),0.5);
            second_lateral_width_cell=pow(secondlayer_cell2/secondlayer_e-pow((secondlayer_cell/secondlayer_e),2),0.5);
            third_lateral_width_cell=pow(thirdlayer_cell2/thirdlayer_e-pow((thirdlayer_cell/thirdlayer_e),2),0.5);
            
            pre_Eratio = getERatio(prelayer_cells); 
            first_Eratio = getERatio(firstlayer_cells); 
            second_Eratio = getERatio(secondlayer_cells); 
            third_Eratio = getERatio(thirdlayer_cells); 
            
            // Variables only for second layer
            second_R_eta = getRatio(secondlayer_cells, 3, 7, 7, 7);                         
            second_R_phi = getRatio(secondlayer_cells, 3, 3, 3, 7);                         
            second_lateral_width_eta_weta2 = getWidthAngle(secondlayer_cells, 3, 5);        
 
            // Variables only for first layer
            first_lateral_width_eta_w3 = getWidthIndex(firstlayer_cells, 3); 
            first_lateral_width_eta_w20 = getWidthIndex(firstlayer_cells, 20); 
            first_fraction_fside = getFractionSide(firstlayer_cells, 3, 7); 
            first_dEs = getDeltaE(firstlayer_cells); 
             
            tout->Fill();

        } // end of all events

        tout->Write();
        fout->Close();
        return 0;
}
