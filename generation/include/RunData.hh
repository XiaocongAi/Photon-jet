//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// $Id: RunData.hh 69223 2013-04-23 12:36:10Z gcosmo $
// 
/// \file RunData.hh
/// \brief Definition of the RunData class

#ifndef RunData_h
#define RunData_h 1

#include "G4Run.hh"
#include "globals.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

enum {
  kAbs = 0,
  kGap = 1,
  kDim = 2, 
  kNumCells = 960 + 4 // 4 overflow bins for the three calo layers
};  

///  Run data class
///
/// It defines data members to hold the energy deposit and track lengths
/// of charged particles in Absober and Gap layers.
/// 
/// In order to reduce the number of data members a 2-dimensions array 
/// is introduced for each quantity:
/// - fEdep[], fTrackLength[].
///
/// The data are collected step by step in SteppingAction, and
/// the accumulated values are filled in histograms and entuple
/// event by event in EventAction.

class RunData : public G4Run
{
public:
  RunData();
  virtual ~RunData();

  // void Add(G4int id, G4double de, G4double dl);
  void Add(G4int id, G4double de);
  void FillPerEvent();
  
  void Reset();

  // Get methods
  // G4String  GetVolumeName(G4int id) const;
  G4double  GetEdep(G4int id) const;
  G4double GetTotalEnergy(){return TotalEnergy;};
  G4double GetPx_Mom(){return px_mom;};
  G4double GetPy_Mom(){return py_mom;};
  G4double GetPz_Mom(){return pz_mom;};
  G4double GetPDG_Mom(){return pdg_mom;};
  G4double GetKe_Dau1(){return ke_dau1;};
  G4double GetPx_Dau1(){return px_dau1;};
  G4double GetPy_Dau1(){return py_dau1;};
  G4double GetPz_Dau1(){return pz_dau1;};
  G4double GetPDG_Dau1(){return pdg_dau1;};
  G4double GetKe_Dau2(){return ke_dau2;};
  G4double GetPx_Dau2(){return px_dau2;};
  G4double GetPy_Dau2(){return py_dau2;};
  G4double GetPz_Dau2(){return pz_dau2;};
  G4double GetPDG_Dau2(){return pdg_dau2;};
  void SetTotalEnergy(G4double e){TotalEnergy = e;};
  void SetMomentum(G4double px, G4double py, G4double pz, G4int pdg){
	  px_mom= px;
	  py_mom= py;
	  pz_mom= pz;
	  pdg_mom= pdg;
  };
  void SetDaughter1(G4double ke, G4double px, G4double py, G4double pz, G4int pdg){
	  ke_dau1= ke;
	  px_dau1= px;
	  py_dau1= py;
	  pz_dau1= pz;
	  pdg_dau1=pdg;
  };
  void SetDaughter2(G4double ke, G4double px, G4double py, G4double pz, G4int pdg){
	  ke_dau2= ke;
	  px_dau2= px;
	  py_dau2= py;
	  pz_dau2= pz;
	  pdg_dau2=pdg;
  };
  // G4double  GetTrackLength(G4int id) const; 

private:
  // G4String  fVolumeNames[kDim];
  G4double  fEdep[kNumCells];
  G4double TotalEnergy;
  G4double px_mom;
  G4double py_mom;
  G4double pz_mom;
  G4double pdg_mom;
  G4double ke_dau1;
  G4double px_dau1;
  G4double py_dau1;
  G4double pz_dau1;
  G4double pdg_dau1;
  G4double ke_dau2;
  G4double px_dau2;
  G4double py_dau2;
  G4double pz_dau2;
  G4double pdg_dau2;
  // G4double  fTrackLength[kDim];
};

// inline functions

// inline void RunData::Add(G4int id, G4double de, G4double dl) {
inline void RunData::Add(G4int id, G4double de) {
  fEdep[id] += de; 
  // fTrackLength[id] += dl;
}

// inline G4String  RunData::GetVolumeName(G4int id) const {
//   return fVolumeNames[id];
// }

inline G4double  RunData::GetEdep(G4int id) const {
  return fEdep[id];
}   

// inline G4double  RunData::GetTrackLength(G4int id) const {
//   return fTrackLength[id];
// }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

