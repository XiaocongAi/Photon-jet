#include "PhotonJetPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4DecayPhysics.hh"

PhotonJetPhysics::PhotonJetPhysics() : G4VPhysicsConstructor(), fDarkPhotonMass(default_darkPhoton_mass)
{ 
	G4cout << "Defining femtoCoulomb" << G4endl;                                                                                  
	new G4UnitDefinition("femtoCoulomb", "fC", "Electric charge", coulomb/1e15);                                                 

        fDecayPhysicsList = new G4DecayPhysics();
}

PhotonJetPhysics::~PhotonJetPhysics()
{
	delete fDecayPhysicsList;
}

void PhotonJetPhysics::ConstructParticle()
{
	G4cout << "PhotonJetPhysics::ConstructParticle() called" << G4endl;
	fDecayPhysicsList->ConstructParticle();
	Newaxion1::axion1Definition(fDarkPhotonMass);
	Newaxion2::axion2Definition(fDarkPhotonMass);
	Newscalar1::scalar1Definition(fDarkPhotonMass);
}

void PhotonJetPhysics::ConstructProcess()
{
	fDecayPhysicsList->ConstructProcess();
}
