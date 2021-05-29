#ifndef PhotonJetPhysics_hh
#define PhotonJetPhysics_hh 1

#include "G4VPhysicsConstructor.hh"
#include "G4SystemOfUnits.hh"
#include "globals.hh"

#include "Newaxion1.hh"
#include "Newaxion2.hh"
#include "Newscalar1.hh"

class PhotonJetPhysics: public G4VPhysicsConstructor
{
	public:
		PhotonJetPhysics();
		virtual ~PhotonJetPhysics();

		virtual void ConstructParticle();

		virtual void ConstructProcess();

		void setDarkPhotonMass(G4double mass) { if (Newaxion1::axion1() == nullptr) fDarkPhotonMass = mass; }
		G4double getDarkPhotonMass() { return (Newaxion1::axion1() != nullptr ? Newaxion1::axion1()->GetPDGMass() : fDarkPhotonMass); }

		static constexpr G4double default_darkPhoton_mass = 1000.0 * MeV;

	private:
		G4double fDarkPhotonMass;
		G4VPhysicsConstructor*  fDecayPhysicsList;

};
#endif
