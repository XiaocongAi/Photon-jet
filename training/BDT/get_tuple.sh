particles=("axion1" "axion2" "scalar1" "gamma" "electron" "pi0")
date=08-12-21-20

rootpath=/nfs/dust/atlas/user/xiaocong/photonJet/generation/output
filesuffix=40-250GeV_100k_mass0p5GeV.root

for ((m=0; m<${#particles[@]};++m)); do
   particle=${particles[m]};
   filepath=$rootpath/$date/root/ 
   filename=${particle}_${filesuffix}
  
   echo "processing file: $filepath/$filename"
   
   exe="root -lq 'newtuple.C(\"${filepath}\", \"${filename}\")'"
   echo $exe 
   eval $exe

done

