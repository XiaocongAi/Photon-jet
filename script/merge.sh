# export GAN_FNAME="generated_file_test.root"
#./generation -m input/axion1_100GeV_20k.mac

#particles=("axion1" "gamma" "electron" "pi0")
#enes=("100GeV" "500GeV" "1TeV")

particles=("gamma")
enes=("1TeV")
nSamples=19
nEvents=`expr $nSamples + 1`
echo nEvents = ${nEvents}k

out_dir=/nfs/dust/atlas/user/xiaocong/photonJet/generation/output/05-26-21-21/out
root_dir=/nfs/dust/atlas/user/xiaocong/photonJet/generation/output/05-26-21-21/root


# check if all requested files exist
for ((m=0; m<${#particles[@]};++m)); do
    for ((n=0; n<${#enes[@]};++n)); do
	for ((l=0; l<${nSamples};++l)); do
	    particle=${particles[m]}
	    ene=${enes[n]}

	    file0=${out_dir}/${particle}_${ene}_1k_${l}_t0.root
	    file1=${out_dir}/${particle}_${ene}_1k_${l}_t1.root
	    if [ ! -e $file0 ]; then
		echo ERROR: $file0 does not exist!
	    fi
	    if [ ! -e $file1 ]; then
		echo ERROR: $file1 does not exist!
	    fi
	done

	hadd ${root_dir}/${particle}_${ene}_${nEvents}k.root ${out_dir}/${particle}_${ene}_1k_*_t0.root ${out_dir}/${particle}_${ene}_1k_*_t1.root 

    done
done


