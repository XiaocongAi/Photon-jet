#!/bin/bash


#particles=("axion1" "gamma" "electron" "pi0")
#enes=("100GeV" "500GeV" "1TeV")

particles=("electron")
enes=("100GeV" "500GeV" "1TeV")
nSamples=19


time_stamp=05-26-21-21
submit_dir="/nfs/dust/atlas/user/xiaocong/photonJet/generation/submit/${time_stamp}"
output_dir="/nfs/dust/atlas/user/xiaocong/photonJet/generation/output/${time_stamp}"

exe_dir="/nfs/dust/atlas/user/xiaocong/photonJet/Photon-jet/build/generation"
cfg_dir="/nfs/dust/atlas/user/xiaocong/photonJet/generation/config"
input_dir="/nfs/dust/atlas/user/xiaocong/photonJet/generation/input"

if [ ! -d "${submit_dir}" ]; then
    echo "The submit_dir does not exist!" 
fi

if [ ! -d "${exe_dir}" ]; then
    echo "The exe_dir does not exist!" 
fi

list=${input_dir}/input.txt
if [ ! -f ${list} ]; then
    echo "The ${list} does not exist!"
fi 


for ((m=0; m<${#particles[@]};++m)); do
    for ((n=0; n<${#enes[@]};++n)); do
        for ((l=0; l<${nSamples};++l)); do
            particle=${particles[m]}
            ene=${enes[n]}
            jobname=${particle}_${ene}_1k_$l
            
            # check the submit file exists or not 
            submitfilename="${submit_dir}/job/${jobname}.sub"
            if [ ! -f ${submitfilename} ]; then
                echo ${submitfilename} does not exist!
                exit 0
            fi
            echo Resubmitting job $submitfilename

            condor_submit ${submitfilename}

        done
    done
done

