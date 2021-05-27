#!/bin/bash
if [[ $# -ne 1 ]]; then
    echo "USAGE: submit.sh <nSamples>"
    exit
fi

nSamples=$1

submitJob=true

#time_stamp=`date +"%m-%d-%y-%H"`
time_stamp=05-26-21-21
submit_dir="/nfs/dust/atlas/user/xiaocong/photonJet/generation/submit/${time_stamp}"
output_dir="/nfs/dust/atlas/user/xiaocong/photonJet/generation/output/${time_stamp}"

exe_dir="/nfs/dust/atlas/user/xiaocong/photonJet/Photon-jet/build/generation"
cfg_dir="/nfs/dust/atlas/user/xiaocong/photonJet/generation/config"
input_dir="/nfs/dust/atlas/user/xiaocong/photonJet/generation/input"

if [ ! -d "${exe_dir}" ]; then
    echo "The exe_dir does not exist!" 
fi

list=${input_dir}/input.txt
if [ ! -f ${list} ]; then
    echo "The ${list} does not exist!"
fi 


#mkdir -p ${submit_dir}/job
#mkdir -p ${submit_dir}/err
#mkdir -p ${submit_dir}/log
#mkdir -p ${submit_dir}/out
#mkdir -p ${submit_dir}/exe
#mkdir -p ${submit_dir}/input
#
#mkdir -p ${output_dir}/log
#mkdir -p ${output_dir}/err
#mkdir -p ${output_dir}/out
#mkdir -p ${output_dir}/root
#mkdir -p ${output_dir}/h5

work_dir=${submit_dir}/exe 

# one line is one config file
while read line
do
    echo read line $line
    jobname=`echo $line | sed "s/.mac//g"`
    echo jobname = $jobname

    for (( ijob=0; ijob<=${nSamples}; ijob++ ))
    do
	
	config=${cfg_dir}/${line}
	if [ ! -f "${config}" ]; then
	    echo ERROR: The config file ${config} does not exist!
	fi
	
	echo processing input file ${config} and produces output file ${GAN_FNAME} 
	run="${exe_dir}/generation -m ${config}"
	
	if ${submitJob}; then
            ################# script of qsub job  ##################
            jobfilename="${submit_dir}/job/${jobname}_"$ijob".job"
            rm -fr  ${jobfilename}
            echo "#!/bin/bash"   > ${jobfilename}
            echo "source /nfs/dust/atlas/user/xiaocong/run2Ana/wzy-fit/TRExFitter/setup.sh " >> ${jobfilename} 
            echo "export GAN_FNAME=${jobname}_${ijob}.root ">> ${jobfilename}
            echo "pushd ${work_dir}" >> ${jobfilename} 
            echo "${run} > ${submit_dir}/log/${jobname}_${ijob}.log 2>${submit_dir}/err/${jobname}_${ijob}.err " >> ${jobfilename}
            echo "mv ${work_dir}/${jobname}_${ijob}_t*.root ${output_dir}/out/" >> ${jobfilename}
            echo "popd" >> ${jobfilename}  
            chmod a+x ${jobfilename}
	    
            submitfilename="${submit_dir}/job/${jobname}_"$ijob".sub"
            rm -fr  ${submitfilename}
            echo "executable     = ${jobfilename}" > ${submitfilename}
            echo "should_transfer_files   = Yes"  >> ${submitfilename}
            echo "when_to_transfer_output = ON_EXIT"   >> ${submitfilename}
            #echo "input          = ${submit_dir}/input/${jobname}_"$ijob".txt"   >> ${submitfilename}
            echo "output         = ${submit_dir}/out/${jobname}_"$ijob".out2" >> ${submitfilename}
            echo "error          = ${submit_dir}/err/${jobname}_"$ijob".err2" >> ${submitfilename}
            echo "log            = ${submit_dir}/log/${jobname}_"$ijob".log2" >> ${submitfilename}
            echo "+RequestRuntime = 30000" >> ${submitfilename} 
            echo "universe       = vanilla"  >> ${submitfilename}
            echo "RequestMemory   = 2G" >> ${submitfilename}
            echo "queue" >> ${submitfilename}
	    
            #condor_submit ${submitfilename}
            ##########################################################
	fi
	
    done

done <${list}
