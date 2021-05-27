# export GAN_FNAME="generated_file_test.root"
#./generation -m input/axion1_100GeV_20k.mac

#particles=("axion1" "gamma" "electron" "pi0")
#enes=("100GeV" "500GeV" "1TeV")

particles=("axion1")
enes=("1TeV")
nSamples=19
nEvents=`expr $nSamples + 1`
echo nEvents = ${nEvents}k

script_dir=/nfs/dust/atlas/user/xiaocong/photonJet/generation/script
output_dir=/nfs/dust/atlas/user/xiaocong/photonJet/generation/output/05-26-21-21
root_dir=${output_dir}/root
h5_dir=${output_dir}/h5

if [ ! -d ${script_dir} ];then
  echo ${script_dir} does not exist!
fi

if [ ! -d ${output_dir} ];then
  echo ${output_dir} does not exist!
fi

# check if all requested files exist
for ((m=0; m<${#particles[@]};++m)); do
    for ((n=0; n<${#enes[@]};++n)); do
        particle=${particles[m]}
        ene=${enes[n]}

        file=${root_dir}/${particle}_${ene}_${nEvents}k.root
        if [ ! -e $file ]; then
            echo ERROR: $file does not exist!
        fi

        #python ${script_dir}/convert.py --in-file ${file} --out-file ${h5_dir}/${particle}_${ene}_${nEvents}k.h5  --tree fancy_tree 

    done
done


