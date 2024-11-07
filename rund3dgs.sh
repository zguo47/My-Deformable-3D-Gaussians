#!/bin/bash
#SBATCH --partition=vulcan-dpart
#SBATCH --time=12:00:00
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-default
#SBATCH --gres=gpu:p6000:1

set -x

module unload cuda/10.2.89
module add cuda/11.7.0
export WORK_DIR="/fs/nexus-projects/video-depth-pose/videosfm/test/Deformable-3D-Gaussians/slurm_${SLURM_JOBID}"
source /vulcanscratch/zguo47/miniconda3/bin/activate
conda activate deformable_gaussian_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vulcanscratch/zguo47/miniconda3/envs/deformable_gaussian_env/lib

lambdas=(0.05 0.2 0.5)
names=(copier deskbox cupboard phonebooth)

cd /fs/nexus-projects/video-depth-pose/videosfm/test/Deformable-3D-Gaussians 
for lambda in "${lambdas[@]}"; do
  for name in "${names[@]}"; do
    echo "Processing $name with lambda $lambda"
    
    # Run the Python script for the current directory
    python train.py -s /fs/nexus-projects/video-depth-pose/videosfm/datasets/torf_data/${name}/${name} -m output/${name}_w_depth_${lambda} --eval --iterations 20000 --lambda_depth ${lambda}

  done
done
