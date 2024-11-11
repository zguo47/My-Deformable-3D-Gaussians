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
export CUDA_VISIBLE_DEVICES=0

names=(baseball fan jacks1 occlusion)

cd /fs/nexus-projects/video-depth-pose/videosfm/test/Deformable-3D-Gaussians 
for name in "${names[@]}"; do
  echo "Processing $name"
  
  # Run the Python script for the current directory
  python train.py -s /fs/nexus-projects/video-depth-pose/videosfm/datasets/quad/${name} -m output/${name}_w_depth --eval --iterations 20000 

done
