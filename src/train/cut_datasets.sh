#!/bin/bash

#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --account=def-beltrame

module load opencv/4.10.0
module load python/3.10.13
python -m venv $SLURM_TMPDIR/.venv
source $SLURM_TMPDIR/.venv/bin/activate
python -m pip install -r requirements.txt
cd ../../../third_party/diffusion_policy/
python -m pip install .
cd -
python -m pip install hf_xet addict triton
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3
python -m pip install .
cd ../

mkdir $SLURM_TMPDIR/data
INPUT_DIR="/home/koki/projects/def-beltrame/vnm_datasets/processed_datasets"
#cp $INPUT_DIR/go_stanford.tar.gz $SLURM_TMPDIR/data/
#cp $INPUT_DIR/huron.tar.gz $SLURM_TMPDIR/data/
cp $INPUT_DIR/recon.tar.gz $SLURM_TMPDIR/data/
#cp $INPUT_DIR/scand.tar.gz $SLURM_TMPDIR/data/

#tar -xf $SLURM_TMPDIR/data/go_stanford.tar.gz -C $SLURM_TMPDIR/data/
#rm $SLURM_TMPDIR/data/go_stanford.tar.gz
#tar -xf $SLURM_TMPDIR/data/huron.tar.gz -C $SLURM_TMPDIR/data/
#rm $SLURM_TMPDIR/data/huron.tar.gz
tar -xf $SLURM_TMPDIR/data/recon.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/recon.tar.gz
#tar -xf $SLURM_TMPDIR/data/scand.tar.gz -C $SLURM_TMPDIR/data/
#rm $SLURM_TMPDIR/data/scand.tar.gz

#python cut_backwards.py -i $SLURM_TMPDIR/data/scand -o $SLURM_TMPDIR/data/scand_cut
#python cut_backwards.py -i $SLURM_TMPDIR/data/huron -o $SLURM_TMPDIR/data/sacson_cut
python cut_backwards.py -i $SLURM_TMPDIR/data/recon -o $SLURM_TMPDIR/data/recon_cut
#python cut_backwards.py -i $SLURM_TMPDIR/data/go_stanford -o $SLURM_TMPDIR/data/go_stanford_cut

#tar -czf $SLURM_TMPDIR/data/scand_cut.tar.gz -C $SLURM_TMPDIR/data/ scand_cut
#tar -czf $SLURM_TMPDIR/data/sacson_cut.tar.gz -C $SLURM_TMPDIR/data/ sacson_cut
tar -czf $SLURM_TMPDIR/data/recon_cut.tar.gz -C $SLURM_TMPDIR/data/ recon_cut
#tar -czf $SLURM_TMPDIR/data/go_stanford_cut.tar.gz -C $SLURM_TMPDIR/data/ go_stanford_cut

#cp $SLURM_TMPDIR/data/scand_cut.tar.gz $INPUT_DIR/
#cp $SLURM_TMPDIR/data/sacson_cut.tar.gz $INPUT_DIR/ 
cp $SLURM_TMPDIR/data/recon_cut.tar.gz $INPUT_DIR/ 
#cp $SLURM_TMPDIR/data/go_stanford_cut.tar.gz $INPUT_DIR/ 
