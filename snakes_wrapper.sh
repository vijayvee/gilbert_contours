#!/bin/bash
n_machines=20
start_id=1
n_totl_images=500000
script_name='snakes2_wrapper.py'
username='vveeraba'
data_root='/home/vveeraba/pathfinder_21/gilbert_contours/imgs'
# submit job
PARTITION='general_short' # batch # bibs-smp # bibs-gpu # gpu # small-batch

for i_machine in $(seq 1 $n_machines); do
sbatch -J "$script_name[$i_machine]" <<EOF
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH -n 1
#SBATCH -t 8:00:00
#SBATCH --mem=8G
#SBATCH --begin=now
#SBATCH --output=/home/vveeraba/slurm_output/pathfinder_gen/slurm-%j.out
#SBATCH --error=/home/vveeraba/slurm_output/pathfinder_gen/slurm-%j-err.out

echo "Starting job $i_machine on $HOSTNAME"

python $script_name $n_machines $i_machine $n_totl_images $data_root
EOF
done
