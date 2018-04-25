#!/bin/bash
n_machines=500
start_id=1
n_totl_images=50000
script_name='snakes_wrapper.py'
username='jk9'

# submit job
PARTITION='batch' # batch # bibs-smp # bibs-gpu # gpu # small-batch
QOS='bibs-tserre-condo' # pri-jk9

for i_machine in $(seq 1 $n_machines); do
sbatch -J "$script_name[$i_machine]" <<EOF
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH --mem=8G
#SBATCH --begin=now
#SBATCH --qos=$QOS
#SBATCH --output=/gpfs/scratch/$username/slurm/slurm-%j.out
#SBATCH --error=/gpfs/scratch/$username/slurm/slurm-%j.out

echo "Starting job $i_machine on $HOSTNAME"
LC_ALL=en_US.utf8 \
module load boost hdf5 ffmpeg/1.2 cuda/7.5.18

python $script_name $n_machines $i_machine $n_totl_images
EOF
done
