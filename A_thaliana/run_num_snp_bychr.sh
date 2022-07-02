#!/bin/sh


for chr in 1 2 3 4 5; do
for maf in 0.01 0.05 0.1; do
sbatch --cpus-per-task 6 --mem-per-cpu 16G --wrap "python /home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/num_snp_bygene_bychr.py --maf $maf --chr $chr"

done
done