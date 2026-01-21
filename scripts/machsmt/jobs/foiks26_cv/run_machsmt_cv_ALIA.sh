#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --account=def-vganesh
#SBATCH --qos=normal
#SBATCH --constraint=rome
#SBATCH --job-name=machsmt_cv_ALIA
#SBATCH --output=JHC/AlgoSelect/DescriptionAgent/results/foiks26_cv/ALIA-slurm-%j.out
#SBATCH --error=JHC/AlgoSelect/DescriptionAgent/results/foiks26_cv/ALIA-slurm-%j.err

echo "================================================================================"
echo "SLURM JOB INFORMATION"
echo "================================================================================"
echo "Job ID:              $SLURM_JOB_ID"
echo "Job Name:            $SLURM_JOB_NAME"
echo "Node(s):             $SLURM_JOB_NODELIST"
echo "CPUs per task:       $SLURM_CPUS_PER_TASK"
echo "Start time:          $(date)"
echo "Working directory:   $SLURM_SUBMIT_DIR"
echo ""

echo "================================================================================"
echo "STARTING MachSMT Cross-Validation for ALIA"
echo "================================================================================"

python JHC/AlgoSelect/DescriptionAgent/scripts/run_MachSMT_cv.py --config /home/jchen688/projects/def-vganesh/jchen688/github/z3alpha/JHC/AlgoSelect/DescriptionAgent/config/foiks26_cv/ALIA_cv_config.json

echo ""
echo "================================================================================"
echo "JOB COMPLETION"
echo "================================================================================"
echo "End time:            $(date)"
echo "================================================================================"
