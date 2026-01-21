#!/bin/bash
#
# Master script to submit all MachSMT cross-validation SLURM jobs
#
# Usage: ./submit_all.sh
#
# This script should be run from the z3alpha repository root directory

SCRIPT_DIR="JHC/AlgoSelect/DescriptionAgent/scripts/foiks26_cv"

echo "================================================================================"
echo "Submitting MachSMT Cross-Validation Jobs for all logics"
echo "================================================================================"
echo "Date: $(date)"
echo ""

LOGICS="ABV ALIA BV QF_IDL QF_LIA QF_NRA QF_SLIA UFLIA UFNIA"

for logic in $LOGICS; do
    script="${SCRIPT_DIR}/run_machsmt_cv_${logic}.sh"
    if [ -f "$script" ]; then
        echo "Submitting job for $logic..."
        sbatch "$script"
    else
        echo "ERROR: Script not found: $script"
    fi
done

echo ""
echo "================================================================================"
echo "All jobs submitted. Use 'squeue -u $USER' to monitor job status."
echo "================================================================================"
