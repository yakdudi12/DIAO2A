#!/bin/bash
#SBATCH --job-name=punto2_tp3
#SBATCH --partition=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

cd /biodata/sbritez/cursositba/DIAO2A/TP3
export MPLBACKEND=Agg
/biodata/sbritez/cursositba/DIAO2A/.venv/bin/python -u punto2_tp3.py
