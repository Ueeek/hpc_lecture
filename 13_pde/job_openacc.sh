#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=00:10:00

PGI_ACC_TIME=1 ./open_acc
