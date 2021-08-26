#!/bin/bash

# ==================================================================
# Constraint Transfer - ICRL
# ==================================================================

# Point on AntWall
for i in {1..6}
do
    taskset --cpu-list 0-4 python run_me.py cpg -p ICRL-FE2 --group Point-CT-ICRL --cn_path ./icrl/expert_data/ConstraintTransfer/ICRL/Point/files/best_cn_model.pt -cosd 0 1 -casd -1 -tei PointCircle-v0 -eei PointCircleTestBack-v0 -tk 0.01 -t 1.5e6 -plr 1.0
done

# AntBroken on AntWall
for i in {1..6}
do
    taskset --cpu-list 0-4 python run_me.py cpg -p ICRL-FE2 --group AntBroken-CT-ICRL --cn_path ./icrl/expert_data/ConstraintTransfer/ICRL/AntBroken/files/best_cn_model.pt -tei AntWallBroken-v0 -eei AntWallBrokenTest-v0 -tk 0.01 --batch_size 128 --reward_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -t 2e6 -plr 1.0
done

# ==================================================================
# Constraint Transfer - GAIL-Lambda
# ==================================================================

# Point On AntWall
for i in {1..6}
do
    taskset --cpu-list 0-4 python run_me.py cpg -p ICRL-FE2 --group Point-CT-Glag -cosd 0 1 -casd -1 --load_gail --cn_path ./icrl/expert_data/ConstraintTransfer/GAIL/Point/files/gail_discriminator.pt -tk 0.01 -t 1.5e6 -tei PointCircle-v0 -eei PointCircleTestBack-v0;
done

# AntBroken On AntWall
for i in {1..6}
do
    taskset --cpu-list 10,11 python run_me.py cpg -p ICRL-FE2 --group AntBroken-CT-GLag --load_gail --cn_path ./icrl/expert_data/ConstraintTransfer/GAIL/AntBroken/files/gail_discriminator.pt -tei AntWallBroken-v0 -eei AntWallBrokenTest-v0 -tk 0.01 --batch_size 128 --reward_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -t 3e6 -plr 1.0
done

# ==================================================================
# Constraint Transfer - GAIL-LogC
# ==================================================================

# Point On AntWall
for i in {1..6}
do
    taskset --cpu-list 0-4 python run_me.py gail -p ICRL-FE2 --group Point-CT-GLC -dosd 0 1 -dasd -1 --freeze_gail_weights --gail_path ./icrl/expert_data/ConstraintTransfer/GAIL/Point/files/gail_discriminator.pt -ep icrl/expert_data/AntWall -er 1 -tk 0.01 -t 1.5e5 -tei PointCircle-v0 -eei PointCircleTestBack-v0;
done

# AntBroken On AntWall
for i in {1..6}
do
    taskset --cpu-list 5-9 python run_me.py gail -p ICRL-FE2 --group AntBroken-CT-GLC --freeze_gail_weights --gail_path ./icrl/expert_data/ConstraintTransfer/GAIL/AntBroken/files/gail_discriminator.pt -ep icrl/expert_data/AntWall -er 2 -tei AntWallBroken-v0 -eei AntWallBrokenTest-v0 -tk 0.01 --batch_size 128 --reward_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -t 2e6
done

# ==================================================================
# ICRL
# ==================================================================

# DD2B?

# LapGridWorld
for i in {1..10}:
do
    taskset --cpu-list 5-9 python run_me.py icrl -p ICRL-FE2 --group LapGrid-ICRL -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -tk 0.01 -cl 20 -clr 0.003 -ft 0.5e5 -ni 10 -bi 20 -dno -dnr -dnc;
done

# Ant
for i in {1..11}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p ICRL-FE2 --group AntWall-ICRL -ep icrl/expert_data/AntWall -er 45 -cl 40 40 -clr 0.005 -aclr 0.9 -crc 0.6 -bi 5 -ft 2e5 -ni 20 -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -piv 0.1 -plr 0.05 -psis -tk 0.02 -ctkno 2.5
done

# HC
for i in {1..11}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p ICRL-FE2 --group HC-ICRL -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -bi 10 -ft 2e5 -ni 30 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5 -psis -ctkno 2.5
done

# ==================================================================
# GAIL-logC
# ==================================================================

# Ant
for i in {1...6}
do
    taskset --cpu-list 30-34 python run_me.py gail -p ICRL-FE2 --group AntWall-GLC -ep icrl/expert_data/AntWall -er 45 -dl 40 40 -dlr 0.005 -t 4e6 -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -lc
done

# HC
for i in {1..6}
do
    taskset --cpu-list 5-9 python run_me.py gail -p ICRL-FE2 --group HC-GLC -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -t 4e6 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -dl 30 -dlr 0.003 -lc
done

# ==================================================================
# GAIL-Lagrangian
# ==================================================================

# LapGridWorld
for i in {1..10}:
do
    taskset --cpu-list 5-9 python run_me.py icrl -p ICRL-FE2 --group LapGrid-Glag --train_gail_lambda -nis -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -tk 0.01 -cl 20 -clr 0.003 -ft 0.5e5 -ni 10 -bi 20 -dno -dnr -dnc;
done

# Ant
for i in {1..6}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p ICRL-FE2 --group AntWall-GLag --train_gail_lambda -nis -ep icrl/expert_data/AntWall -er 45 -cl 40 40 -clr 0.005 -aclr 0.9 -crc 0.6 -bi 5 -ft 2e5 -ni 20 -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -piv 0.1 -plr 0.05 -psis -tk 0.02 -ctkno 2.5
done

# HC
for i in {1..6}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p ICRL-FE2 --group HC-Glag --train_gail_lambda -nis -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 30 -bi 10 -ft 2e5 -ni 30 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5 -psis -ctkno 2.5
done
