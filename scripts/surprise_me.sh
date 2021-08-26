#!/bin/bash


# ==================================================================
# Constraint Transfer
# ==================================================================
#taskset --cpu-list 0-4 python run_me.py cpg --cn_path ./icrl/expert_data/AntWallCT1/files/best_cn_model.pt -cosd 0 1 -casd -1 -tei PointCircle-v0 -eei PointCircleTest-v0 -p ICRL-3 --group PC-transfer -m set1 -tk 0.01 -t 3e6

# ==================================================================
# Gail
# ==================================================================
#python run_me.py gail -er 20 -ep icrl/expert_data/CDD2B-5M -tk 0.01 -ec 0.5 -dl 20 -t 2e6 -tei DD2B-v0 -eei CDD2B-v0 -p GAIL -dlr 0.003;

#taskset --cpu-list 0-4 python run_me.py gail -er 20 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -dl 20 -t 10e6 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -p GAIL -dlr 0.003 --group HC-1dim;

# ---- AntWall
#taskset --cpu-list 20-24 python run_me.py gail -p GAIL --group AntWall -ep icrl/expert_data/AntWall -er 45 -dl 40 40 -dlr 0.005 -t 10e6 -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4

# ---- GAIL Paper Parameters
# --gae_lambda 0.97 --gamma 0.995
# ==================================================================
# ICRL
# ==================================================================
#python run_me.py icrl -piv 1 -er 20 -ep icrl/expert_data/DD2B-5M -tk 0.01 -cl 64 64 -bi 1 -ft 5e5 -ni 100 -b 0.03 -tei DD2B-v0 -eei CDD2B-v0 -p ICRL-3 -clr 0.05 -rp;

# --- HCWithPos
#taskset --cpu-list 0-4 python run_me.py icrl -er 20 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 30 -bi 10 -ft 2e5 -ni 30 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -p ICRL-3 --group HCNewPSIS -clr 0.05 -aclr 0.9 -crc 0.5 -psis

# --- AntWall
#taskset --cpu-list 5-9 python run_me.py icrl -cosd 0 1 -casd -1 -p ICRL-3 --group AntWall-transfer-point -ep icrl/expert_data/AntWall -er 45 -cl 40 40 -clr 0.005 -aclr 0.9 -crc 0.6 -bi 5 -ft 3e5 -ni 20 -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -piv 0.1 -plr 0.05 -psis -tk 0.02 -ctkno 2.5


# ==================================================================
# CPG
# ==================================================================
# ---- AntBroken
#taskset --cpu-list 6-10 python run_me.py cpg -p ICRL-3 --group AntWallBroken-v0 -t 2e6 -tei AntWallBroken-v0 -eei AntWallBrokenTest-v0 --learning_rate 0.00025

# ---- HCWithPos
#python run_me.py cpg -tei HCWithPos-v0 -eei HCWithPosTest-v0 -t 5e6 -piv 3 -p PID --use_pid --seed 34;

# ---- AntCircle
#python run_me.py cpg -p PPO-Debug --timesteps 15e6 --group GoodAnt -tei AntCircle-v0 -eei AntCircle-v0 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -ucde

# ---- AntWall
#taskset --cpu-list 0-4 python run_me.py cpg -p PPO-Debug --timesteps 7e6 --group AntWall-PID-tk-ns -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -upid -tk 0.02 -ns 10000

#taskset --cpu-list 15-19 python run_me.py cpg -p PPO-Debug --timesteps 7e6 --group AntWall-PID-tk-ns -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -upid -tk 0.02 -ns 20000

#taskset --cpu-list 25-29 python run_me.py cpg -p PPO-Debug --timesteps 7e6 --group AntWall-Lagrang-tk -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -tk 0.025

# ---- PointNullReward
#python run_me.py cpg -tei PointNullReward-v0 -eei PointNullRewardTest-v0 -t 3e6 -piv 3 -p NullReward --group Point -ucde;

# ---- DD2B
taskset --cpu-list 5-9 python run_me.py icrl -tei DD2B-v0 -eei CDD2B-v0 -er 20 -ep icrl/expert_data/DD2B -wt 1e5 -ft 1e5 -bi 5 -ni 30 -cl 64 64 -crc 0.8 -clr 0.01 -aclr 0.95 -ec 0.3 -plr 10. -piv 1 -dno -dnr -dnc

#taskset --cpu-list 0-4 python run_me.py icrl -p ICRL-FE2 --group AntWall-GLag --train_gail_lambda -nis -ep icrl/expert_data/AntWall -er 45 -cl 40 40 -clr 0.005 -aclr 0.9 -crc 0.6 -bi 5 -ft 2e5 -ni 20 -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -piv 0.1 -plr 0.05 -psis -tk 0.02 -ctkno 2.5

#taskset --cpu-list 0-4 python run_me.py icrl -p ICRL-FE2 --group HC-Glag --train_gail_lambda -nis -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 30 -bi 10 -ft 2e5 -ni 30 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5 -psis -ctkno 2.5



# ==================================================================
# Torque Constraint
# ==================================================================
# ---- HalfCheetah
#taskset --cpu-list 21-25 python run_me.py cpg -tei HalfCheetah-v3 -eei HalfCheetahTest-v0 -t 10e6 -piv 3 -plr 1.0 --cost_gae_lambda 0.52 -tk 0.01 -p TorqueConstraint --group HalfCheetahMC --n_steps 10240 -lr 3e-5;
#taskset --cpu-list 26-30 python run_me.py cpg -tei HalfCheetah-v3 -eei HalfCheetahTest-v0 -t 10e6 -piv 3 -plr 1.0 -tk 0.01 -p TorqueConstraint --group HalfCheetahMC --n_steps 10240 -lr 3e-5;

# ---- Walker
#python run_me.py cpg -tei Walker2d-v2 -eei Walker2dTest-v0 -t 5e6 -piv 3 -tk 0.01 -p TorqueContraint --group Walker --budget 0.1;

# ---- Swimmer
#python run_me.py cpg -tei Swimmer-v3 -eei SwimmerTest-v0 -t 5e6 -piv 3 -tk 0.01 -p TorqueConstraint --group Swimmer --budget 0.1;

# ---- Ant
#python run_me.py cpg -tei Ant-v3 -eei AntTest-v0 -t 5e6 -piv 3 -tk 0.01 -p TorqueConstraint --group Ant1 --budget 0.1;
#python run_me.py cpg -tei Ant-v3 -eei AntTest-v0 --n_steps 20480 -t 15e6 -tk 0.01 -p TorqueContraint --group Ant --budget 0.1 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4;

# --- RCPO Paper Parameters
# policy_learning_rate 3e-4, critic_learning_rate 1.5e-4, penalty_initial_value 0 penalty_learning_rate 5e-7 penalty_decay_rate 0.99
# ==================================================================
# Safety gym
# ==================================================================
#taskset --cpu-list 31-35 python run_me.py cpg -tei Safexp-PointGoal2-v0 -eei Safexp-PointGoal2-v0 -t 10e6 -tk 0.01 -ec 0.1 -uls -p Safety_Gym;

# ==================================================================
# ICRL Torque Constraint
# ==================================================================
# ---- HalfCheetah
#python run_me.py icrl -ep icrl/expert_data/HCTorqueConstraint -er 20 -tk 0.01 -cosd -1 -piv 1 -cl 30 30 -clr 0.05 -aclr 0.9 -bi 10 -ft 2e5 -psis -ni 100 -tei HalfCheetah-v3 -eei HalfCheetahTest-v0 -p ICRL-3 --group HCTorqueConstraint;


# ==================================================================
# AIRL/New GAIL Code

# ==================================================================
# Torque Constraint
# ==================================================================
# ---- HalfCheetah
#taskset --cpu-list 21-25 python run_me.py cpg -tei HalfCheetah-v3 -eei HalfCheetahTest-v0 -t 10e6 -piv 3 -plr 1.0 --cost_gae_lambda 0.52 -tk 0.01 -p TorqueConstraint --group HalfCheetahMC --n_steps 10240 -lr 3e-5;
#taskset --cpu-list 26-30 python run_me.py cpg -tei HalfCheetah-v3 -eei HalfCheetahTest-v0 -t 10e6 -piv 3 -plr 1.0 -tk 0.01 -p TorqueConstraint --group HalfCheetahMC --n_steps 10240 -lr 3e-5;

# ---- Walker
#python run_me.py cpg -tei Walker2d-v2 -eei Walker2dTest-v0 -t 5e6 -piv 3 -tk 0.01 -p TorqueContraint --group Walker --budget 0.1;

# ---- Swimmer
#python run_me.py cpg -tei Swimmer-v3 -eei SwimmerTest-v0 -t 5e6 -piv 3 -tk 0.01 -p TorqueConstraint --group Swimmer --budget 0.1;

# ---- Ant
#python run_me.py cpg -tei Ant-v3 -eei AntTest-v0 -t 5e6 -piv 3 -tk 0.01 -p TorqueConstraint --group Ant1 --budget 0.1;
#python run_me.py cpg -tei Ant-v3 -eei AntTest-v0 --n_steps 20480 -t 15e6 -tk 0.01 -p TorqueContraint --group Ant --budget 0.1 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4;

# --- RCPO Paper Parameters
# policy_learning_rate 3e-4, critic_learning_rate 1.5e-4, penalty_initial_value 0 penalty_learning_rate 5e-7 penalty_decay_rate 0.99
# ==================================================================
# Safety gym
# ==================================================================
#taskset --cpu-list 31-35 python run_me.py cpg -tei Safexp-PointGoal2-v0 -eei Safexp-PointGoal2-v0 -t 10e6 -tk 0.01 -ec 0.1 -uls -p Safety_Gym;

# ==================================================================
# ICRL Torque Constraint
# ==================================================================
# ---- HalfCheetah
#python run_me.py icrl -ep icrl/expert_data/HCTorqueConstraint -er 20 -tk 0.01 -cosd -1 -piv 1 -cl 30 30 -clr 0.05 -aclr 0.9 -bi 10 -ft 2e5 -psis -ni 100 -tei HalfCheetah-v3 -eei HalfCheetahTest-v0 -p ICRL-3 --group HCTorqueConstraint;


# ==================================================================
# AIRL/New GAIL Code

# ==================================================================
# Torque Constraint
# ==================================================================
# ---- HalfCheetah
#taskset --cpu-list 21-25 python run_me.py cpg -tei HalfCheetah-v3 -eei HalfCheetahTest-v0 -t 10e6 -piv 3 -plr 1.0 --cost_gae_lambda 0.52 -tk 0.01 -p TorqueConstraint --group HalfCheetahMC --n_steps 10240 -lr 3e-5;
#taskset --cpu-list 26-30 python run_me.py cpg -tei HalfCheetah-v3 -eei HalfCheetahTest-v0 -t 10e6 -piv 3 -plr 1.0 -tk 0.01 -p TorqueConstraint --group HalfCheetahMC --n_steps 10240 -lr 3e-5;

# ---- Walker
#python run_me.py cpg -tei Walker2d-v2 -eei Walker2dTest-v0 -t 5e6 -piv 3 -tk 0.01 -p TorqueContraint --group Walker --budget 0.1;

# ---- Swimmer
#python run_me.py cpg -tei Swimmer-v3 -eei SwimmerTest-v0 -t 5e6 -piv 3 -tk 0.01 -p TorqueConstraint --group Swimmer --budget 0.1;

# ---- Ant
#python run_me.py cpg -tei Ant-v3 -eei AntTest-v0 -t 5e6 -piv 3 -tk 0.01 -p TorqueConstraint --group Ant1 --budget 0.1;
#python run_me.py cpg -tei Ant-v3 -eei AntTest-v0 --n_steps 20480 -t 15e6 -tk 0.01 -p TorqueContraint --group Ant --budget 0.1 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4;

# --- RCPO Paper Parameters
# policy_learning_rate 3e-4, critic_learning_rate 1.5e-4, penalty_initial_value 0 penalty_learning_rate 5e-7 penalty_decay_rate 0.99
# ==================================================================
# Safety gym
# ==================================================================
#taskset --cpu-list 31-35 python run_me.py cpg -tei Safexp-PointGoal2-v0 -eei Safexp-PointGoal2-v0 -t 10e6 -tk 0.01 -ec 0.1 -uls -p Safety_Gym;

# ==================================================================
# ICRL Torque Constraint
# ==================================================================
# ---- HalfCheetah
#python run_me.py icrl -ep icrl/expert_data/HCTorqueConstraint -er 20 -tk 0.01 -cosd -1 -piv 1 -cl 30 30 -clr 0.05 -aclr 0.9 -bi 10 -ft 2e5 -psis -ni 100 -tei HalfCheetah-v3 -eei HalfCheetahTest-v0 -p ICRL-3 --group HCTorqueConstraint;


# ==================================================================
# AIRL/New GAIL Code
# ==================================================================
#taskset --cpu-list 5-9 python run_me.py airl -er 40 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -t 1e6 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -p DeleteMe -dlr 0.003 --group AIRL-Testing2 -drl 20;
