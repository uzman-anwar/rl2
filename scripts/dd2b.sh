#!/bin/bash

#taskset --cpu-list 6-10 python run_me.py icrl -tei DD2B-v0 -eei CDD2B-v0 -er 20 -ep icrl/expert_data/DD2B -ft 200000 -bi 10 -ec 0.09 -rg 0.92 -cg 0.92 -rgl 0.9 -cgl 0.9 -plr 0.1 -piv 1 -ni 50
#taskset --cpu-list 6-10 python run_me.py icrl -tei DD2B-v0 -eei CDD2B-v0 -er 20 -ep icrl/expert_data/DD2B -ft 2e5 -bi 10 -ec 0.09 -rg 0.92 -cg 0.92 -rgl 0.9 -cgl 0.9 -plr 0.1 -piv 1 -ni 50 -upid -ki 0.001 -kp 100 -dno

taskset --cpu-list 0-5 python run_me.py icrl -tei DD2B-v0 -eei CDD2B-v0 -er 20 -ep icrl/expert_data/DD2B -ft 2e5 -tk 0.1 -bi 20 -cl 30 30 -clr 0.001 -ec 0.1 -plr 1. -piv 1 -ni 50 -dno -dnr -dnc -b 0.01
taskset --cpu-list 5-9 python run_me.py icrl -tei DD2B-v0 -eei CDD2B-v0 -er 20 -ep icrl/expert_data/DD2B -ft 2e5 -tk 0.1 -bi 20 -cl 30 30 -clr 0.001 -ec 0.1 -plr 1. -piv 1 -ni 50 -dno -dnr -dnc -b 0.01


# Next
# 1. Old reward
# 2. Increase clr
