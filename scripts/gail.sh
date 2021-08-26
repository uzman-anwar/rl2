#!/bin/bash

#taskset --cpu-list 0-5 python run_me.py gail -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 0.5e6;

taskset --cpu-list 5-9 python run_me.py cpg -p ICRL-FE2 --group AntBroken-Expert -tk 0.01 -t 4e6 -tei AntWallBroken-v0 -eei AntWallBrokenTest-v0;
exit
for i in {1..5}
do
    taskset --cpu-list 5-9 python run_me.py icrl -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -p GAIL -tk 0.01 -cl 20 -clr 0.01 -ft 10000 --n_steps 2000 -ni 10 -bi 10 -dno -dnr -dnc --group LG-ICRL4;
done

for i in {1..5}
do
    taskset --cpu-list 5-9 python run_me.py icrl -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -p GAIL -tk 0.01 -cl 20 -clr 0.01 -ft 10000 --n_steps 2000 -ni 10 -bi 10 -dno --group LG-ICRL5;
done

for i in {1..5}
do
    taskset --cpu-list 5-9 python run_me.py icrl -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -p GAIL -tk 0.01 -cl 20 -clr 0.01 -ft 5e4 -ni 10 -bi 10 -plr 10.0 -dno -dnr -dnc --group LG-ICRL6;
done

for i in {1..5}
do
    taskset --cpu-list 5-9 python run_me.py icrl -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -p GAIL -tk 0.01 -cl 20 -clr 0.01 -ft 5e4 -ni 10 -bi 10 -dno --group LG-ICRL7;
done

taskset --cpu-list 5-9 python run_me.py cpg -p ICRL-FE2 --group AntWall-Nominal -tk 0.01 -t 4e6 -tei AntWall-v0 -eei AntWallTest-v0 -unc;
taskset --cpu-list 5-9 python run_me.py cpg -p ICRL-FE2 --group AntBroken-Expert -tk 0.01 -t 4e6 -tei AntWallBroken-v0 -eei AntWallBrokenTest-v0;
taskset --cpu-list 5-9 python run_me.py cpg -p ICRL-FE2 --group AntBroken-Nominal -tk 0.01 -t 4e6 -tei AntWallBroken-v0 -eei AntWallBrokenTest-v0 -unc;
exit


#    taskset --cpu-list 0-4 python run_me.py gail -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 5e5 -dno -dnr --group LG-GAIL2;
ID=$1
echo "SID: $ID"

if [[ $ID = 1 ]]
then
    for i in {1..3}
    do
        # DD2B
        taskset --cpu-list 0-4 python run_me.py gail -er 20 -ep icrl/expert_data/DD2B -tei  DD2B-v0 -eei CDD2B-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 5e6;
        taskset --cpu-list 0-4 python run_me.py gail -er 20 -ep icrl/expert_data/DD2B -tei  DD2B-v0 -eei CDD2B-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 5e6 -lc;
        taskset --cpu-list 0-4 python run_me.py gail -er 20 -ep icrl/expert_data/DD2B -tei CDD2B-v0 -eei CDD2B-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 5e6;

        # LGW
        taskset --cpu-list 0-4 python run_me.py gail -er 20 -ep icrl/expert_data/LGW -tei  LGW-v0 -eei CLGW-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 5e6;
        taskset --cpu-list 0-4 python run_me.py gail -er 20 -ep icrl/expert_data/LGW -tei  LGW-v0 -eei CLGW-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 5e6 -lc;
        taskset --cpu-list 0-4 python run_me.py gail -er 20 -ep icrl/expert_data/LGW -tei CLGW-v0 -eei CLGW-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 5e6;
    done

elif [[ $ID = 2 ]]
then
    for i in {1..3}
    do
        # DD3B
        taskset --cpu-list 5-9 python run_me.py gail -er 20 -ep icrl/expert_data/DD3B -tei  DD3B-v0 -eei CDD3B-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 5e6;
        taskset --cpu-list 5-9 python run_me.py gail -er 20 -ep icrl/expert_data/DD3B -tei  DD3B-v0 -eei CDD3B-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 5e6 -lc;
        taskset --cpu-list 5-9 python run_me.py gail -er 20 -ep icrl/expert_data/DD3B -tei CDD3B-v0 -eei CDD3B-v0 -p GAIL -tk 0.01 -dl 20 -dlr 0.003 -t 5e6;
    done
fi
