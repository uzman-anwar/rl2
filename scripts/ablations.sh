#!/bin/bash

# =====================================================================
# Expert Rollouts
# =====================================================================
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-ER1 -er 1 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -bi 10 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5 -psis -ctkno 2.5
done

for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-ER5 -er 2 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -bi 10 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5 -psis -ctkno 2.5
done

for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-ER10 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -bi 10 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5 -psis -ctkno 2.5
done

for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-ER20 -er 20 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -bi 10 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5 -psis -ctkno 2.5
done

# =====================================================================
# IS=No, Early Stopping=No
# =====================================================================
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-NoIS-NoES-BI1 -nis -ctkno 1e6 -ctkon 1e6 -bi 1 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-NoIS-NoES-BI5 -nis -ctkno 1e6 -ctkon 1e6 -bi 5 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-NoIS-NoES-BI10 -nis -ctkno 1e6 -ctkon 1e6 -bi 10 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-NoIS-NoES-BI20 -nis -ctkno 1e6 -ctkon 1e6 -bi 20 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done

# =====================================================================
# IS=No, Early Stopping=Yes
# =====================================================================
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-NoIS-ES-BI1 -nis -ctkno 2.5 -ctkon 10 -bi 1 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-NoIS-ES-BI5 -nis -ctkno 2.5 -ctkon 10 -bi 5 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-NoIS-ES-BI10 -nis -ctkno 2.5 -ctkon 10 -bi 10 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-NoIS-ES-BI20 -nis -ctkno 2.5 -ctkon 10 -bi 20 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done

# =====================================================================
# IS=Yes, Early Stopping=No
# =====================================================================
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-IS-NoES-BI1 -psis -ctkno 1e6 -ctkon 1e6 -bi 1 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-IS-NoES-BI5 -psis -ctkno 1e6 -ctkon 1e6 -bi 5 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-IS-NoES-BI10 -psis -ctkno 1e6 -ctkon 1e6 -bi 10 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-IS-NoES-BI20 -psis -ctkno 1e6 -ctkon 1e6 -bi 20 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done

# =====================================================================
# IS=Yes, Early Stopping=Yes
# =====================================================================
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-IS-ES-BI1 -psis -ctkno 2.5 -ctkon 10 -bi 1 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-IS-ES-BI5 -psis -ctkno 2.5 -ctkon 10 -bi 5 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-IS-ES-BI10 -psis -ctkno 2.5 -ctkon 10 -bi 10 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done
for i in {1..2}
do
    taskset --cpu-list 0-4 python run_me.py icrl -p Ablations --group A-IS-ES-BI20 -psis -ctkno 2.5 -ctkon 10 -bi 20 -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -ft 2e5 -ni 20 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5
done


