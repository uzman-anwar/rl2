#!/bin/bash

#python run_pruning.py -b 1. -tei LWOP -eei LWOP -ed cifar10 -t 100000 -tk 0.1 -piv 0. -ecs null -ns 128 -ee 128 #-pn SigmoidPolicy -us -ssf -1


# Run policy
#python run_pruning.py run_policy -l pruning/wandb/run-20210101_123934-1suvohr3 -e LWOP -en vgg11 -ed cifar10 -efi 25000 -efbs 60 -eo adam -elr 0.0003 -eutd -nr 1 -li 32768
#exit

for i in {1..1}
do
    python -m pruning.cpg -tei FP -eei FP -en vgg11 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 1. --group fp-vgg11-b-0.10-ac-0.95 -b 0.10 -ebs 20000 -eefi 25000 -ee 2000 -teaci 0.95 -teacg 0. -eeac 0.95
    python -m pruning.cpg -tei FP -eei FP -en vgg11 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 1. --group fp-vgg11-b-0.05-ac-0.95 -b 0.05 -ebs 20000 -eefi 25000 -ee 2000 -teaci 0.95 -teacg 0. -eeac 0.95
    python -m pruning.cpg -tei FP -eei FP -en vgg16 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 1. --group fp-vgg16-b-0.10-ac-0.95 -b 0.10 -ebs 20000 -eefi 35000 -ee 2000 -teaci 0.95 -teacg 0. -eeac 0.95
    python -m pruning.cpg -tei FP -eei FP -en vgg16 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 1. --group fp-vgg16-b-0.05-ac-0.95 -b 0.05 -ebs 20000 -eefi 35000 -ee 2000 -teaci 0.95 -teacg 0. -eeac 0.95
done
