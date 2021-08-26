#!/bin/bash

# Install python, pip.
apt-get install python3-venv
apt install python3-pip

# Setup virtual environment
pip3 install --upgrade pip
mkdir ~/Desktop/ICRL
cd ~/Desktop/ICRL
python3 -m venv .env

# Copy code
apt-get install sshpass
sshpass -p Bayesian scp linuxubuntu@172.16.20.100:/home/linuxubuntu/Desktop/shehryar-usman/InverseCMDP/continuous/* ./
sshpass -p Bayesian scp -r linuxubuntu@172.16.20.100:/home/linuxubuntu/Desktop/shehryar-usman/InverseCMDP/continuous/expert ./

# Install ffmpeg.
apt-get install ffmpeg

# For displaying CPU stats.
apt-get install gnome-shell-extensions
apt install gir1.2-gtop-2.0 gir1.2-nm-1.0 gir1.2-clutter-1.0

# Wandb login.
echo e510cbd509e5745726a0b7e3ee94a39670ada954
wandb login

# Installing Mujoco
# Install binaries from https://www.roboti.us/, rename to mujoco200 and place in ~/home/.mujoco
# place license file in ~/home/.mujoco
sudo add-apt-repository ppa:jamesh/snap-support
sudo apt-get update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/usman/.mujoco/mujoco200/bin     # add to .bashrc
source ~/.bashrc
pip3 install -U 'mujoco-py<2.1,>=2.0'

# Aliases
alias pull_sb="cd ~/rl_codebase/; rm -rf stable_baselines3;scp -r linuxubuntu@172.16.20.100:/home/linuxubuntu/Desktop/shehryar-usman/rl_codebase/stable_baselines3/ ./"
alias pull_envs="cd ~/rl_codebase/; rm -rf custom_envs;scp -r linuxubuntu@172.16.20.100:/home/linuxubuntu/Desktop/shehryar-usman/rl_codebase/custom_envs/ ./"
alias pull_icrl="cd ~/rl_codebase/icrl/; rm *;scp linuxubuntu@172.16.20.100:/home/linuxubuntu/Desktop/shehryar-usman/rl_codebase/icrl/* .;cd ../"
alias pull_expert="cd ~/rl_codebase/icrl/; rm -rf expert_data;scp -r linuxubuntu@172.16.20.100:/home/linuxubuntu/Desktop/shehryar-usman/rl_codebase/icrl/expert_data/ ./;cd ../"
alias pull_configs="cd ~/rl_codebase/; rm -rf configs;scp -r linuxubuntu@172.16.20.100:/home/linuxubuntu/Desktop/shehryar-usman/rl_codebase/configs/ ./"
alias pull_all="cd ~/rl_codebase; pull_sb; pull_envs; pull_icrl; pull_expert; pull_configs; rm *; scp linuxubuntu@172.16.20.100:/home/linuxubuntu/Desktop/shehryar-usman/rl_codebase/* ."
alias RL="cd ~/rl_codebase; source .env/bin/activate"

# Video playing plugins
sudo apt-get install ffmpeg
sudo apt install libdvdnav4 libdvd-pkg gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly libdvd-pkg -y
sudo apt install ubuntu-restricted-extras vlc -y

# Python packages
pip install wandb==0.10.12 torch==1.5.0 gym==0.15.7 matplotlib==3.3.2 numpy==1.17.5 cloudpickle==1.2.2 tqdm pandas
