#!/bin/bash
eval $(/usr/bin/ssh-agent -s) >/dev/null
export SSH_ASKPASS=/tmp/ssh_pass.sh
export DISPLAY=1
setsid ssh-add /home/sergkurchev/.ssh/id_ed25519 < /dev/null
rsync -av --progress --ignore-times -e "ssh -p 2221 -o StrictHostKeyChecking=no" odin_weights/model_best.pth root@176.109.83.84:/root/skurchev/workspace/nbv_with_obstacles/weights/model_best.pth
