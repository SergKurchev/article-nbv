#!/bin/bash
eval $(ssh-agent -s)
DISPLAY=1 SSH_ASKPASS_REQUIRE=force SSH_ASKPASS=/tmp/ssh_pass.sh ssh-add ~/.ssh/id_ed25519
ssh -p 2221 -o StrictHostKeyChecking=no root@176.109.83.84 "$@"
