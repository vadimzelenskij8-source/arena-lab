#!/bin/bash
# Extra trainer args per worker W. Usage: groups.sh W [smoke]
# The network learns from the old teacher (it held ~51% there); the strength comes from the search, measured in separate jobs.
W=$1; X="--fork 0 --meta-share 0.4 --strong 1 --mb 32"
[ -s champ.pt ] && X="$X --champ champ.pt"
[ -s api.json ] && X="$X --api-decks api.json"
[ -s mw.json ] && X="$X --matchups mw.json"
[ -s cw.json ] && X="$X --card-w cw.json"
echo "$X"
