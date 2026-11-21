#!/bin/bash
# Extra trainer args per worker W (groups for comparison). Usage: groups.sh W [smoke]
# 0: base = A (matchups) + E (strong opponents); 1: + D (rewards); 2: + better imitation; 3: + D + imitation
# W 14-15 (W 3 in the smoke test): exploiters, trained only against the frozen champion.
W=$1; X="--fork 0 --meta-share 0.4"
[ -s champ.pt ] && X="$X --champ champ.pt"
[ -s api.json ] && X="$X --api-decks api.json"
[ -s mw.json ] && X="$X --matchups mw.json"
if [ "$W" -ge 14 ] || { [ "$2" = smoke ] && [ "$W" = 3 ]; }; then
  echo "$X --exploit 1 --pg 0.5 --beta1 0.2"; exit 0
fi
case $((W % 4)) in
  0) echo "$X --strong 1" ;;
  1) echo "$X --strong 1 --pg 0.5 --beta1 0.2" ;;
  2) echo "$X --strong 1 --epochs 3 --wait-w 0.08" ;;
  3) echo "$X --strong 1 --pg 0.5 --beta1 0.2 --epochs 3 --wait-w 0.08" ;;
esac
