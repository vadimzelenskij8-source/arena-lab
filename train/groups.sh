#!/bin/bash
# Extra trainer args per worker W. Usage: groups.sh W [smoke]
# Main loop (W 0-11): the network + search teaches the network (expert iteration), strong opponents, matchups.
# W 12-13: + the network's position value inside the search. W 14-15: fast search (2 candidates, shorter lookahead).
W=$1; X="--fork 0 --meta-share 0.4 --strong 1 --selfsearch 1"
[ -s champ.pt ] && X="$X --champ champ.pt"
[ -s api.json ] && X="$X --api-decks api.json"
[ -s mw.json ] && X="$X --matchups mw.json"
[ -s cw.json ] && X="$X --card-w cw.json"
G=main
if [ "$2" = smoke ]; then [ "$W" = 2 ] && G=value; [ "$W" = 3 ] && G=fast
else [ "$W" -ge 12 ] && G=value; [ "$W" -ge 14 ] && G=fast; fi
case $G in
  value) echo "$X --sw 0.15" ;;
  fast) echo "$X --top 2 --horizon 100" ;;
  *) echo "$X" ;;
esac
