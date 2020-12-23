#!/bin/bash
for CONFIG in configs/Vent/dummy/*.json; do
  echo "Processing $CONFIG"
  for COUNT in 0 1 2 3 4; do
    CONFIG_TMP="$CONFIG.SEED$COUNT"
    cat $CONFIG | sed "s/\"seed\": 0/\"seed\": $COUNT/g" > $CONFIG_TMP
    python3 src/run.py -c $CONFIG_TMP
    rm $CONFIG_TMP
    echo "Processed seed: $COUNT"
  done
  echo "Done with $CONFIG"
done
