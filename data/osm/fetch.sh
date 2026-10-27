#!/bin/bash
# Percobaan bertahap dengan jeda eksponensial: 3 putaran, 2 server bergantian
servers=("https://overpass.kumi.systems/api/interpreter" "https://overpass-api.de/api/interpreter")
wait=60
for round in 1 2 3; do
  for s in "${servers[@]}"; do
    code=$(curl -s -S --max-time 200 -A "sambas-research/1.0 (academic)" -w "%{http_code}" --data-urlencode "data@q.txt" -o jalan_tempat.json "$s")
    echo "putaran $round $s -> HTTP $code size $(stat -c %s jalan_tempat.json)"
    if [ "$code" = "200" ] && head -c 1 jalan_tempat.json | grep -q '{'; then echo "SELESAI"; exit 0; fi
  done
  echo "tunggu ${wait}s"; sleep $wait; wait=$((wait*2))
done
echo "GAGAL"
