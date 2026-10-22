#!/usr/bin/env bash
# Generate traffic so Prometheus/Grafana have data to show.
URL=${1:-http://localhost:5000/predict}
for i in $(seq 1 200); do
  curl -s -X POST "$URL" -H 'Content-Type: application/json' -d '{"features":[5.1,3.5,1.4,0.2]}' >/dev/null
  curl -s -X POST "$URL" -H 'Content-Type: application/json' -d '{"features":[6.3,3.3,6.0,2.5]}' >/dev/null
  curl -s -X POST "$URL" -H 'Content-Type: application/json' -d '{"features":[5.9,3.0,4.2,1.5]}' >/dev/null
  [ $((i % 20)) -eq 0 ] && curl -s -X POST "$URL" -H 'Content-Type: application/json' -d '{"features":[1]}' >/dev/null
  sleep 0.1
done
echo "done"
