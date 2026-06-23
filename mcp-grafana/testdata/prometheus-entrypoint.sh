#!/bin/sh
# Prometheus entrypoint script to backfill recording rules and start Prometheus

set -e

echo "Starting Prometheus entrypoint script..."


backfill() {
    # Calculate time range for backfilling (5 hours ago from now)
    # Get current time in seconds since epoch
    CURRENT_TIME=$(date -u +%s)
    # Subtract 5 hours (5 * 60 * 60 = 18000 seconds)
    START_TIME=$((CURRENT_TIME - 18000))

    # wait until Prometheus is up and running
    until wget http://localhost:9090/-/healthy -q -O /dev/null; do
        sleep 1
    done

    promtool tsdb create-blocks-from \
        rules \
        --url=http://localhost:9090 \
        --start="${START_TIME}" \
        --end="${CURRENT_TIME}" \
        --eval-interval=30s \
        /etc/prometheus/prometheus-seed.yml
}

# Start Prometheus with the regular configuration, this is needed for backfilling
/bin/prometheus \
    --config.file=/etc/prometheus/prometheus.yml &

backfill

# Restarting Prometheus after backfilling will allow to load the new blocks directly
# without having to wait for the next compaction cycle
kill %1
echo "Starting Prometheus server..."
# Start Prometheus with the regular configuration
/bin/prometheus \
    --config.file=/etc/prometheus/prometheus.yml

