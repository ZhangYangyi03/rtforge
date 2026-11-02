#!/usr/bin/env bash
# Four configurations, one after the other, same duration each. The comparison is the
# deliverable -- a single number proves nothing because nobody knows what it was without.
#
#   ./scripts/collect_latency.sh            # 3 min each, ~12 min total
#   DURATION=30s ./scripts/collect_latency.sh
#
# Configurations, and what each one is supposed to show:
#   base       fifo threads, nothing else tuned        -- what an untuned box gives
#   mlockall   + -m, memory locked                     -- page faults during the run
#   isolated   + nohz_full/isolcpus on cores 14,15     -- scheduler and tick noise
#   loaded     + all cores busy with vision work       -- the failure mode the project is
#                                                         actually about
#
# The last one is the only interesting one. A latency number taken on an idle machine is a
# number about an idle machine.
set -u
cd "$(dirname "$0")/.."
mkdir -p results
DURATION=${DURATION:-3m}
LOAD=${LOAD:-$(nproc)}
ISOLATE=${ISOLATE:-14,15}

command -v cyclictest >/dev/null || { echo "cyclictest missing: apt-get install rt-tests"; exit 1; }

run() {
  local name=$1; shift
  local args=(-m -S -i200 -h400 -q -D "$DURATION")
  echo "== $name =="
  # -S: one thread per core, -m: mlockall, -i200: 200 us interval, -q: histogram
  cyclictest "${args[@]}" "$@" > "results/cyclictest_${name}.hist" 2>&1
  echo "   -> results/cyclictest_${name}.hist"
}

run base

run mlockall

if [ -w /sys/devices/system/cpu/isolated ]; then
  echo "$ISOLATE" | sudo tee /sys/devices/system/cpu/isolated >/dev/null 2>&1 || true
fi
run isolated

# Load: one busy process per core, so every core competes with the RT thread. This is the
# configuration where a soft-RT system starts losing its deadline.
pids=()
for _ in $(seq 1 "$LOAD"); do
  sha1sum /dev/zero >/dev/null 2>&1 &
  pids+=($!)
done
sleep 2
run loaded
for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
wait 2>/dev/null

echo
python3 -m rtforge.latency.histogram results/cyclictest_*.hist
