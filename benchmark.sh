#!/bin/bash
# Benchmark sweep for BitWeaving-S: runs ./bitweaving over every combination of
# rows x bits x threads and collects the average scan time into a CSV.
#
# Usage:
#   ./benchmark.sh
#
# The sweep can be customized with environment variables, e.g.:
#   ROWS_LIST="1000000 8000000" BITS_LIST="8 32" THREADS_LIST="1 2 4" ./benchmark.sh
#
#   ROWS_LIST     rows to test           (default: 1000000 4000000 16000000)
#   BITS_LIST     bits per value to test (default: 8 16 32)
#   THREADS_LIST  thread counts to test  (default: 1 2 4 ... up to nproc)
#   LOOPS         measurement loops per run (default: 10)
#   PRED          predicate to use          (default: between)
#   OUT           output CSV file           (default: benchmark_results.csv)

set -u
cd "$(dirname "$0")"

BIN=./bitweaving
ROWS_LIST=${ROWS_LIST:-"1000000 4000000 16000000"}
BITS_LIST=${BITS_LIST:-"8 16 32"}
LOOPS=${LOOPS:-10}
PRED=${PRED:-between}
OUT=${OUT:-benchmark_results.csv}

# default threads: powers of two up to the number of cores
if [ -z "${THREADS_LIST:-}" ]; then
    NPROC=$(nproc)
    THREADS_LIST=""
    t=1
    while [ "$t" -le "$NPROC" ]; do
        THREADS_LIST="$THREADS_LIST $t"
        t=$((t * 2))
    done
fi

make -s all || exit 1

echo "Sweep: rows [$ROWS_LIST] x bits [$BITS_LIST] x threads [$THREADS_LIST]"
echo "Loops per run: $LOOPS, predicate: $PRED"
echo

echo "rows,bits,threads,avg_time_s,speedup_vs_1thread,avg_energy_j,verified" > "$OUT"

for n in $ROWS_LIST; do
    for b in $BITS_LIST; do
        base_time=""
        for t in $THREADS_LIST; do
            output=$($BIN -n "$n" -b "$b" -l "$LOOPS" -p "$PRED" -t "$t" 2>&1)
            if [ $? -ne 0 ]; then
                echo "run failed: -n $n -b $b -t $t"
                echo "$output" | tail -2
                echo "$n,$b,$t,ERROR,,,no" >> "$OUT"
                continue
            fi
            avg=$(echo "$output" | awk '/Average Time/ {print $3}')
            energy=$(echo "$output" | awk '/Average CPU Energy/ {print $4}')
            [ -z "$energy" ] && energy="n/a"
            if echo "$output" | grep -q "VERIFICATION FAILED"; then
                verified=no
            else
                verified=yes
            fi
            if [ -z "$base_time" ]; then
                base_time=$avg
            fi
            speedup=$(awk -v b="$base_time" -v a="$avg" 'BEGIN {printf "%.2f", b / a}')
            echo "$n,$b,$t,$avg,$speedup,$energy,$verified" >> "$OUT"
            printf "rows=%-9s bits=%-2s threads=%-2s avg=%-10ss speedup=%-5s energy=%-9s verified=%s\n" \
                   "$n" "$b" "$t" "$avg" "$speedup" "$energy" "$verified"
        done
    done
done

echo
echo "Results written to $OUT:"
awk -F, '{printf "%-10s %-6s %-9s %-12s %-19s %-13s %s\n", $1, $2, $3, $4, $5, $6, $7}' "$OUT"
