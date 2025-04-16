#!/bin/bash

DATA_DIR="./data/majorTOM"
START_DATE="2017-01-01"
END_DATE="2025-01-01"
SOURCES=("Core-S2L2A" "Core-S2L1C" "Core-S1RTC" "Core-DEM")

python3 majortom/download_world.py \
    --data-dir $DATA_DIR \
    --sources "${SOURCES[@]}" \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --cloud-cover 0 10 \
    --subset-name "rome" \
    --bbox 12.2 41.6 13.0 42.2 \
    --criteria "latest" \
    --n-samples 10 \
    --seed 42