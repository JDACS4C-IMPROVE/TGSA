#!/bin/bash

set -a

if [ -z "$CANDLE_DATA_DIR" ]; then
  echo "CANDLE_DATA_DIR not set"
  exit 421
fi

python pilot_preprocessing.py
