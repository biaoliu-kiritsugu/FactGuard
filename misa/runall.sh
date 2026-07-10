#!/bin/env bash

set -ex

# $1: python script to run
# extra args passed to python script
extra=${@:2}

python $1 --dataset ancient-book  $extra
python $1 --dataset chinese-law $extra
python $1 --dataset gutenberg $extra
python $1 --dataset pile-of-law $extra
