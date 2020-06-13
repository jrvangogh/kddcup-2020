#! /usr/local/bin/bash

# Run from inside kddcup-2020/
zip -rj result.zip model -x '*.git*' -x '*__pycache__*'