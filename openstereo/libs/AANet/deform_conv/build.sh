#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
