#!/usr/bin/env bash

rm build/ -rf
rm bin/ -rf

mkdir -p build
cd build
cmake ..
make -j4
cd ..
