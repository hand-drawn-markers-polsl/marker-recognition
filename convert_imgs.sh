#!/bin/bash

# This script helps to fix invalid photos shot on some phones
# (e.g. on iPhone XR)

for i in data/raw/m*/**/*.jpg; do
  echo Converting $i
  magick "$i" "$i"
done
