#!/bin/bash

# This script helps to fix invalid photos shot on some phones
# (e.g. on iPhone XR)

# This globs for "m*" because currently "mz" prefixed directories
# include photos that need fixing.
for i in data/raw/m*/**/*.jpg; do
  echo Converting "$i"
  magick "$i" "$i"
done
