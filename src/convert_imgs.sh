#!/bin/sh

# This script helps to fix invalid photos shot on some phones
# (e.g. on iPhone XR), imagemagick is required.

# This globs for "mz*" because currently "mz" prefixed directories
# include photos that need fixing.
for i in data/raw/mz*/**/*.jpg; do
  echo Converting "$i"
  magick "$i" "$i"
done
