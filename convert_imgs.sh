#!/bin/bash

for i in data/raw/m*/**/*.jpg; do
  echo Converting $i
  magick "$i" "$i"
done
