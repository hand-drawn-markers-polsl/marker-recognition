#!/bin/bash

for i in ../images/m*/**/*.jpg;
do
  magick "$i" "$i"
done
