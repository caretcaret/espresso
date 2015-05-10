#!/bin/sh

# argument is directory of images

for name in $1/*.jpg; do
  convert -resize 256x256\! $name $name.ppm
done
