#!/bin/bash

run="$1"

pwd
echo "$run"/images
cd "$run"/images
ffmpeg -r 5 -i %04d.png -c:v libx264 ../video.mp4
rm -rf ../"images"
cd -