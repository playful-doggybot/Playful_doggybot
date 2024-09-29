#!/bin/bash

folder="$1"
run="$2"

cd logs/"$folder"/"$run"/images
ffmpeg -r 50 -i %04d.png -c:v libx264 ../"$run".mp4
mpv ../"$run".mp4
xdg-open ..
rm -rf ../"images"
cd -