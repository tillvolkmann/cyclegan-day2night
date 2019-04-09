#!/bin/bash
set -e

# Parameters:
#   input_folder: directory containing the video
#   output_folder: specified relative to input folder
#   file_extension: file extension of the videos to convert
#
# Example usage:
#   split_video <input_folder> <output_folder> <file_extension>
#   ./split_video.sh /home/till/projects/git-forks/pytorch-CycleGAN-and-pix2pix/results/videos  /home/till/projects/git-forks/pytorch-CycleGAN-and-pix2pix/results/videos mp4

# If called without arguments, provide usage information
if [ "$1" == '' ] || [ "$2" == '' ] || [ "$3" == '' ]; then
    echo "Usage: $0 <input_folder> <output_folder> <file_extension>";
    exit;
fi

# Split all videos in folder and store frames in separate sub-directories
for file in "$1"/*."$3"; do
    destination="$2${file:${#1}:${#file}-${#1}-${#3}-1}";

    echo "file: ${file}"
    echo "destination: ${destination}"

    mkdir -p "$destination";
    ffmpeg -i "$file" -start_number 0 "$destination/frame-%d.png";
done