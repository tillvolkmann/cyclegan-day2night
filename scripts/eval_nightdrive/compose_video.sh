#!/bin/bash
# makeVideos


if [ "$1" == '' ] || [ "$2" == '' ] || [ "$3" == '' ]  || [ "$4" == '' ]; then
    echo "Usage: $0 <folder> <frame_rate> <frame_basename> <output_suffix>";
    exit;
fi

# loop through every sub_folder in <folder> and combine all contained images into a video sub_folder.mp4
for folder in "$1"/*/; do
    # filename="$(cut -d'/' -f3 <<<"$folder")".mp4
    filename=$(basename ${folder})
    filename+="-${4}.mp4"
    echo $filename
    # ffmpeg -r "$2" -f image2 -i "$folder"/frame-%d.png -vcodec libx264 -crf 18 "$folder"/"$filename"
    ffmpeg -r "$2" -f image2 -i "${folder}${3}%d.png" -vcodec libx264 -crf 18 "$folder"/"$filename"
done
