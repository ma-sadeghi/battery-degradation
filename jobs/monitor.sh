#!/bin/bash

file="gra95-13616827.out"

# Print last line only if changed (max frequency = 1 second)
# prev_line=""
# 
# while true; do
#     last_line=$(tail -n 1 $file)
#     if [ "$last_line" != "$prev_line" ]; then
#         echo -ne "\r$last_line\033[K"
#         prev_line=$last_line
#     fi
#     sleep 1
# done

# Print last line every second
while true; do
    last_line=$(tail -n 1 "$file")
    echo -ne "\r$last_line\033[K"
    sleep 1
done
