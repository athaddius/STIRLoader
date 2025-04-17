#!/bin/bash
TESTDIR=$1
while [[ -z "$INPUT" ]]
do
  file=$(find $TESTDIR -iname "*.mp4" | grep -v left | shuf -n 1)
  echo "$file\n"
  vlc $file &> /dev/null
  read -s -p "Type enter to continue, or anything to continue" INPUT
done
