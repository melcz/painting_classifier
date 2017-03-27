#!/bin/sh
 
max_images=250
url="http://www.abcgallery.com"
painter="$1"
letter=${painter:0:1}
letter=${letter^^}
echo letter
for i in {1..30}
do
	value=$[RANDOM%$max_images+1];
	link="${url}/${letter}/${painter}/${painter}${value}.JPG"
    echo "DOWNLOADING: $link"
    wget $link -P ./paintings/$painter
done
