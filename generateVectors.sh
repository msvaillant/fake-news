#!/bin/bash
FILES=news/*
for filename in $FILES
do
    
    name=${filename##*/}
    base=${name%.news}
    ./Text2Vect.py emotion_words.json news/"$base.news"
done
