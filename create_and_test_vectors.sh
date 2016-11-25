#!/bin/bash

mkdir LanguageModels

for i in europarl multiun opener; do
mkdir LanguageModels/$i
done

python3 Scripts/create_vec_reps.py
