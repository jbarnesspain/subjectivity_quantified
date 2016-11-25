#!/bin/bash

corpora="europarl multiun"

cd opinionfinderv2.0/database/docs
for corp in $corpora; do
    mkdir $corp
done

#get europarl v7 corpus, keep english part and break into smaller sections for use with opinionfinder
cd europarl
curl http://www.statmt.org/europarl/v7/es-en.tgz | tar xvz
rm europarl-v7.es-en.es
split europarl-v7.es-en.en -l 10000
rm europarl-v7.es-en.en
cd ..

cd multiun
wget http://opus.lingfil.uu.se/download.php?f=MultiUN%2Fen-es.txt.zip -O temp.zip
unzip temp.zip
rm temp.zip
rm MultiUN.en-es.es
split MultiUN.en-es.en -l 10000
rm MultiUN.en-es.en
cd ..


cd ../../..

#get rt-polaritydata
mkdir rt-polaritydata
cd rt-polaritydata
wget https://www.cs.cornell.edu/people/pabo/movie%2Dreview%2Ddata/rt-polaritydata.tar.gz 
tar -xvfz rt-polaritydata.tar.gz
rm rt-polaritydata.tar.gz
cd ..

mv sentence_polarity opinionfinderv2.0/database/docs
mv wikipedia opinionfinderv2.0/database/docs
mv opener opinionfinderv2.0/database/docs
