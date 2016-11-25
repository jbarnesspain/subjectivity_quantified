#!/bin/bash


function get_doclist {
    #This function creates a doclist which can be used by opinionfinder
    name=$1

    for i in $( ls opinionfinderv2.0/database/docs/$name )
    do
	echo database/docs/$name/$i
    done > opinionfinderv2.0/$name.doclist; }

function run_opinionfinder {
    doclist=$1
    for doc in $(less $doclist); do
    java -Xmx3g -classpath ./lib/weka.jar:./lib/stanford-postagger.jar:opinionfinder.jar opin.main.RunOpinionFinder $doc -s -r preprocessor,cluefinder,rulebasedclass,subjclass,polarityclass
    done
    }
#################################################################
# MAIN
#################################################################

corpora="europarl multiun opener sentence_polarity"

#create doclists for corpora
for corpus in $corpora; do
    get_doclist $corpus
done

#create directory for results
mkdir Subjectivity_Results

#change to opinionfinder dir to run opinionfinder
cd opinionfinderv2.0

#run opinion finder
for corpus in $corpora; do
    run_opinionfinder $corpus.doclist
done

python3 ../Scripts/test_subjectivity.py

#extract subjective corpora
mkdir database/docs/extracted_subjectivity_corpora
for i in europarl multiun
do
    mkdir database/docs/extracted_subjectivity_corpora/$i
done

python3 ../Scripts/get_subjective_portion_of_corpus.py
