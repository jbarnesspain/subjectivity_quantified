#!/bin/bash

weka_dir=/home/jbarnes/Weka/weka-3-8-0/

cd $weka_dir


java -cp ./weka.jar weka.classifiers.meta.FilteredClassifier -t /home/jbarnes/Documents/Experiments/Subjectivity_Quantified/Weka_arffs/train.arff -T /home/jbarnes/Documents/Experiments/Subjectivity_Quantified/Weka_arffs/test.arff -o -F "weka.filters.unsupervised.attribute.StringToWordVector -R first-last -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\"\"" -W weka.classifiers.functions.SMO -- -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007" -calibrator "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4" > /home/jbarnes/Documents/Experiments/Subjectivity_Quantified/Weka_arffs/results.log
