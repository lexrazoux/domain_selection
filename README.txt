This folder consists of the datasets and code used in the paper 
"Distance Based Source Domain Selection for Automated Sentiment Classification" 
by Lex Razoux Schultz and Marco Loog, Peyman Mohajerin Esfahani in cooperation with Crunchr BV.



Reproducing results 
----------------------------------------------------------------------------------------------------------|
To reproduce the plots in the paper, run "main.py" for figure 3 and 4 and table 1.
These scripts first calculate the Chi2, MMD, EMD and KLD for all possible data set pairs and subsequently performs source domain selection
as described in the paper and results are plotted.

As calculating the distances between data sets is computational expensive, the Pickle files "measures_and_accuracies.p" and 
"measures_and_accuracies_corrupted" are added and are imported preliminary to the plotting code, the distance calculations can therefore be skipped.
Code only has to be run partly. Do run the first section first in order to be able to import the data sets.

The significance of all results are calculated in the "significance_ttest_all.py". Please be aware, this code is mainly written for private use.


Referencing
----------------------------------------------------------------------------------------------------------|
If code is used, please cite:

@unpublished{razouxschultz2018distance,
author = {Razoux Schultz, Lex E. and Loog, Marco and Mohajerin Esfahani, Peyman},
title = {Distance Based Source Domain Selection for Automated Sentiment Classification},
note = {Unpublished Manuscript},
year = {2018}
}



Files summary 
----------------------------------------------------------------------------------------------------------|
Name:			Type:			Content:
backup.p		pickle file		stored results that are used to construct figure 4 
datasets		folder			datasets used in experiments
classify.py		python script		library to calculate inner and cross domain classification error
main.py			python script		script to reproduce results, uses "classify.py", "measures.py" and "mmdxx.py"
measures.py		python script		library to calculate distances between datasets, uses "mmdxx.py"
mmdxx.py		python script		script to calculate the 
paper_fig3.png		png file		figure 3 of paper
paper_fig4.png		png file		figure 4 of paper
significance.p		pickle file		stored results for signifiance calculations
significance_ttest_all	python script		significance of the results calculations (raw code, not suitable for external use)
table1.txt		txt file		table 1 of paper
XI.p			pickle file		stored resuls for training on n random selected domains

Data sets
----------------------------------------------------------------------------------------------------------|
All data sets are publicly available and retrieved from different sources. When using them, please reference to their source.

Dataset 2:
@online{Michigan2017sentiment,
  
	author = {{University of Michigan}},
  
	title = {UMICH SI650 - Sentiment Classification},
  
	year = 2011,
  
	url = {https://www.kaggle.com/c/si650winter11/data},
  
	urldate = {2017-12-15}
}


Dataset 6,7,8:

@inproceedings{kotzias2015group,
  
	title={From group to individual labels using deep features},
  
	author={Kotzias, Dimitrios and Denil, Misha and De Freitas, Nando and Smyth, Padhraic},
  
	booktitle={Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  
	pages={597--606},
  
	year={2015},
  
	organization={ACM}
}




Dataset 3,4,5,9,10,11,12,13,14
@online{CrowdFlower2017data,
  
	author = {CrowdFlower},
  
	title = {Data for everyone},
  
	year = 2017,
  
	url = {https://www.crowdflower.com/data-for-everyone/},
  
	urldate = {2017-12-15}
}
