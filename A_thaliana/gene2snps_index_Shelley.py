import os
import csv
from xml.etree.ElementInclude import default_loader

import numpy as np
from IPython import embed
import pandas as pd
import argparse
from utils import *


def main(args):
	
	# Loading file
	chr_ = load_file(args.gene2snp)
	bim  = pd.read_csv(args.bim, sep = '\t', header = None, names = ['chrom', 'name', '-', 'pos', '/', '.'])

	# Getting the mapping of the SNPs onto the genes
	mapping = gene_snp_mapping(chr_, bim)
	save_file(args.outfile, mapping)
	
	return 0



def gene_snp_mapping(chr_, bim):
	'''
	Function of obtaining the snps mapped onto each gene involved in the network

	Input
	----------
	chr_:                dictionary with chromosomes as keys. Values are two 
						 vectors, one containing the genes and one the respec-
						 tive positions they start and end on the genome
	bim:                 bim file
	Output
	----------
	gene_snps_positions: dictionary in which the keys are the genes' names and 
	                     the values are boolean arrays capturing SNPs belolng-
	                     ing 
	'''
	gene_snps_positions = {}
	pos = bim['pos'].values
	chromosome = bim['chrom'].values
	for num, chrom in enumerate(np.array(['Chr1', 'Chr2', 'Chr3', 'Chr4', 'Chr5'])):
		details = chr_[chrom]
		gene_list = details[1]
		for gene in gene_list:
			index = np.where(gene_list == gene)
			gene_limit = details[0][index]
			snp_index = ((pos >= gene_limit[0][0]) & (pos <= gene_limit[0][1]) & (chromosome == num+1))
			
			if (snp_index.sum() > 0): # just keeping genes with SNPs mapped on them
				gene_snps_positions[gene] = snp_index

	return gene_snps_positions



def parse_arguments():
	'''
	Definition of the command line arguments

	Input
	---------
	Output
	---------
	data:	folder name of the used data, i.e. a_thaliana is the folder 
			for A. thaliana data set.
	'''
	parser = argparse.ArgumentParser()
	# parser.add_argument('--gene2snp', default='/Users/ShelleyShu/Desktop/DeepGWAS/chr_gen_pos_dictionary.pkl',required = True)
	# parser.add_argument('--bim',  default='/Users/ShelleyShu/Desktop/DeepGWAS/X_genic/X_genic_0.1.bim',  required = True)
	# parser.add_argument('--outfile', default='/Users/ShelleyShu/Desktop/DeepGWAS/test.csv', required = True)
	parser.add_argument('--gene2snp', default='/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/chr_gen_pos_dictionary.pkl')
	parser.add_argument('--bim',  default='/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/X_genic/X_genic_0.1.bim')
	parser.add_argument('--outfile', default='/home/zixshu/DeepGWAS/DeepGWAS-GeneReps/A_thaliana/X_genic_0.1.pkl')
	args = parser.parse_args()

	return args


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)

