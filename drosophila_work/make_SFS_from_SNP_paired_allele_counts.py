"""
reads a text file with SNP allele counts formatted like this:
    Neut_total_allele_count Neut_derived_count Selected_total_allele_count Selected_derived_count
    167	142 190 1
    162	11  165 2
    165	133 201 5
    
the first column is the total NEUTRAL allele count, followed by count of NEUTRAL derived alleles 
the third and fourth columns are for SELEECTED allele counts 
return a file with two SFSs, ready for input to SF_Ratios.py 
"""
import sys
import argparse
import os.path as op
import numpy as np 
import random
from scipy.stats import hypergeom


def stochastic_round(number):
    floor_number = int(number)
    # Probability of rounding up is the fractional part of the number
    if np.random.uniform() < (number - floor_number):
        return floor_number + 1
    else:
        return floor_number
    
def readfile(infilename):
    Nsnpcounts = {}
    Ssnpcounts = {}
    Nminnc = float('inf')
    Sminnc = float('inf')

    with open(infilename, "r") as f:
        for li,line in enumerate(f):
            if len(line) > 2 and (True if li > 1 else  line[0] in "0123456789"): # skip header line if it is present
                l = line.strip().split()
                if Nminnc > int(l[0]):
                    Nminnc = int(l[0])
                snc = int(l[0])
                sa = int(l[1])
                if Nminnc > snc:
                    Nminnc = snc
                if snc not in Nsnpcounts:
                    Nsnpcounts[snc] = {}
                if sa not in Nsnpcounts[snc]:
                    Nsnpcounts[snc][sa] = 0
                Nsnpcounts[snc][sa] += 1
                if Sminnc > int(l[2]):
                    Sminnc = int(l[2])
                snc = int(l[2])
                sa = int(l[3])
                if Sminnc > snc:
                    Sminnc = snc
                if snc not in Ssnpcounts:
                    Ssnpcounts[snc] = {}
                if sa not in Ssnpcounts[snc]:
                    Ssnpcounts[snc][sa] = 0
                Ssnpcounts[snc][sa] += 1                
    return Nsnpcounts,Nminnc,Ssnpcounts,Sminnc

def make_SFS_with_downsampling(args,snpcounts):
    sfs = [0 for i in range(args.nc+1)]
    for snc in snpcounts:
        for sa in snpcounts[snc]:
            pi = sa
            numg = snc
            for si in range(args.nc+1):
                prob = hypergeom.pmf(si,numg,pi,args.nc)
                sfs[si] += prob*snpcounts[snc][sa]
    if args.dostochasticround:
        sfs = list(map(stochastic_round,sfs))
    if args.fixedbin == False:
        return sfs[:-1]
    else:
        return sfs

def make_SFS_with_subsampling(args,snpcounts):
    sfs = [0 for i in range(nc+1)]
    for snc in snpcounts:
        for sa in snpcounts[snc]:
            pi = sa
            numg = snc    
            samples = np.random.hypergeometric(pi,numg-pi,args.nc,snpcounts[snc][sa])
            for c in samples:
                sfs[c] += 1
    if args.fixedbin == False:
        return sfs[:-1]
    else:
        return sfs

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",dest="dostochasticround",default = False,action="store_true",help = "if downsampling (-d) apply stochastic rounding to get an integer")
    parser.add_argument("-d",dest="downsample",action="store_true", default = False,help=" use hypergeometic downsampling")    
    parser.add_argument("-e",dest="seed",type = int, default = 1,help=" random number seed for sub-sampling and misspecification")    
    parser.add_argument("-f",dest="foldit",action="store_true",help="fold the resulting SFS")    
    parser.add_argument("-i",dest="infilename",required=True,type = str, help="input file path/name")  
    parser.add_argument("-n",dest="nc",type = int,required=True,help=" new sample size")   
    parser.add_argument("-o", dest="sfsfilename",required=True,type = str, help="Path for output SFS file")
    parser.add_argument("-m", dest="comment",default = None,type = str, help="comment for output file headers")
    parser.add_argument("-p", dest="population",default = "",type = str, help="population name")
    parser.add_argument("-s",dest="subsample",action="store_true", default =False,help=" use random subsampling")   
    parser.add_argument("-y",dest="fixedbin",default = False,action="store_true",help=" SFSs file includes fixed sites,  i.e. last value is count of # of fixed sites") 
    parser.add_argument("-z",dest="nozerobin",default = False,action="store_true",help=" SFSs in file begin at count 1,  not 0 ") 
    
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    if args.downsample is True and args.subsample is True:
        parser.error(' cannot do both -d and -s')
    if args.downsample is False and args.subsample is False:
        parser.error(' must use either -d or -s')
    if args.downsample is False  and args.dostochasticround is True:
        parser.error(' cannot do -c without -d')
    return args

def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    Nsnpcounts,Nminnc,Ssnpcounts,Sminnc = readfile(args.infilename)
    
    if args.downsample:
        Nsfs = make_SFS_with_downsampling(args,Nsnpcounts)
        Ssfs = make_SFS_with_downsampling(args,Ssnpcounts)
    elif args.subsample:
        Nsfs = make_SFS_with_subsampling(args,Nsnpcounts)
        Ssfs = make_SFS_with_subsampling(args,Ssnpcounts)

    nciseven = args.nc % 2 == 0
    bothSFSs = []
    for sfs in [Nsfs,Ssfs]:
        if args.foldit:
            if nciseven:
                fsfs = [0] + [sfs[j]+sfs[args.nc-j] for j in range(1,args.nc//2)] + [sfs[args.nc//2]]
            else:
                fsfs = [0] + [sfs[j]+sfs[args.nc-j] for j in range(1,1 + args.nc//2)] 
            if args.fixedbin: # fold bins 0 and -1
                fsfs[0] = sfs[0] + sfs[-1]
            else:
                fsfs[0] = sfs[0]
            rsfs = fsfs
        else:
            rsfs = sfs
        if args.nozerobin:
            rsfs = sfs[1:]
        bothSFSs.append(rsfs)
    fout = open(args.sfsfilename,'w')
    headers = ["Neutral_SI_SFS_{}_{}_{}_{}\n".format(args.population,args.nc,("folded" if args.foldit else "unfolded"),args.comment)]
    headers.append("Selected_SFS_{}_{}_{}_{}\n".format(args.population,args.nc,("folded" if args.foldit else "unfolded"),args.comment))
    for si,sfs in enumerate(bothSFSs):
        fout.write(headers[si])
        if args.downsample and args.dostochasticround == False:  # values are floats
            sfsline = " ".join(f"{x:.1f}" for x in sfs)
        else:
            sfsline = " ".join(map(str,sfs))
        fout.write(sfsline + "\n")
    fout.close()


if __name__ == '__main__':
    args = parsecommandline()
    run(args)
   