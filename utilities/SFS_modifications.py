"""
    for making changes to a text file with one or more SFSs 
    text file has a first line of text,  then one or more SFSs each on a single line
    can handle empty lines between SFSs 
    can handle an SFS starting with a 0 

    usage: SFS_modifications.py [-h] [-c] [-d DOWNSAMPLE] [-e SEED] -f FOLDSTATUS -i SFSFILENAME [-o OUTFILEPATH] [-p MISSPEC]
                            [-s SUBSAMPLE]

    options:
    -h, --help      show this help message and exit
    -c              if downsampling (-d) apply stochastic rounding to get an integer
    -d DOWNSAMPLE   downsampling, new sample size
    -e SEED         random number seed for sub-sampling and misspecification
    -f FOLDSTATUS   usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded'
    -i SFSFILENAME  Path for SFS file
    -o OUTFILEPATH  results file path/name
    -p MISSPEC      apply ancestral misspecifications, assumes the distribution is unfolded before the misspec is applied
    -s SUBSAMPLE    subsampling, new sample size


    to fold 
        python SFS_modifications -i ./infile.txt -o ./outfile.txt  -f foldit 
    
    to generate misspecified unfolded data from folded at a rate of 0.1 with random number seed 11
        python SFS_modifications -i ./infile.txt -o ./outfile.txt  -p 0.1 -e 11

    to generate downsampled data 
        python SFS_modifications -i ./infile.txt -o ./outfile.txt  -d 50

    to generate downsampled data and stochastically round to nearest integer
        python SFS_modifications -i ./infile.txt -o ./outfile.txt  -d 50 -c -z 11  

    to generate subsampled data with a random number seed 11 
        python SFS_modifications -i ./infile.txt -o ./outfile.txt  -s 50 -z 11

    to apply misspecification,  then downwsample and then fold 
        python SFS_modifications -i ./infile.txt -o ./outfile.txt  -p 0.1 -d 50 -f foldit -z 11 
    
    In cases of multiple flags
        does misspecification first
        then downsample or subsample
        does folding last 
"""
import sys
import argparse
import os.path as op
import numpy as np 
from scipy.stats import hypergeom


def getsampsize(sfs,isfolded):
    if sfs[0] == 0:
        zerostart = True
        numsamplingbins = len(sfs)-1 
    else:
        zerostart = False
        numsamplingbins = len(sfs)
    sampsize = numsamplingbins + 1 
    if isfolded:
        nc = 2*numsamplingbins
    else:
        nc = numsamplingbins + 1 
    return nc,zerostart,numsamplingbins,sampsize

def stochastic_round(number):
    floor_number = int(number)
    # Probability of rounding up is the fractional part of the number
    if np.random.uniform() < (number - floor_number):
        return floor_number + 1
    else:
        return floor_number


def misspecswap(sfs,misspec,zerostart):
    checksum = sum(sfs)
    if zerostart == False:
        sfs = [0] + sfs
    ncplus1 = len(sfs)
    newsfs = [0]*len(sfs)
    checknewsum = 0
    for i,c in enumerate(sfs):
        ncminusi = ncplus1 - i 
        for j in range(c):
            if np.random.uniform() < misspec:
                newsfs[ncminusi] += 1
            else:
                newsfs[i] += 1
            checknewsum += 1 
    
    if zerostart == False:
        newsfs = newsfs[1:]
    assert checksum == checknewsum
    return newsfs 

def subsample(original_sfs, subsampsize,nc,zerostart,numsamplingbins,sampsize):
    def random_subsample_draw(i):
        """
        Returns a random draw of the number of times an item is observed in a subsample.
        
        Parameters:
        i (int): Number of times the item was observed in the original sample.
        origsampsize (int): Size of the original sample.
        subsampsize (int): Size of the subsample, where subsampsize < origsampsize.
        
        Returns:
        int: The number of times the item is observed in the subsample.
        """
        # Ensure subsampsize < origsampsize
        if subsampsize >= sampsize:
            raise ValueError("subsampsize must be less than origsampsize")
        
        # The number of successes in the population (i),
        # the number of successes in the sample (subsampsize),
        # and the population size minus the number of successes (origsampsize-i) give us the parameters for the hypergeometric distribution.
        j = np.random.hypergeometric(ngood=i, nbad=sampsize-i, nsample=subsampsize, size=1)
    
        return j[0]

    newsfs = [0]*(subsampsize+1)
    if zerostart == False:
        original_sfs = [0] + original_sfs
    for i in range(1,sampsize):
        for j in range(original_sfs[i]):
            newsfs[random_subsample_draw(i)] += 1
    if zerostart == False:
        newsfs = newsfs[1:]
    else:
        newsfs[0] = 0 
    return newsfs 



def downsample(original_sfs, downsampsize,nc,zerostart,numsamplingbins,sampsize):
    newsfs = [0]*(downsampsize+1)
    for pi,popcount in enumerate(original_sfs):
        for si in range(downsampsize+1):
            prob = hypergeom.pmf(si,sampsize,pi,downsampsize)
            newsfs[si] += prob * popcount
    newsfs = [round(a,2) for a in newsfs]
    if zerostart:
        newsfs[0] = 0
    else:
        newsfs = newsfs[1:]
    return newsfs

def read_file_to_lists(filename):
    x = []
    addnewline = False
    header = ""
    lines = open(filename,'r').readlines()
    for i, line in enumerate(lines):
        # Check if the line is the first line and contains non-numeric characters
        if i == 0 and not line.strip().replace(',', '').replace('\t', '').replace(' ', '').isdigit():
            header = line.strip()
            continue
        # Skip empty lines
        if line.strip() == '':
            addnewline = i < len(lines)
            continue
        # Replace commas and tabs with spaces, then split on spaces and filter out empty strings
        spacer = ',' if ',' in line else ('\t' if '\t' in line else ' ')
        numbers = line.replace(',', ' ').replace('\t', ' ').split()
        # Convert the split strings to integers and append as a list to x
        x.append([int(num) for num in numbers if num.isdigit()])
    return header, x,addnewline,spacer

def writefile(outfilename, args, header, x,addnewline,spacer):
    header = "command line: {}  Original header: {}\n".format(args.commandstring,header)
    of = open(outfilename,'w')
    of.write(header)
    for sfs in x:
        of.write(spacer.join(map(str,sfs)) + "\n")
        if addnewline:
            of.write("\n")
    of.close()
        


def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",dest="dostochasticround",default = False,action="store_true",help = "if downsampling (-d) apply stochastic rounding to get an integer")
    parser.add_argument("-d",dest="downsample",type = int, default = None,help=" downsampling,  new sample size")    
    parser.add_argument("-e",dest="seed",type = int, default = None,help=" random number seed for sub-sampling and misspecification")    
    parser.add_argument("-f",dest="foldstatus",required=True,help="usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded' ")    
    parser.add_argument("-i", dest="sfsfilename",required=True,type = str, help="Path for SFS file")
    parser.add_argument("-o",dest="outfilepath",default = "", type=str, help="results file path/name")  
    parser.add_argument("-p",dest="misspec",type=float,default=0.0,help=" apply ancestral misspecifications, assumes the distribution is unfolded before the misspec is applied  ") 
    parser.add_argument("-s",dest="subsample",type = int, default = None,help=" subsampling,  new sample size")    
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    if args.misspec > 0.0 and args.foldstatus == "isfolded":
        parser.error(' sfs is folded (-f), but can not apply misspecification (-p) to a folded SFS')
    if args.downsample is not None and args.subsample is not None:
        parser.error(' cannot do both -d and -s')
    if args.downsample is None and args.dostochasticround == True:
        parser.error(' cannot do -c without -d')
    if args.seed == None and (args.misspec > 0 or args.subsample is not None or args.dostochasticround == True):
        parser.error (' -p, -s and -c  require a random number seed (-e)')
    return args

if __name__ == '__main__':

    args = parsecommandline()
    if args.seed is not None:
        np.random.seed(args.seed)
    header,sfslist,addnewline,spacer = read_file_to_lists(args.sfsfilename)
    newsfslist = []
    for sfs in sfslist:
        nc,zerostart,numsamplingbins,sampsize = getsampsize(sfs,args.foldstatus=="isfolded")
        if args.misspec > 0.0:
            sfs = misspecswap(sfs,args.misspec,zerostart)
        if args.downsample is not None:
            assert args.downsample < numsamplingbins + 1 
            sfs = downsample(sfs, args.downsample,nc,zerostart,numsamplingbins,sampsize)
            if args.dostochasticround:
                sfs = [stochastic_round(v) for v in sfs]
            nc,zerostart,numsamplingbins,sampsize = getsampsize(sfs,args.foldstatus=="isfolded")
        if args.subsample is not None:
            assert args.subsample < numsamplingbins + 1 
            sfs = subsample(sfs, args.subsample,nc,zerostart,numsamplingbins,sampsize)
            nc,zerostart,numsamplingbins,sampsize = getsampsize(sfs,args.foldstatus=="isfolded")
        if args.foldstatus == "foldit":
            if zerostart:
                sfs = [0] + [sfs[j]+sfs[nc-j] for j in range(1,nc//2)] + [sfs[nc//2]]
            else:
                sfs = [sfs[j-1]+sfs[nc-j-1] for j in range(1,nc//2)] + [sfs[nc//2-1]]
        newsfslist.append(sfs)
    writefile(args.outfilepath, args, header, newsfslist,addnewline,spacer)


            

