"""
    for making changes to a text file with one or more SFSs 
    Any line in the text file with nonnumerica characters is a header
        other non-empty lines are SFSs
    can handle empty lines between SFSs 
    SFSs include the zero frequency bin, which can be 0 as the value, whatever it is, is ignored:
        if NOZEROBIN is true,  then an empty 0th bin is added to the front of the SFS when processing 
            the 0 bin is then removed after processing before writing to the output file
        regardless,  the 0 bin is not used for updating values of bins for non-zero frequencies

    usage: SFS_modifications.py [-h] [-c] [-d DOWNSAMPLE] [-e SEED] -f FOLDSTATUS -i SFSFILENAME [-o OUTFILEPATH] [-p MISSPEC] [-s SUBSAMPLE] [-z NOZEROBIN]

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
    -z NOZEROBIN    SFSs in file begin at count 1, not 0


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

    to apply misspecification,  then downsample and then fold 
        python SFS_modifications -i ./infile.txt -o ./outfile.txt  -p 0.1 -d 50 -f foldit -z 11 
    
    if data file SFSs start at bin 1,  then add -z  
        (output will also start at bin 1)

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


# this does not work 
# def get_number_of_sampled_genomes(sfs,isfolded):
#     numsamplingbins = len(sfs)-1 
#     sampsize = numsamplingbins + 1 
#     if isfolded:
#         numg = 2*numsamplingbins
#     else:
#         numg = numsamplingbins + 1 
#     return numg,numsamplingbins,sampsize

def get_number_of_sampled_genomes(sfs,isfolded):
    # numg  = # of sampled genomes or gene copies,  i.e. 2x # of diploid individuals
    leniseven = len(sfs) % 2 == 0
    if isfolded:
        if leniseven:
            numg = 2*(len(sfs)-1)
        else:
            numg = 2*len(sfs) - 1 
    else:
        numg = len(sfs)
        # numg = numsamplingbins + 1 
    return numg


def stochastic_round(number):
    floor_number = int(number)
    # Probability of rounding up is the fractional part of the number
    if np.random.uniform() < (number - floor_number):
        return floor_number + 1
    else:
        return floor_number


def misspecswap(sfs,misspec):
    checksum = sum(sfs)
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
    assert checksum == checknewsum
    return newsfs 

def subsample(original_sfs, subsampsize,numg):
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
    sampsize = numg
    newsfs = [0]*(subsampsize+1)
    for i in range(1,sampsize):
        for j in range(original_sfs[i]):
            newsfs[random_subsample_draw(i)] += 1
    newsfs = newsfs[:-1] # drop off the last bin which has fixed counts 
    newsfs[0] = 0  # fix the zero bin at zero as it does not get used 
    return newsfs 



def downsample(original_sfs, downsamp_numg,numg):
    """
        sample fro 0 copies up to, but not including downsamp_numg copies,  as these would be fixed in the sample
    """
    newsfs = [0]*(downsamp_numg+1)
    for pi,popcount in enumerate(original_sfs):
        for si in range(downsamp_numg+1):
            prob = hypergeom.pmf(si,numg,pi,downsamp_numg)
            newsfs[si] += prob * popcount
    newsfs = [round(a,2) for a in newsfs]
    newsfs = newsfs[:-1] # drop off the last bin which has fixed counts 
    newsfs[0] = 0  #fix the zero bin at zero as it does not get used 
    return newsfs

def read_file_to_lists(filename,nozerobin,fixedbin):
    x = []
    addnewline = False
    headers = []
    lines = open(filename,'r').readlines()
    for i, line in enumerate(lines):
        # Check if the line is the first line and contains non-numeric characters
        if any(c.isalpha() for c in line):
            headers.append(line.strip())
            continue
        # Skip empty lines
        if line.strip() == '':
            addnewline = i < len(lines)
            continue
        # Replace commas and tabs with spaces, then split on spaces and filter out empty strings
        spacer = ',' if ',' in line else ('\t' if '\t' in line else ' ')
        numbers = line.replace(',', ' ').replace('\t', ' ').split()
        # Convert the split strings to integers and append as a list to x
        # x.append([int(num) for num in numbers if num.isdigit()])
        sfs = [int(round(float(num))) for num in numbers]
        if fixedbin: # drop off the last value 
            sfs = sfs[:-1]
        if nozerobin:
            sfs.insert(0,0)
        x.append(sfs)
    if len(headers) > len(x):
        topheader = " ".join(headers[:-len(x)]) 
        headers = headers[-len(x):] 
    else:
        topheader = ""
    return topheader,headers, x,addnewline,spacer

def writefile(outfilename, args, topheader,headers, x,addnewline,spacer):
    of = open(outfilename,'w')
    of.write("SFS_modifications.py command line: {}\n".format(args.commandstring))
    if topheader != "":
       of.write("\tOriginal top header: {}\n".format(topheader))
    for si,sfs in enumerate(x):
        if len(headers)> si:
            of.write("{}\n".format(headers[si]))
        if args.nozerobin:
            sfs = sfs[1:]
        # of.write(spacer.join(map(str,sfs)) + "\n")
        of.write(spacer.join(map(lambda x: f"{x:.2f}", sfs)) + "\n")

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
    parser.add_argument("-y",dest="fixedbin",default = False,action="store_true",help=" SFSs file includes fixed sites,  i.e. last value is count of # of fixed sites") 
    parser.add_argument("-z",dest="nozerobin",default = False,action="store_true",help=" SFSs in file begin at count 1,  not 0 ") 
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
    topheader,headers,sfslist,addnewline,spacer = read_file_to_lists(args.sfsfilename,args.nozerobin,args.fixedbin)
    newsfslist = []
    for sfs in sfslist:
        numg = get_number_of_sampled_genomes(sfs,args.foldstatus=="isfolded")
        if args.misspec > 0.0:
            sfs = misspecswap(sfs,args.misspec)
        if args.downsample is not None:
            assert args.downsample < numg
            sfs = downsample(sfs, args.downsample,numg)
            if args.dostochasticround:
                sfs = [stochastic_round(v) for v in sfs]
            numg = get_number_of_sampled_genomes(sfs,args.foldstatus=="isfolded")
        if args.subsample is not None:
            assert args.subsample < numg + 1 
            sfs = subsample(sfs, args.subsample,numg)
            numg = get_number_of_sampled_genomes(sfs,args.foldstatus=="isfolded")
        nciseven = numg % 2 == 0
        if args.foldstatus == "foldit":
            if nciseven:
                sfs = [0] + [sfs[j]+sfs[numg-j] for j in range(1,numg//2)] + [sfs[numg//2]]
            else:
                sfs = [0] + [sfs[j]+sfs[numg-j] for j in range(1,1 + numg//2)] 
        newsfslist.append(sfs)
    writefile(args.outfilepath, args, topheader,headers, newsfslist,addnewline,spacer)


            

