#!/usr/bin/env python3
"""
    for NC and ZI drosophila populations 
    using SnpEff annotations, and ancestral allele 'AA'  in Vitors 2L,2R,3L,3R vcf files 
    and using introns < 85 bp trimmed -3 bases on each end 
    get the closest pairs of SNPs,  i.e. Nonynonymous and nearest short intron SNP,  or synonymous and nearest short intron SNP
    write each SNP with a minimum total allele count to a file 
    some chunks of this written by claude 
"""

import subprocess
import sys
import pysam
from itertools import combinations
from copy import deepcopy
import numpy as np
import os.path as op
import os
from bisect import bisect_left
import argparse
import pickle
import make_SFS_from_SNP_paired_allele_counts as makeSFSfile
from types import SimpleNamespace

def get_sequence(chrom, start, end, strand):
    """
    written by claude.ai 
    """
    """
    Extract sequence from a genomic interval using samtools faidx
    
    Args:
        chrom (str): Chromosome name
        start (int): Start position (0-based)
        end (int): End position
        strand (str): '+' or '-' strand
        fasta_path (str): Path to the indexed fasta file
    
    Returns:
        str: The DNA sequence for the interval
    """
    fasta_path = "../GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna"
    try:
        # Format region string for samtools (1-based coordinates)
        region = f"{chrom}:{start+1}-{end}"
        
        # Run samtools faidx command
        cmd = ['samtools', 'faidx', fasta_path, region]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the sequence from the output (skip header line)
        sequence = ''.join(result.stdout.split('\n')[1:]).strip().upper()
        
        # Reverse complement if on minus strand
        if strand == '-':
            complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 
                        'N': 'N', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
            sequence = ''.join(complement.get(base, base) for base in reversed(sequence))
            
        return sequence
        
    except subprocess.CalledProcessError as e:
        print(f"Error running samtools: {e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


from intervaltree import IntervalTree, Interval
import sys

def load_bed_to_tree(bed_file, target_chrom):
    """
    written by claude 

    Load BED file intervals for a single chromosome into an interval tree
    
    Args:
        bed_file (str): Path to BED file
        target_chrom (str): Chromosome to load
    
    Returns:
        IntervalTree: Interval tree for the specified chromosome
    """
    tree = IntervalTree()
    
    try:
        with open(bed_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Skip empty lines and comments
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    fields = line.split('\t')
                    if len(fields) < 3:
                        print(f"Warning: Line {line_num} has fewer than 3 fields: {line}", 
                              file=sys.stderr)
                        continue
                    
                    chrom = fields[0]
                    
                    # Skip if not our target chromosome
                    if chrom != target_chrom:
                        continue
                    
                    try:
                        start = int(fields[1])
                        end = int(fields[2])
                    except ValueError:
                        print(f"Warning: Invalid coordinates at line {line_num}: {line}", 
                              file=sys.stderr)
                        continue
                        
                    if start < 0 or end < start:
                        print(f"Warning: Invalid interval at line {line_num}: {start}-{end}", 
                              file=sys.stderr)
                        continue
                    
                    # Add interval using Interval object
                    tree.add(Interval(start, end))
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}", file=sys.stderr)
                    continue
            
            # Merge overlapping intervals
            tree.merge_overlaps()
                    
    except FileNotFoundError:
        print(f"Error: Could not find BED file: {bed_file}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error loading BED file: {e}", file=sys.stderr)
        raise
        
    if len(tree) == 0:
        print(f"Warning: No valid intervals found for chromosome {target_chrom}", 
              file=sys.stderr)
        
    return tree

def variant_in_interval(pos, tree):
    """
    Check if a variant position falls within any interval
    
    Args:
        pos (int): Position to check (1-based VCF coordinate)
        tree (IntervalTree): Interval tree from load_bed_to_tree()
    
    Returns:
        bool: True if position is in an interval, False otherwise
    """
    # Convert 1-based VCF coordinate to 0-based for interval tree
    pos_0based = pos - 1
    
    # Query the interval tree
    # overlaps = tree.overlap(pos_0based)
    overlaps = tree.overlap(pos_0based, pos_0based + 1)

    return len(overlaps) > 0


def revcomp(sequence):
    """
    Return the reverse complement of a DNA sequence in upper case.
    Input can be any case.
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
                 'a': 'T', 't': 'A', 'g': 'C', 'c': 'G',
                 'N': 'N', 'n': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence.upper()))

def build_SNP_types(args):
    """
    each type is a string of 4 letters
        base to the left of ancestral base
        SNP ancestral base
        SNP derived base
        base to the right of ancestral base
    there are 192 types
    but half are reverse complements of the other half,  so only 96 distinct types
    however, because the ancestral base is the second base even after taking reverse complement,  
    the reverse complement operation incluces swapping bases 2 and 3 


    snptypeRCdic is each snptype as a key and the reverse complement as a value 

    snptypeNUMdic is a dictionary with 192 entries,  keys are snptypes,  values are 0 thru 95,  a snptype and its reverse complement have the same value 

    if args.fofolding then there are only 48 numbers,  i.e.  for each SNP there are 4 possible combinations 
    """
    def snptypeRC(s):
        """s has 4 bases in it
            positions 1 and 2 are for the ancestral and derived bases respectively
            so we take the RC by taking the complment of both of these, but by keeping the complements values in positions 1 and 2 respectively 
        """

        return ''.join([revcomp(s[3]),revcomp(s[1]),revcomp(s[2]),revcomp(s[0])])
    
    def snptypeReOrdering(s):
        return s[0] + s[2] + s[1] + s[3]


    pairs = list(combinations(bases, 2))
    snptypes = []
    for base1 in bases:
        for pair in pairs:
            for base2 in bases:
                snptypes.append("".join([base1,pair[0],pair[1],base2]))
                snptypes.append("".join([base1,pair[1],pair[0],base2]))
    i = 0
    if args.forfolding == False:
        snptypeRCdic = {} # dictionary with 4 base snptypes as values, and 4 based RC, using ordered SNP types, as value 
        for snptype in snptypes:
            snptypeRCdic[snptype] = snptypeRC(snptype) 
        snptypeNUMdic = {} # a dictionary with a 4 base snptype as key and a number (0:96),  both a snptype and its RC, using ordered SNP types, get the same value
        for snptype in snptypes:
            assert snptype not in snptypeNUMdic
            if snptypeRCdic[snptype] in snptypeNUMdic: # if the RC of the snptype is in the dictionary,  then put the snptype in the dictionary with the save value as the RC
                snptypeNUMdic[snptype] = snptypeNUMdic[snptypeRCdic[snptype]]
            else:
                snptypeNUMdic[snptype] = i
                i += 1
    else: # for a folded SFS,  pool snptypes so that they reflect an unordered numbering system  
        snptypeRCdic = {} # dictionary with 4 base snptypes as values, and 4 based RC, using ordered SNP types, as value 
        for snptype in snptypes:
            snptypeRCdic[snptype] = snptypeRC(snptype) 
        snptypeROdic = {} # a dictionary with the altnernative ordering of a snp 
        for snptype in snptypes:
            snptypeROdic[snptype] = snptypeReOrdering(snptype)
        snptypeNUMdic = {} # a dictionary with a 4 base snptype as key and a number (0:96),  both a snptype and its RC, using ordered SNP types, get the same value
        for snptype in snptypes:
            assert snptype not in snptypeNUMdic
            RCsnptype = snptypeRCdic[snptype]
            ROsnptype = snptypeROdic[snptype]
            RORCsnptype = snptypeROdic[RCsnptype]
            RCROsnptype = snptypeRCdic[ROsnptype]
            assert snptype not in snptypeNUMdic
            if snptypeRCdic[snptype] in snptypeNUMdic: # if the RC of the snptype is in the dictionary,  then put the snptype in the dictionary with the save value as the RC
                snptypeNUMdic[snptype] = snptypeNUMdic[snptypeRCdic[snptype]]
            elif ROsnptype in snptypeNUMdic:
                snptypeNUMdic[snptype] = snptypeNUMdic[ROsnptype]
            elif RORCsnptype in snptypeNUMdic:
                snptypeNUMdic[snptype] = RORCsnptype 
            elif RCROsnptype in snptypeNUMdic:
                snptypeNUMdic[snptype] = RCROsnptype
            else:
                snptypeNUMdic[snptype] = i
                i += 1
    # for snptype in snptypes:
    #     i = snptypeNUMdic[snptype]
    #     # if  list(snptypeNUMdic.values()).count(i) == 2:
    #     #     print(snptype,i,list(snptypeNUMdic.values()).count(i))
    #     print(snptype,i,list(snptypeNUMdic.values()).count(i))
    return snptypes,snptypeRCdic,snptypeNUMdic

def get_SNP_type(record):
    startpos0based = record.pos - 1
    bases3 =get_sequence(chrlabels_chr_alt[record.chrom], startpos0based-1,startpos0based+2, "+")
    ancestor = record.info['AA']
    if ancestor != record.alleles[0]:
        derived = record.alleles[0]
        assert bases3[1] == derived
        # if bases3[1] == derived:
        #     pass
        assert ancestor == record.alleles[1]
    else:
        derived = record.alleles[1]
        assert bases3[1] == ancestor
        # if bases3[1] != ancestor:
        #     pass
        assert ancestor == record.alleles[0]
    snptype = ''.join([bases3[0],ancestor,derived,bases3[2]])
    return snptype
    



class snpinfo:
    def __init__(self, nc, nalt, pos):
        self.nc = nc
        self.nalt = nalt
        self.pos = pos
    
    def __lt__(self, other):
        # Handle both snpinfo objects and raw position values
        if isinstance(other, snpinfo):
            return self.pos < other.pos
        return self.pos < other
    
    def __eq__(self, other):
        if isinstance(other, snpinfo):
            return self.pos == other.pos
        return self.pos == other
    
# Function to find closest position
def find_unique_closest_pairs(list1, list2):
    """
    written by claude 
    
    Find unique pairs of closest positions between two sorted lists of snpinfo objects
    using NumPy for efficient distance calculations
    Returns list of tuples (index_in_list1, index_in_list2, distance)
    """
    # Convert positions to NumPy arrays for vectorized operations
    pos1 = np.array([snp.pos for snp in list1])
    pos2 = np.array([snp.pos for snp in list2])
    
    # Build distance matrix using broadcasting
    distances = np.abs(pos1[:, np.newaxis] - pos2)
    
    pairs = []
    mask = np.ones_like(distances, dtype=bool)
    
    while True:
        # Find minimum distance in remaining valid positions
        if not np.any(mask):
            break
            
        # Get indices of minimum value in masked array
        min_idx = np.argmin(np.where(mask, distances, np.inf))
        i, j = np.unravel_index(min_idx, distances.shape)
        
        # If no valid pairs left, break
        if distances[i, j] == np.inf:
            break
            
        # Add pair and update mask
        pairs.append((i, j, distances[i, j]))
        mask[i, :] = False  # replace values in  row to False
        mask[:, j] = False  # replace value in column to False 
    
    return pairs


def get_snp_pairs_write_files(SORN_snptype_dic,intron_snptype_dic,type = ""):
    numw = 0
    if type=="NONSYN":
        rf = open(NONSYN_SI_pair_results_filename,'a')
    else:
        rf = open(SYN_SI_pair_results_filename,'a')
    for snptypeNUM in SORN_snptype_dic:
        if snptypeNUM in intron_snptype_dic:
            SORN_snptype_dic[snptypeNUM].sort(key=lambda x: x.pos)
            intron_snptype_dic[snptypeNUM].sort(key=lambda x: x.pos)
            pairs = find_unique_closest_pairs(SORN_snptype_dic[snptypeNUM], intron_snptype_dic[snptypeNUM])
            for pair in pairs:
                rf.write("{}\t{}\t{}\t{}\n".format(intron_snptype_dic[snptypeNUM][pair[1]].nc,intron_snptype_dic[snptypeNUM][pair[1]].nalt,
                                                    SORN_snptype_dic[snptypeNUM][pair[0]].nc,SORN_snptype_dic[snptypeNUM][pair[0]].nalt))
                numw += 1
    rf.close()
    return numw

def parse_snp_string(s):
    inside_parens = s[s.find('(')+1:s.rfind(')')]
    return inside_parens.split('|')

def checkexpression(excutoff,tr_to_ex_dic,efftuple):
    meanex = 0.0
    numex = 0
    for effstr in efftuple:
        for checktr in parse_snp_string(effstr):
            if "FBtr" in checktr and checktr in tr_to_ex_dic:
                meanex += tr_to_ex_dic[checktr]
                numex += 1
    return numex > 0 and meanex/numex > excutoff

def run(args,vcffn,interval_tree,intron_snptype_dic,syn_snptype_dic,nonsyn_snptype_dic):
    debugmaxsnps = 20
    snptypes,snptypeRCdic,snptypeNUMdic=build_SNP_types(args)
    vcf_in = pysam.VariantFile(vcffn)
    nintronsnps = 0
    for record in vcf_in:
        # info = record.info
        # if variant_in_interval(pos, interval_tree)

        if len(record.alleles) > 2:
            continue 
        if len(record.alleles[0]) != 1 or len(record.alleles[1]) != 1:
            continue 
        gts = [s['GT'] for s in record.samples.values()]
        refcount = sum(x.count(0) for x in gts)//2
        altcount = sum(x.count(1) for x in gts)//2        
        if refcount + altcount < args.nc:
            continue 
        rikeys = record.info.keys()
        snpdic = None
        if 'AA' in rikeys and 'EFF' in rikeys:

            rituple = record.info['EFF']
            if args.useANYorALL == "ALL":
                if all("NON_SYNONYMOUS" in s for s in rituple):
                    if args.expressionfilter != None:
                        if checkexpression(args.expressionfilter,args.fbtr_to_ex,record.info['EFF']):
                            snpdic = nonsyn_snptype_dic
                    else:
                        snpdic = nonsyn_snptype_dic
                elif all("SYNONYMOUS" in s for s in rituple):
                    if args.expressionfilter != None:
                        if checkexpression(args.expressionfilter,args.fbtr_to_ex,record.info['EFF']):
                           snpdic = syn_snptype_dic                
                    else:
                        snpdic = syn_snptype_dic              
                elif all("INTRON" in s for s in rituple):
                    if variant_in_interval(record.pos, interval_tree):
                        snpdic =intron_snptype_dic
                        nintronsnps += 1
            elif args.useANYorALL == "ANY":
                if any("NON_SYNONYMOUS" in s for s in rituple):
                    if args.expressionfilter != None:
                        if checkexpression(args.expressionfilter,args.fbtr_to_ex,record.info['EFF']):
                            snpdic = nonsyn_snptype_dic
                    else:
                        snpdic = nonsyn_snptype_dic
                elif any("SYNONYMOUS" in s for s in rituple):
                    if args.expressionfilter != None:
                        if checkexpression(args.expressionfilter,args.fbtr_to_ex,record.info['EFF']):
                           snpdic = syn_snptype_dic                
                    else:
                        snpdic = syn_snptype_dic                
                elif any("INTRON" in s for s in rituple):
                    if variant_in_interval(record.pos, interval_tree):
                        snpdic =intron_snptype_dic
                        nintronsnps += 1
        if snpdic is not None:
            nonecount =  sum(x.count(None) for x in gts)
            snptype = get_SNP_type(record)
            snptypeNUM = snptypeNUMdic[snptype] 
            if snptypeNUM not in snpdic:
                snpdic[snptypeNUM] = []
                
            snpdic[snptypeNUM].append(snpinfo(altcount+refcount,altcount,record.pos))
        if args.debug and nintronsnps >= debugmaxsnps:
            break
    # Make a deep copy of the dictionary
    # intron_snptype_dic_copy = deepcopy(intron_snptype_dic)
    Nnumw = get_snp_pairs_write_files(nonsyn_snptype_dic,intron_snptype_dic,type = "NONSYN")
    Snumw = get_snp_pairs_write_files(syn_snptype_dic,intron_snptype_dic,type="SYN")
    return Nnumw, Snumw

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",dest="comment",required=True,type=str, help = "text for output file name with run detils,  no spaces")
    parser.add_argument("-d",dest="debug",action="store_true", default = False,help=" use debug mode")   
    parser.add_argument("-f",dest="forfolding",action="store_true", default = False,help=" SFS will be folded, treat ancestral and dervied the same")   
    parser.add_argument("-p",dest="population",required=True,type = str, help="NC or ZI") 
    parser.add_argument("-n",dest="nc",type = int, default = 160,help=" downsample size")   
    parser.add_argument("-u",dest="useANYorALL",required=True,type = str, help=" ANY or ALL")  
    parser.add_argument("-x",dest="expressionfilter",default = None,type=float, help = "optional value between 0 and 1 to filtering for higher gene expression, e.g. 0.9 means only genes with ranked expresson higher than 0.9")
  
    
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    return args


if __name__ == '__main__':
    args = parsecommandline()

    bases = ["A","C","G","T"]
    chrlabels = {"NC_004354.4":"X","NT_033779.5":"2L","NT_033778.4":"2R","NT_037436.4":"3L","NT_033777.3":"3R","NC_004353.4":"4"}
    chrlabelsalt = {chrlabels[key]:key for key in chrlabels}
    chrlabels_chr_alt = {"chr"+chrlabels[key]:key for key in chrlabels}
    beddir = "/mnt/d/genemod/better_dNdS_models/popgen/SF_ratios_manuscript/PlosGenetics_submit/revision/intron_work"
    bedfile = op.join(beddir,"dm6_short_trimmed_introns.bed")
    vcfdir = "/mnt/d/genemod/better_dNdS_models/popgen/SF_ratios_manuscript/PlosGenetics_submit/revision/vcfs"
    outdir = "/mnt/d/genemod/better_dNdS_models/popgen/SF_ratios_manuscript/PlosGenetics_submit/revision/SFSwork"
    sfsdir = "/mnt/d/genemod/better_dNdS_models/popgen/SF_ratios_manuscript/PlosGenetics_submit/revision/SFSwork/SF_Ratios_input"
    if args.expressionfilter != None:
        with open("/mnt/d/genemod/better_dNdS_models/popgen/SF_ratios_manuscript/PlosGenetics_submit/revision/FBgn_Expression.p", "rb") as f:
            fbtr_to_ex = pickle.load(f)
            args.fbtr_to_ex = fbtr_to_ex
    else:
        args.fbtr_to_ex = None
    if args.comment != "" and args.comment[0] != "_":
        args.comment = "_" + args.comment
    NONSYN_SI_pair_results_filename = op.join(outdir,"{}_SI_and_NONSYN_allele_counts_nc{}{}.txt".format(args.population,args.nc,args.comment))
    SYN_SI_pair_results_filename = op.join(outdir,"{}_SI_and_SYN_counts_nc{}{}.txt".format(args.population,args.nc,args.comment))
    NONSYN_SI_SFSs_filename = op.join(sfsdir,"{}_SI_and_NONSYN_nc{}{}_SFSs.txt".format(args.population,args.nc,args.comment))
    SYN_SI_SFSs_filename = op.join(sfsdir,"{}_SI_and_SYN_nc{}{}_SFSs.txt".format(args.population,args.nc,args.comment))
    #remove files as needed,  create new files, add headers 
    if op.exists(NONSYN_SI_pair_results_filename):
        os.remove(NONSYN_SI_pair_results_filename)
    temp = open(NONSYN_SI_pair_results_filename,"w")
    temp.write("Neut_total_allele_count\tNeut_derived_count\tSelected_total_allele_count\tSelected_derived_count\n")
    temp.close()

    if op.exists(SYN_SI_pair_results_filename):
        os.remove(SYN_SI_pair_results_filename)
    temp = open(SYN_SI_pair_results_filename,"w")
    temp.write("Neut_total_allele_count\tNeut_derived_count\tSelected_total_allele_count\tSelected_derived_count\n")
    temp.close()
    if args.debug:
        chridlist = ["2L"]
    else:
        chridlist = ["2L","2R","3L","3R"]
    for chrid in chridlist:
        print(chrid," ", end = "")
        interval_tree = load_bed_to_tree(bedfile,chrid)
        intron_snptype_dic = {}
        syn_snptype_dic = {}
        nonsyn_snptype_dic = {}
        chrNCBIid = chrlabelsalt[chrid]
        vcffn =  op.join(vcfdir,"{}_Chr{}_remade_rooted_lifted_filtered_ann.vcf".format(args.population,chrid))
        Nnumw, Snumw = run(args,vcffn,interval_tree,intron_snptype_dic,syn_snptype_dic,nonsyn_snptype_dic)    
        print(" {} {}".format(Nnumw,Snumw))
        pass
    #make SFS file 
    from types import SimpleNamespace
    sargs = SimpleNamespace()
    sargs.dostochasticround = False
    sargs.downsample = True
    sargs.subsample = False
    sargs.seed = 1 
    sargs.foldit = args.forfolding
    sargs.infilename = NONSYN_SI_pair_results_filename 
    sargs.sfsfilename = NONSYN_SI_SFSs_filename 
    sargs.nc = args.nc
    sargs.fixedbin = False
    sargs.nozerobin = False 
    sargs.comment = args.comment + "_NONSYN"
    sargs.population = args.population 
    sargs.commandstring = " ".join(sys.argv[1:])
    makeSFSfile.run(sargs)
    sargs.infilename = SYN_SI_pair_results_filename 
    sargs.sfsfilename = SYN_SI_SFSs_filename 
    sargs.population = args.population
    sargs.comment = args.comment + "_SYN"
    makeSFSfile.run(sargs)
