"""
from claude.ai

prompt:
 I've downloaded all gene descriptions. Here is a sample:
"
#bin    name    chrom    strand    txStart    txEnd    cdsStart    cdsEnd    exonCount    exonStarts    exonEnds    score    name2    cdsStartStat    cdsEndStat    exonFrames
9    NM_166811.2    chr4    -    1035287    1051378    1037277    1050519    15    1035287,1037475,1037645,1038358,1043467,1045155,1047263,1047452,1047634,1047822,1048503,1049186,1050297,1050454,1051258,    1037422,1037590,1037740,1038437,1043524,1045248,1047390,1047575,1047729,1048148,1048558,1049246,1050392,1050609,1051378,    0    CaMKII    cmpl    cmpl    2,1,2,1,1,1,0,0,1,2,1,1,2,0,-1,
9    NM_001169361.2    chr4    -    1035287    1051378    1037277    1050519    17    1035287,1037475,1037645,1038358,1040012,1043467,1045155,1047263,1047452,1047634,1047822,1048503,1049186,1050297,1050454,1050693,1051342,    1037422,1037590,1037740,1038434,1040078,1043524,1045248,1047390,1047575,1047729,1048148,1048558,1049246,1050392,1050609,1050759,1051378,    0    CaMKII    cmpl    cmpl    2,1,2,1,1,1,1,0,0,1,2,1,1,2,0,-1,-1,
9    NM_166810.5    chr4    -    1035287    1051378    1037277    1050519    14    1035287,1037475,1037645,1038358,1045155,1047263,1047452,1047634,1047822,1048503,1049186,1050297,1050454,1051342,    1037422,1037590,1037740,1038437,1045248,1047390,1047575,1047729,1048148,1048558,1049246,1050392,1050609,1051378,    0    CaMKII    cmpl    cmpl    2,1,2,1,1,0,0,1,2,1,1,2,0,-1,
9    NM_166812.3    chr4    -    1035287    1053703    1037277    1050519    16    1035287,1037475,1037645,1038358,1040012,1043467,1045155,1047263,1047452,1047634,1047822,1048503,1049186,1050297,1050454,1053497,    1037422,1037590,1037740,1038434,1040078,1043524,1045248,1047390,1047575,1047729,1048148,1048558,1049246,1050392,1050609,1053703,    0    CaMKII    cmpl    cmpl    2,1,2,1,1,1,1,0,0,1,2,1,1,2,0,-1,
9    NM_001014696.2    chr4    -    1035287    1053703    1037277    1050519    15    1035287,1037475,1037645,1038358,1043467,1045155,1047263,1047452,1047634,1047822,1048503,1049186,1050297,1050454,1053497,    
"
I need a script that pulls out all the introns in between the end of a cds coding region 
and the start of the next cds coding regions using cds using cdsStart and cdsEnd.  this script 
should write a BED format file of all these introns.  then it should sort the bedfile and for all 
overlapping introns,  do a merge that generates intervals that span the full extent of any overlapping pair.
    inputfile = "../intron_work/dm6_refseq_genes.txt"
"""
#!/usr/bin/env python3
#!/usr/bin/env python3

import sys
import csv
from collections import defaultdict

def parse_comma_list(s):
    """Parse comma-separated list of numbers, removing empty strings"""
    return [int(x) for x in s.strip(',').split(',') if x]

def validate_coordinates(start, end, gene_id, chrom):
    """Validate that coordinates are properly ordered"""
    if start >= end:
        print(f"Warning: Invalid coordinates (start >= end) in {gene_id} on {chrom}: {start} >= {end}", 
              file=sys.stderr)
        return False
    if start < 0 or end < 0:
        print(f"Warning: Negative coordinates in {gene_id} on {chrom}: {start}, {end}", 
              file=sys.stderr)
        return False
    return True

def get_coding_exon_indices(starts, ends, cds_start, cds_end):
    """Find indices of exons that overlap with CDS region"""
    coding_indices = []
    for i in range(len(starts)):
        if ends[i] > cds_start and starts[i] < cds_end:
            coding_indices.append(i)
    return coding_indices

def get_coding_introns(row):
    """Extract introns between coding exons, handling strand properly"""
    # Parse required fields
    chrom = row['chrom']
    if chrom not in ["chrX","chr4","chr2L","chr2R","chr3L","chr3R"]:
        return None
    strand = row['strand']
    gene_id = row['name']  # transcript ID for error reporting
    
    try:
        cds_start = int(row['cdsStart'])
        cds_end = int(row['cdsEnd'])
        exon_starts = parse_comma_list(row['exonStarts'])
        exon_ends = parse_comma_list(row['exonEnds'])
    except (ValueError, KeyError) as e:
        print(f"Error parsing coordinates for {gene_id} on {chrom}: {e}", 
              file=sys.stderr)
        return []
    
    # Validate CDS coordinates
    if not validate_coordinates(cds_start, cds_end, gene_id, chrom):
        return []
    
    # Find exons that overlap with CDS
    coding_indices = get_coding_exon_indices(exon_starts, exon_ends, cds_start, cds_end)
    
    if not coding_indices:
        return []
        
    introns = []
    
    # Sort indices based on strand
    if strand == '-':
        coding_indices.sort(reverse=True)
    else:
        coding_indices.sort()
    
    # Get introns between coding exons
    for i in range(len(coding_indices)-1):
        if strand == '+':
            curr_idx = coding_indices[i]
            next_idx = coding_indices[i+1]
            intron_start = exon_ends[curr_idx]
            intron_end = exon_starts[next_idx]
        else:  # minus strand
            curr_idx = coding_indices[i]
            next_idx = coding_indices[i+1]
            intron_start = exon_ends[next_idx]
            intron_end = exon_starts[curr_idx]
            
        # Ensure start is always less than end
        if intron_start > intron_end:
            intron_start, intron_end = intron_end, intron_start
            
        # Validate intron coordinates
        if validate_coordinates(intron_start, intron_end, gene_id, chrom):
            # Additional sanity checks
            if intron_end - intron_start > 100000:
                print(f"Warning: Very large intron ({intron_end - intron_start}bp) in {gene_id} on {chrom}", 
                      file=sys.stderr)
            introns.append((chrom, intron_start, intron_end, strand, gene_id))
    
    return introns

def main():
    # Read input file
    inputfile = "../intron_work/dm6_refseq_genes.txt"
    outputfile = "../intron_work/dm6_all_introns.bed"
    inf = open(inputfile,'r')
    introns = []
    reader = csv.DictReader(inf, delimiter='\t')
    
    # Process each gene
    for row in reader:
        temp = get_coding_introns(row)
        if temp:
            introns.extend(temp)
    
    # Output BED format, sorting by chromosome and start position
    introns.sort(key=lambda x: (x[0], x[1]))
    fout = open(outputfile,'w')
    fout.write("#chrom\tstart\tend\tgene_id\tstrand\n")
    for chrom, start, end, strand, gene_id in introns:
        fout.write(f"{chrom}\t{start}\t{end}\t{gene_id}\t0\t{strand}\n")
        # print(f"{chrom}\t{start}\t{end}\t{gene_id}\t0\t{strand}")
    inf.close()
    fout.close()

if __name__ == "__main__":
    main()