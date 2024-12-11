"""
    generate a pickle of a dictionary with transcript ids as keys and expression level as values

    gff3 has one or more transcripts for each gene 
    gene expression data just uses FBgn (gene values)
    the vcf annotation uses transcript values 
    so if I have a  tr_to_g_dic with transcripts as keys and genes as values made from the gff3 file
    and if I have a g_to_ex_dic with genes as keys and expression values as values, 

    then I can build a tr_to_ex_dic 
    and lod this into 'get_short_intron_paired_SNP_allele_coutns.py'
    if I want to filter for expression level. 

    expression is a value between 0 and 1, based on ranks  with 1 being the highest 
    

   """
import pickle 
import os.path
import gzip

# Get paths in supervening folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# fbgn_fbtr_file = os.path.join(parent_dir, "FBgn_FBtr_values_biomart_export.txt")
fbgn_ex_file = os.path.join(parent_dir, "FBgn_values_and_sum_lifestage_expression_values.txt")
pickle_out = os.path.join(parent_dir, "FBgn_Expression.p")
gff3file = os.path.join(parent_dir, "Drosophila_melanogaster.BDGP6.46.113.gff3.gz")

with gzip.open(gff3file,'rt') as fin:    
    tr_to_g_dic = {}    
    for line in fin:        
        # print('got line', line)
        start = line.find(":FBgn")
        gresult = False
        if start != -1:
            end = line.find(";", start + 1)  # Find the next semicolon after ";FBgn"
            if end != -1:
                gresult = line[start + 1:end]  # Extract the substring, excluding the leading ";"
                # print(result)  # Output: FBgn0037870
        if gresult:
            start = line.find(":FBtr")
            tresult = False
            if start != -1:
                end = line.find(";", start + 1)  # Find the next semicolon after ";FBgn"
                if end != -1:
                    tresult = line[start + 1:end]  # Extract the substring, excluding the leading ";"
                    # print(result)  # Output: FBgn0037870
            if tresult:
                tr_to_g_dic[tresult] = gresult

# Read FBgn to Expression mapping
g_to_ex_dic = {}
g_to_exprop_dic_sorted = {}
with open(fbgn_ex_file) as f:
    for line in f:
        fields = line.strip().split()
        fbgn = fields[0]
        ex = float(fields[1])
        g_to_ex_dic[fbgn] = ex
g_to_ex_dic_sorted = dict(sorted(g_to_ex_dic.items(), key=lambda item: item[1]))
dl = len(g_to_ex_dic_sorted)
for vi,g in enumerate(g_to_ex_dic_sorted):
    g_to_exprop_dic_sorted[g] = (vi+1)/dl 


# Build FBtr to Expression dictionary
tr_to_ex_dic  = {}
for t in  tr_to_g_dic:
    g = tr_to_g_dic[t]
    if g in g_to_exprop_dic_sorted:
        tr_to_ex_dic[t] = g_to_exprop_dic_sorted[g]
# Save dictionary to pickle file
pickle.dump(tr_to_ex_dic, open(pickle_out, "wb"))

# for k in tr_to_g_dic:
#     if tr_to_g_dic[k] == 'FBgn0066084':
#         print(k)