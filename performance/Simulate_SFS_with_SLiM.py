"""
    runs slim SFS simulations,  generates folded SFSs for neutral and for selected sites 
    sample size n is diploid (as slim expects) and the resulting SFS is folded 
"""
import os
import sys
import argparse
import csv
import time
# import shlex, subprocess
import subprocess
import numpy as np
import math
import socket

# This might fix the RuntimeWarning with np.divide
np.seterr(divide='ignore', invalid='ignore')

#---FUNCTION DEFINITION---#

# Function to read one SFS from a file
def readSFS(file):
    for line in open (file, 'r'):
        if len(line) > 1 and line[0] != "#":
            try:
                sfs = line.strip().split()
                sfs = [int(x) for x in sfs]
            except:
                print(sfs)
                pass
    return sfs


# Function to run SLiM as a external process
def runSlim(simulation, mu, rec, popSize, seqLen, ns, diploidsamplesize, modelfile,nsdist, param1, param2, outdir, donotcleandir = False):
    # Path to SLiM
    # slim = "/usr/local/bin/slim"
    # modelsdir = "/Users/tur92196/WorkDir/prfratio/slim/models"
    host = socket.gethostname()
    if host=="compute":
        slim = "/home/tuf29449/download/SLiM/build/slim"
        modelsdir ="/home/tuf29449/prfratio/models"
    else:
        slim = "/usr/bin/slim"
        # modelsdir = "/mnt/d/genemod/better_dNdS_models/popgen/prfratio/slim/models"
        modelsdir = "/mnt/d/genemod/better_dNdS_models/popgen/PRF-Ratio/slim_work/models"

    # # Model dictionary
    # avail_models = {"constant": "Constant size model",
    #                 "iexpansion": "Instantaeous expansion model",
    #                 "ibottlneck": "Instantaneous bottleneck model",
    #                 "popstructure": "Constant size, population structure model",
    #                 "popstructureN2": "N/2 population size after split, population structure model",
    #                 "OOAgravel2011": "Gravel et al. 2011 Out-of-Africa Human demography model"}
    
    # # if model in avail_models:
    # #         print("Ok, " + avail_models.get(model) + " is available!")
    # # else:
    # #     print("Sorry, model " + model + " does not exists!")
    # #     sys.exit()
    # if model not in avail_models:
    #     print("Sorry, model " + model + " does not exists!")
    #     sys.exit()        

    # Distribution of fitness effects dict
    avail_nsdist = {"fixed": "Fixed 2Ns values",
                    "lognormal": "2Ns values sample from a lognormal with param1=meanlog, param2=sdlog",
                    "gamma": "2Ns values sample from a gamma with param1=shape, param2=scale"}
    if nsdist in avail_nsdist:
            # print("Ok, " + avail_nsdist.get(nsdist) + " is available!")
            if nsdist != "fixed":
                if param1 + param2 == 0.0:
                    print("You selected a different distribution, but did not changed the default values. Are you sure you want to proceed with no selection?")
                    sys.exit()

                # This enforces non-neutral mutations starts neutrally 
                # because s values are calculate with a custom function
                # for lognormal or gamma (not the ones built-in SLiM)
                elif ns != 0:
                    ns = 0
            
            else:
                # This enforces the logic of not using these parameters for fixed 2Ns simulations
                # The values doens't matter becuase they are not used in the _fixed.slim simulations
                if param1 + param2 != 0:
                    param1 = param2 = 0
    else:
        print("Sorry, DFE " + nsdist + " is not supported!")
        sys.exit()

    # Sample a seed every function call
    seed = str(int(np.random.uniform(low=100000000, high=900000000)))
    #seed = str(123456) # debugging only
    
    # Run SLiM as a python subprocess
    run = subprocess.run([slim, "-s", seed, "-d", ("simu="+str(simulation)), "-d", ("MU="+str(mu)), "-d", ("R="+str(rec)),
                          "-d", ("N="+str(popSize)), "-d", ("L="+str(seqLen)), "-d", ("Ns="+str(ns)), "-d", ("n="+str(diploidsamplesize)),
                          "-d", ("param1="+str(param1)), "-d", ("param2="+str(param2)),
                          "-d", ("outDir="+"'"+outdir+"'"), 
                          modelsdir + "/" + modelfile + ".slim"
                        #   (modelsdir + "/" + model + "_" + nsdist + ".slim")
                          ], capture_output=True)
    if run.stderr != b'':
        print("slim problem ",run.stderr)
        exit()
    neutralsfsfile = (outdir + "/" + ("sfs_neutral_" + str(simulation) + "_" + seed + ".txt"))
    selectedsfsfile = (outdir + "/" + ("sfs_selected_" + str(simulation) + "_" + seed + ".txt"))
    if not os.path.exists(neutralsfsfile):
        neutralsfsfile = (outdir + "/" + ("fsfs_neutral_" + str(simulation) + "_" + seed + ".txt"))
    if not os.path.exists(selectedsfsfile):
        selectedsfsfile = (outdir + "/" + ("fsfs_selected_" + str(simulation) + "_" + seed + ".txt"))
    neutral_sfs = readSFS(file = neutralsfsfile)
    selected_sfs = readSFS(file = selectedsfsfile)
    if donotcleandir==False:
        os.remove(neutralsfsfile)
        os.remove(selectedsfsfile)
    
    return seed, neutral_sfs, selected_sfs, run.stderr, run.stdout

def combineSFSs(sfslist, nbins):
    nbins = len(sfslist[0])
    csfs = [0]*(nbins)
    for li, list in enumerate(sfslist):
        for bi, bin in enumerate(list):
            csfs[bi] += bin
    return csfs

def writeCombinedSFS(file, header, csfs):
    with open(file, 'w') as f:
        f.write(header + "\n")
        wr = csv.writer(f, delimiter=" ")
        wr.writerows(csfs)
   
# Function to combine individuals SFSs in a file (each one in a row) ## NEED to FIX it??
def readANDcombineSFSs(diploidsamplesize, filename, path):
    
    csfs = [0]*(diploidsamplesize)
    file = (path + "/" + filename + ".txt")
    for line in open(file, 'r'):
        if len(line) > 1 and line[0] != "#":
            try:
                sfs = line.strip().split()
                for i in range(len(csfs)): 
                    csfs[i] += int(sfs[i])
            except:
                print(sfs)
                pass
    return csfs

#---PROGRAM DEFINITION---#
# Pipeline to run multiple sequences (or genes, chrms, etc) for multiples simulations (or replicates)
# It is a wrapper for SLiM code and it allows run different models with different parameters, especially with Ns
# Each replicate (or simulated SFS) is actually a combination of many sequence or gene SFSs#.
def simulateSFSslim(nsimulations = 3, mu = 1e-6/4, rec = 1e-6/4, popSize = 1000, seqLen = 10000, 
                     ns = 0.0, nsdist = "lognormal", nsdistargs = [1.0, 1.0], diploidsamplesize = 40,isfolded = False,
                     modelfile = "constant", nSeqs = 5, parent_dir = "results/prfratio", savefile = True, optdirstr=False, donotcleandir = False):

    # Constant value parameters:
    # Intron length and total intron size
    intronL = 810
    intron_totalL = (8*intronL) + 928 # Hard-coded
    
    # Exon length and total exon size
    exonL = 324
    exon_totalL = 8*exonL

    # SFS format
    if isfolded:
        sfs_format = "folded"
        numbins = diploidsamplesize-1
    else:
        sfs_format = "unfolded"
        numbins = 2*diploidsamplesize-1
    
    # Define thetas 

    thetaNeutral = (4*popSize*mu)*intron_totalL*nSeqs
    if "ibottleneck" in modelfile:
        thetaNeutral = thetaNeutral/10
    if "iexpansion"  in modelfile:
        thetaNeutral = thetaNeutral * 10
    if "OOAgravel2011" in modelfile:
        thetaNeutral = thetaNeutral * 122.24
    
    # Selected theta
    thetaSelected = (4*popSize*mu)*exon_totalL*nSeqs
    if "ibottleneck" in modelfile:
        thetaSelected = thetaSelected /10
    if "iexpansion" in modelfile:
        thetaSelected  = thetaSelected * 10
    if "OOAgravel2011" in modelfile:
        thetaSelected = thetaSelected * 122.24

    # Second, check if the distribution exists
    # and if parameters are correct!
    avail_nsdist = {"fixed": "Fixed 2Ns values",
                    "lognormal": "2Ns values sample from a lognormal with param1=meanlog, param2=sdlog",
                    "gamma": "2Ns values sample from a gamma with param1=shape, param2=scale"}
    
    if True : #nsdist in avail_nsdist:
        print("Ok, " + avail_nsdist.get(nsdist) + " exists")
        if nsdist != "fixed":
            # Combine output directory
            if optdirstr:
                # path = os.path.join(parent_dir, modelfile, nsdist, optdirstr,(str(nsdistargs[0]) + "-" + str(nsdistargs[1]))) 
                path = os.path.join(parent_dir, modelfile, optdirstr,(str(nsdistargs[0]) + "-" + str(nsdistargs[1]))) 
            else:
                # path = os.path.join(parent_dir, modelfile, nsdist, optdirstr,(str(nsdistargs[0]) + "-" + str(nsdistargs[1]))) 
                path = os.path.join(parent_dir, modelfile, (str(nsdistargs[0]) + "-" + str(nsdistargs[1]))) 
            # Prapare the headers for each model/Ns simulation
            header_neutral = "# 4Nmu(intron total length)={t} distribution={nsdist} dist_pars={nsdistargs} n={n} Neutral {sfs} SFS".format(t=thetaNeutral, nsdist=nsdist, nsdistargs=nsdistargs, n=diploidsamplesize, sfs=sfs_format)
            header_selected = "# 4Nmu(exon total length)={t} distribution={nsdist} dist_pars={nsdistargs} n={n} Selected {sfs} SFS".format(t=thetaSelected,nsdist=nsdist, nsdistargs=nsdistargs, n=diploidsamplesize, sfs=sfs_format)
        else:
            # Combine output directory
            if optdirstr:
                # path = os.path.join(parent_dir, modelfile, nsdist,optdirstr, str(ns)) 
                path = os.path.join(parent_dir, modelfile, optdirstr, str(ns)) 
            else:
                # path = os.path.join(parent_dir, modelfile, nsdist,str(ns)) 
                path = os.path.join(parent_dir, modelfile, str(ns)) 

            # Prapare the headers for each model/Ns simulation
            header_neutral = "# 4Nmu(intron total length)={t} distribution={nsdist} Ns={ns} n={n} Neutral {sfs} SFS".format(t=thetaNeutral, nsdist=nsdist, ns=ns, n=diploidsamplesize, sfs=sfs_format)
            header_selected = "# 4Nmu(exon total length)={t} distribution={nsdist} Ns={ns}  n={n} Selected {sfs} SFS".format(t=thetaSelected, nsdist=nsdist, ns=ns, n=diploidsamplesize, sfs=sfs_format)

    else:
        print("Sorry, Ns distribution " + nsdist + " does not exists!")
        sys.exit()

    # Check if output already exists
    if not os.path.exists(path):
        os.makedirs(path)

    if nsimulations == 1:
        simulation = 1
        list_neutral_sfss = []
        list_selected_sfss = []
        list_seeds = []

        j = 0
        while j < nSeqs:
            # RUN SLiM
            seed, neutral_sfs, selected_sfs, stderr, stdout = runSlim(simulation=simulation,mu=mu,rec=rec,popSize=popSize,seqLen=seqLen,ns=ns,diploidsamplesize=diploidsamplesize,model=modelfile,nsdist=nsdist,param1=nsdistargs[0],param2=nsdistargs[1],outdir=path,donotcleandir=donotcleandir) 

            list_neutral_sfss.append(neutral_sfs)
            list_selected_sfss.append(selected_sfs)
            list_seeds.append(seed)

            j += 1

        csfs_neutral = combineSFSs(list_neutral_sfss, nbins=numbins)
        csfs_selected = combineSFSs(list_selected_sfss, nbins=numbins)
        # calculate the ratio
        csfs_ratio = np.divide(csfs_selected, csfs_neutral).tolist()
        csfs_ratio = [0 if math.isnan(x) else x for x in csfs_ratio]
        
        # insert 0 to the 0-bin
        csfs_neutral.insert(0,0) 
        csfs_selected.insert(0,0) 
        csfs_ratio.insert(0,0)

        # Write files containing the simulation combined SFS 
        # One SFS for each simulation (that is a aggregation of subsimulations SFS)
        if savefile:
            sim_csfs_neutral = []
            sim_csfs_selected = []
            sim_csfs_neutral.append(csfs_neutral)
            sim_csfs_selected.append(csfs_selected)
            # Define the output file for the combined SFS(s):
            sims_csfs_neutralfile = (path + "/" + "csfs_neutral.txt")
            sims_csfs_selectedfile = (path + "/" + "csfs_selected.txt")
            writeCombinedSFS(sims_csfs_neutralfile, header_neutral, sim_csfs_neutral)
            writeCombinedSFS(sims_csfs_selectedfile, header_selected, sim_csfs_selected)
        
        # Return statement
        return csfs_neutral, csfs_selected, csfs_ratio, list_seeds

    else: 

        # These lists collects each simulation (combined) SFS and lists of seeds
        # for each gene/fragment/subsimulation
        sims_seeds = []
        sims_csfs_neutral = []
        sims_csfs_selected = []
        sims_csfs_ratio = []

        # Run many replicates
        for i in range(0,nsimulations):
            simulation = i
            # These lists collects the output produced by each simulated gene/fragment/subsimulation
            list_neutral_sfss = []
            list_selected_sfss = []
            list_seeds = []

            j = 0
            while j < nSeqs:
                # RUN SLiM
                seed, neutral_sfs, selected_sfs, stderr, stdout = runSlim(simulation=simulation,mu=mu,rec=rec,popSize=popSize,seqLen=seqLen,ns=ns,diploidsamplesize=diploidsamplesize,modelfile=modelfile,nsdist=nsdist,param1=nsdistargs[0],param2=nsdistargs[1],outdir=path,donotcleandir=donotcleandir) 

                list_neutral_sfss.append(neutral_sfs)
                list_selected_sfss.append(selected_sfs)
                list_seeds.append(seed)

                j += 1

            csfs_neutral = combineSFSs(list_neutral_sfss, nbins=numbins)
            csfs_selected = combineSFSs(list_selected_sfss, nbins=numbins)            

            # calculate the ratio
            csfs_ratio = np.divide(csfs_selected, csfs_neutral).tolist()
            csfs_ratio = [0 if math.isnan(x) else x for x in csfs_ratio]
            
            # insert 0 to the 0-bin
            csfs_neutral.insert(0,0)
            csfs_selected.insert(0,0)
            csfs_ratio.insert(0,0)
            
            # Create a list of outputs (one for each simulation)
            sims_csfs_neutral.append(csfs_neutral)
            sims_csfs_selected.append(csfs_selected)
            sims_csfs_ratio.append(csfs_ratio)
            sims_seeds.append(list_seeds)
    
        # Write files containing the simulation combined SFS 
        # One SFS for each simulation (that is a aggregation of subsimulations SFS)
        if savefile:
            # Define the output file for the combined SFS(s):
            sims_csfs_neutralfile = (path + "/" + "csfs_neutral.txt")
            sims_csfs_selectedfile = (path + "/" + "csfs_selected.txt")
            writeCombinedSFS(sims_csfs_neutralfile, header_neutral, sims_csfs_neutral)
            writeCombinedSFS(sims_csfs_selectedfile, header_selected, sims_csfs_selected)
        
        # Return statement
        return sims_csfs_neutral, sims_csfs_selected, sims_csfs_ratio, sims_seeds

# Define the command line arguments
def parsecommandline():
    parser = argparse.ArgumentParser("python simulate_SFS_withSLiM.py",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", help="number of simulations",
                        dest="nsimulations", type=int, required=True)
    parser.add_argument("-R", help="Per site recombination rate per generation",
                        dest="rec", default=1e-6/4, type=float)
    parser.add_argument("-U", help="Per site mutation rate per generation",
                        dest="mu", default=1e-6/4, type=float)
    parser.add_argument("-N", help="Population census size",
                        dest="popSize", default=1000, type=int)
    parser.add_argument("-L", help="Sequence length",
                        dest="seqLen", default=10000, type=int)
    parser.add_argument("-k", help="Number of slim runs per simulation",
                        dest="nSeqs", type=int, required=True)
    parser.add_argument("-f",dest="isfolded",action="store_true",default=False,help="Make folded sfs,  default is unfolded")
    parser.add_argument("-d", dest="nsdist", default=None, help="set a distribution, fixed, lognormal, or gamma")
    parser.add_argument("-g", help="Non-synonymous population selection coefficient 2Ns (for fixed values DFE)",
                        dest="ns",default=0.0, type=float)
    parser.add_argument("-a", help="Set the parameters of the chosen distribution",
                        dest="nsdistargs", nargs= "+", default = [0.0, 0.0], type=float)
    parser.add_argument("-n", help="Diploid sample size, i.e. half the # of chromosomes",
                        dest="diploidsamplesize", default=40, type=int)
    parser.add_argument("-m", help="name of the slim model file,  e.g. 'constant'  for constant.slim",
                        dest="modelfile", default="constant", #required = True,
                        type = str)
    parser.add_argument("-s", help="Save simulated SFS to a file",
                        dest="savefile",default = True, type = bool)
    parser.add_argument("-b",help="optional name of output subdirectory",dest="optdirstr",default = False)
    parser.add_argument("-c",dest="donotcleandir",action="store_true",default=False,help="do not delete temporary sfs files, default is to delete")
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    return args

def main(argv):
    starttime = time.time()
    args = parsecommandline()

    nsimulations = args.nsimulations
    mu = args.mu
    rec = args.rec
    popSize = args.popSize
    seqLen = args.seqLen
    ns = args.ns
    nsdistargs = args.nsdistargs
    diploidsamplesize = args.diploidsamplesize
    modelfile = args.modelfile
    nSeqs = args.nSeqs
    parent_dir = "./slim_work/output"
    savefile = args.savefile
    optdirstr = args.optdirstr
    if args.nsdist == None:
        nsdist = modelfile.split(sep="_")[1]
    else:
        nsdist = args.nsdist

    nfsfs, sfsfs, fsfs_ratio, seeds = simulateSFSslim(nsimulations = nsimulations, mu = mu, rec = rec, popSize = popSize, seqLen = seqLen, 
                                                      ns = ns, nsdist = nsdist, nsdistargs = nsdistargs, diploidsamplesize = diploidsamplesize,isfolded = args.isfolded,
                                                      modelfile = modelfile, nSeqs = nSeqs, parent_dir = parent_dir, savefile = True,
                                                      optdirstr=optdirstr,donotcleandir=args.donotcleandir)

    print("Run finished!!!")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        main(['-h'])
    else:
        main(sys.argv[1:])
        