"""
Program:  PRF_Ratio.py
Author: Jody Hey
Description/Publication: 

reads a file with SFSs 
runs estimators

usage: PRF_Ratios.py [-h] -a SFSFILENAME [-c FIX_THETA_RATIO] [-d DENSITYOF2NS] [-D] [-q THETANSPACERANGE] [-M MAXI] [-o] [-z] [--profile] [-e] -f FOLDSTATUS [-g]
                     [-i OPTIMIZETRIES] [-m FIXEDMAX2NS] [-p POPLABEL] [-r OUTDIR] [-w] [-y] [-x]

options:
  -h, --help           show this help message and exit
  -a SFSFILENAME       Path for SFS file
  -c FIX_THETA_RATIO   set the fixed value of thetaS/thetaN
  -d DENSITYOF2NS      gamma, lognormal, normal, discrete3,fixed2Ns
  -D                   turn on debug mode, only works if JH_turn_off_options_for_release == True
  -q THETANSPACERANGE  optional setting for the range of thetaNspace, alternatives e.g. 25, 400
  -M MAXI              optional setting for the maximum bin index to include in the calculations
  -o                   fix the mode of 2Ns density at 0, only works for lognormal and gamma
  -z                   include a proportion of the mass at zero in the density model, requires normal, lognormal or gamma
  --profile            Enable profiling
  -e                   for unfolded, include a misspecification parameter
  -f FOLDSTATUS        usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded'
  -g                   turn on global optimzation using basinhopping (quite slow, sometimes improves things)
  -i OPTIMIZETRIES     run the minimize optimizer # times
  -m FIXEDMAX2NS       optional setting for 2Ns maximum, use with -d lognormal or -d gamma, if None 2Ns is estimated
  -p POPLABEL          a population name or other label for the output filename and for the chart
  -r OUTDIR            results directory
  -w                   do not estimate both thetas, just the ratio
  -y                   include a proportion of the mass at some point in the density model, requires normal, lognormal or gamma
  -x                   if true and output file already exists, the run is stopped, else a new numbered output file is made
"""
import numpy as np
from scipy.optimize import minimize,minimize_scalar,OptimizeResult
from scipy.optimize import basinhopping, brentq
from scipy.stats import chi2

import os
import os.path as op
import math
import random
import time
import argparse
import sys
import PRF_Ratios_functions_new
# import PRF_Ratios_functions

from functools import lru_cache

starttime = time.time()

#fix random seeds to make runs with identical settings repeatable 
random.seed(1) 
np.random.seed(2) 

JH_turn_off_options_for_release = True  # use this to turn off obscure options not for the release of this program. 
JH_turn_off_options_for_release = False
# to turn various things off when debugging, set to True by setting  JH_turn_off_options_for_release = False and -D on the command line
miscDebug = False 

def stochastic_round(number):
    floor_number = int(number)
    # Probability of rounding up is the fractional part of the number
    if random.random() < (number - floor_number):
        return floor_number + 1
    else:
        return floor_number

def createdebugglobals():
    global debugoutfilename
    global debugoutfile

def getSFSratios(fn,dofolded,isfolded = False):
    """
        neutral is first SFS in file, then selected 
        then selected
        must be the same length
        if isfolded nc is calculated assuming a diploid sample
    """
    lines = open(fn,'r').readlines()
    sfsindices = [i for i, s in enumerate(lines) if s and s[0].isdigit()]
    numsfss = len(sfsindices)
    sfss = []
    for i in range(numsfss):
        line = lines[sfsindices[i]]
        if "." in line:
            vals = list(map(float,line.strip().split()))
            sfs = list(map(stochastic_round,vals))
        else:
            sfs = list(map(int,line.strip().split()))
        sfs[0] = 0
        sfss.append(sfs)
    if isfolded == False:
        fsfss = []
        nc  = len(sfss[0])
        for sfs in sfss:
            fsfs = [0] + ([sfs[i] + sfs[nc-i] for i in range(1,nc//2)] + [sfs[nc//2]] if nc % 2 == 0 else  [sfs[i] + sfs[nc-i] for i in range(1,1+nc//2)])
            # fsfss.append([0] + [sfs[j]+sfs[nc -j] for j in range(1,nc //2)] + [sfs[nc //2]])
    else:
        fsfss = sfss
        sfss = None
        nc = 2*(len(fsfss[0]) - 1)

    if dofolded:
        if isfolded==False:
            # this folding assumes nc is even 
            neusfs = [0] + [sfss[0][j]+sfss[i][nc -j] for j in range(1,nc //2)] + [sfss[0][nc //2]]
            selsfs = [0] + [sfss[1][j]+sfss[i+1][nc -j] for j in range(1,nc //2)] + [sfss[1][nc //2]]
        else:
            neusfs = fsfss[0]
            selsfs = fsfss[1]
    else:
        neusfs = sfss[0]
        selsfs = sfss[1]
    ratios = [math.inf if  neusfs[j] <= 0.0 else selsfs[j]/neusfs[j] for j in range(len(neusfs))]
    thetaNest = sum(neusfs)/sum([1/i for i in range(1,nc )]) # this should work whether or not the sfs is folded 
    thetaSest = sum(selsfs)/sum([1/i for i in range(1,nc )]) # this should work whether or not the sfs is folded 
    thetaNspace = np.logspace(np.log10(thetaNest/math.sqrt(args.thetaNspacerange)), np.log10(thetaNest*math.sqrt(args.thetaNspacerange)), num=101) # used for likelihood calculation,  logspace integrates bettern than linspace
    return nc,neusfs,selsfs,ratios,thetaNest,thetaSest,thetaNspace

def update_table(X, headers, new_data, new_labels):
    # Update headers and columns for table of observed and expected values 
    # called by buildSFStable()
    headers.extend(new_labels)
    # Format and add new data
    for i in range(len(new_data[0])):
        if len(X) <= i:
            X.append([])
        formatted_row = [f"{float(new_data[0][i]):.1f}", f"{float(new_data[1][i]):.1f}", f"{float(new_data[2][i]):.3f}"]
        X[i].extend(formatted_row)
    return X, headers

def buildSFStable(args,paramdic,pm0tempval,pmmasstempval,pmvaltempval,X,headers,nc):
    """
        generates expected values using estimated parameters
        calls update_table()
    """
    tempthetaratio = paramdic['thetaratio'] if args.use_theta_ratio else None
    tempmisspec = None if args.includemisspec==False else paramdic["misspec"]
    if args.densityof2Ns=="lognormal":
        params = (paramdic["mu"],paramdic["sigma"])
        if args.estimatemax2Ns:   
            neusfs,selsfs,ratios = PRF_Ratios_functions_new.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],paramdic["max2Ns"],nc ,None,args.dofolded,
                tempmisspec,args.densityof2Ns,params,pm0tempval, True, tempthetaratio, pmmass = pmmasstempval,pmval = pmvaltempval)
        else:
            neusfs,selsfs,ratios = PRF_Ratios_functions_new.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],args.fixedmax2Ns,nc ,None,args.dofolded,
                tempmisspec,args.densityof2Ns,params,pm0tempval, True, tempthetaratio, pmmass = pmmasstempval,pmval = pmvaltempval)
    elif args.densityof2Ns=='gamma':
        params = (paramdic["alpha"],paramdic["beta"])
        if args.estimatemax2Ns:
            neusfs,selsfs,ratios = PRF_Ratios_functions_new.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],paramdic["max2Ns"],nc ,None,args.dofolded,
                tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio, pmmass = pmmasstempval,pmval = pmvaltempval)
        else:
            neusfs,selsfs,ratios = PRF_Ratios_functions_new.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],args.fixedmax2Ns,nc ,None,args.dofolded,
                tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio, pmmass = pmmasstempval,pmval = pmvaltempval)
    elif args.densityof2Ns=="normal":
        params = (paramdic["mu"],paramdic["sigma"])
        neusfs,selsfs,ratios = PRF_Ratios_functions_new.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,args.dofolded,
            tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio, pmmass = pmmasstempval,pmval = pmvaltempval)
    elif args.densityof2Ns == "discrete3":
        params = (paramdic["p0"],paramdic["p1"])
        neusfs,selsfs,ratios = PRF_Ratios_functions_new.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,args.dofolded,
            tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio, pmmass = pmmasstempval,pmval = pmvaltempval)                
    else:  #args.densityof2Ns =="fixed2Ns"
        params = (paramdic["2Ns"],)
        neusfs,selsfs,ratios = PRF_Ratios_functions_new.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,args.dofolded,
            tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio, pmmass = pmmasstempval,pmval = pmvaltempval)
    if args.use_theta_ratio:
        X, headers = update_table(X, headers,[neusfs,selsfs,ratios], ["Fit*_N","Fit*_S","Fit_Ratio"])
    else:
        X, headers = update_table(X, headers,[neusfs,selsfs,ratios], ["{}_N","{}_S","{}_Ratio"])
    return X,headers,params


def check_filename_make_new_numbered_as_needed(filename,filecheck):
    """
        if file does not exist return False and the filename
        if a file exists  make a new numbered version
        if a numbered version exists,  make one with a higher number 

    """
    if  op.exists(filename):
        if filecheck:
            return False,filename
        if "." in filename:
            base = filename[:filename.rfind(".")]
            extension = filename[filename.rfind("."):]
        else:
            base = filename
            extension = ""
        if base[-1] == ")" and base[-2] in "0123456789":
            base = base[:base.rfind("(")]
        k = 1
        while op.exists(base + "({})".format(k) + extension): 
            k += 1
        return True,base + "({})".format(k) + extension
    else:
        return True,filename

def makeresultformatstrings(args):    
    """
    stuff for formatting output under various models
    """
    resultlabels =["likelihood"]
    resultformatstrs = ["{}\t{:.3f}"]
    if args.use_theta_ratio:
        if args.fix_theta_ratio is None:
            resultlabels += ["thetaratio"]
            resultformatstrs += ["{}\t{:.4f}\t({:.4f} - {:.4f})"]
    else:
        resultlabels += ["thetaN","thetaS"]
        resultformatstrs += ["{}\t{:.2f}\t({:.2f} - {:.2f})","{}\t{:.2f}\t({:.2f} - {:.2f})"]
    if args.densityof2Ns in ("lognormal","normal"):
        resultlabels += ["mu","sigma"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})","{}\t{:.5g}\t({:.5g} - {:.5g})"]
    elif args.densityof2Ns == "gamma":
        resultlabels += ["alpha","beta"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})","{}\t{:.5g}\t({:.5g} - {:.5g})"]
    elif args.densityof2Ns == "discrete3":
        resultlabels += ["p0","p1"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})","{}\t{:.5g}\t({:.5g} - {:.5g})"]        
    elif args.densityof2Ns == "fixed2Ns":
        resultlabels += ["2Ns"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})"]
    else:
        print("density error")
        exit()
    if args.estimate_pointmass0:
        resultlabels += ["pm0"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})"]
    if args.estimate_pointmass:
        resultlabels += ["pm_mass"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})"]
        resultlabels += ["pm_val"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})"]
    if args.estimatemax2Ns:
        resultlabels += ["max2Ns"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})"]
    if args.includemisspec:
        resultlabels += ["misspec"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})"]
    resultlabels += ["expectation","mode"]
    resultformatstrs += ["{}\t{:.3f}","{}\t{:.3f}"]
    paramlabels = resultlabels[1:-2] # result labels includes likelihood in pos 0 and expectation in pos -1,  the rest are parameter labels 
    return resultlabels,resultformatstrs,paramlabels

def set_bounds_and_start_possibilities(args,thetaNest,thetaSest,ntrials):
    """
        bounds for each model,  with a series of random start values 
        boundary values are based on a lot of hunches, trial and error 
    """
    bounds = []
    startvals = [[] for _ in range(ntrials)]
    if not args.fix_theta_ratio:
        if args.use_theta_ratio:
            thetaratioest = thetaSest/thetaNest
            bounds.append((thetaratioest/20,thetaratioest*20))
            for sv in startvals: sv.append(random.uniform(thetaratioest/3,thetaratioest*3))
        else:
            bounds.append((thetaNest/30,thetaNest*30))
            for sv in startvals: sv.append(random.uniform(thetaNest/10,thetaNest*10))
            bounds.append((thetaSest/30,thetaSest*30))
            for sv in startvals: sv.append(random.uniform(thetaSest/10,thetaSest*10))
    if args.densityof2Ns == "lognormal":
        bounds += [(-5,10),(0.01,5)]
        for sv in startvals: sv.append(random.uniform(0.3,3))
        for sv in startvals: sv.append(random.uniform(0.5,1.5))
    elif args.densityof2Ns =="gamma":
        bounds += [(0.5,40),(0.00001,50)]        
        for sv in startvals: sv.append(random.uniform(1,5))
        for sv in startvals: sv.append(random.uniform(0.1,2))
    elif args.densityof2Ns=="normal":
        # bounds += [(-50,20),(0.1,10)]
        bounds += [(-50,20),(0.1,20)]
        for sv in startvals: sv.append(random.uniform(-15,1))
        for sv in startvals: sv.append(random.uniform(0.2,4))
    elif args.densityof2Ns=="discrete3":
        bounds += [(0.0,1.0),(0.0,1.0)]
        for sv in startvals: sv.append(random.uniform(0.1,0.4))
        for sv in startvals: sv.append(random.uniform(0.1,0.4))
    else:# otherwise density == "fixed2Ns"
        bounds += [(-1000,1000)]
        for sv in startvals: sv.append(random.uniform(-10,1))
    if args.estimate_pointmass0:
        bounds += [(0.0,0.5)]
        # bounds += [(0.0,0.99)]
        for sv in startvals: sv.append(random.uniform(0.01,0.49))
    if args.estimate_pointmass:
        # bounds += [(0.0,0.5),(-10.0, 10.0)]        
        bounds += [(0.0,0.5),(-20.0, 20.0)]        
        for sv in startvals: sv.append(random.uniform(0.01,0.49))
        for sv in startvals: sv.append(random.uniform(-5,1))
    if args.estimatemax2Ns:
        bounds += [(-20.0,20.0)]
        for sv in startvals: sv.append(random.uniform(-5,1))
    if args.includemisspec:
        bounds += [(0.0,0.2)] # assume misspecification rate is less than 0.2 
        for sv in startvals: sv.append(random.uniform(0.001,0.1))
    return bounds,startvals 
    
def buildoutpaths(args):
    """
        makes the directory as needed
        and generates an output filename from the command line options 
    """
    os.makedirs(args.outdir, exist_ok=True)
    outpathstart = op.join(args.outdir,"NONSyn" if "nonsyn" in args.sfsfilename else "Syn")            
    if args.poplabel != "":
        outpathstart += "_{}".format(args.poplabel)
    if args.use_theta_ratio:
        outpathstart += "_Qratio"
    else:
        outpathstart += "_QNQS"
    outpathstart += "_{}".format(args.densityof2Ns)
    if args.fixedmax2Ns is not None:
        outpathstart += "_MX{}".format(args.fixedmax2Ns)
    if args.fix_theta_ratio != None:
        outpathstart += "_FXQR{}".format(args.fix_theta_ratio)
    if args.estimate_pointmass0:
        outpathstart += "_PM0"
    if args.estimate_pointmass:
        outpathstart += "_PM"
    if args.fixmode0:
        outpathstart += "_FXM0"
    if args.includemisspec:
        outpathstart += "_MS"
    outfilename = outpathstart + "_estimates.out"
    plotfn = outpathstart + ".png"    
    fileok, outfilename = check_filename_make_new_numbered_as_needed(outfilename,args.filecheck)
    if not fileok:
        print(outfilename," already exists")
        exit()
    fileok, plotfn = check_filename_make_new_numbered_as_needed(plotfn,args.filecheck)
    if not fileok:
        print(plotfn," already exists")
        exit()        
    return outfilename,plotfn

def countparameters(args):
    numparams = 0
    numparams += 0 if args.fix_theta_ratio else (1 if args.use_theta_ratio else 2) # 0 if theta ratio is fixed, otherwise 1 if using the theta ratio parameters,  else 2 when using thetaN and thetaS 
    numparams += 2 if args.densityof2Ns in ("normal","lognormal","gamma","discrete3") else (1 if args.densityof2Ns=="fixed2Ns" else 0) 
    numparams += 1 if args.estimatemax2Ns else 0 
    numparams += 1 if args.estimate_pointmass0 else 0 
    numparams += 2 if args.estimate_pointmass else 0 
    numparams += 1 if args.includemisspec else 0
    return numparams    

def writeresults(args,numparams,thetaNest,paramlabels,resultlabels,resultformatstrs,resultx,likelihood,confidence_intervals,outfilename,message):
    """
        makes a section 
    """
    paramdic = dict(zip(paramlabels,resultx))
    if args.use_theta_ratio:
        paramdic['thetaN']=thetaNest
        if args.fix_theta_ratio is not None:
            paramdic["thetaratio"] = args.fix_theta_ratio
        paramdic['thetaS'] = paramdic["thetaratio"]*thetaNest
    if args.fixmode0:
        if args.densityof2Ns=="lognormal":
            args.fixedmax2Ns = math.exp(paramdic['mu'] - pow(paramdic['sigma'],2))
        if args.densityof2Ns=="gamma":
            if paramdic['alpha'] < 1:
                args.fixedmax2Ns = 0.0
            else:
                args.fixedmax2Ns =  (paramdic['alpha'] - 1)*paramdic['beta']
    #get expectation
    pm0tempval = pmmasstempval = pmvaltempval = None
    densityof2Nsadjust = None
    if args.densityof2Ns == "lognormal":
        expectation,mode,negsd,densityof2Nsadjust,xvals = PRF_Ratios_functions_new.getXrange(args.densityof2Ns,(paramdic['mu'],paramdic['sigma']),(paramdic['max2Ns'] if args.estimatemax2Ns else args.fixedmax2Ns))
    elif args.densityof2Ns == "gamma" :
        expectation,mode,negsd,densityof2Nsadjust,xvals = PRF_Ratios_functions_new.getXrange(args.densityof2Ns,(paramdic['alpha'],paramdic['beta']),(paramdic['max2Ns'] if args.estimatemax2Ns else args.fixedmax2Ns))
    elif args.densityof2Ns == "discrete3":
        expectation = -(11/2)* (-1 + 92*paramdic['p0'] + paramdic['p1'])
        mode = np.nan
    elif args.densityof2Ns == "normal": 
        expectation,mode,negsd,densityof2Nsadjust,xvals = PRF_Ratios_functions_new.getXrange(args.densityof2Ns,(paramdic['mu'],paramdic['sigma']),(paramdic['max2Ns'] if args.estimatemax2Ns else args.fixedmax2Ns))
    else:
        expectation = mode = np.nan
    if args.estimate_pointmass0:
        expectation *= (1-paramdic['pm0'])
        pm0tempval = paramdic['pm0']
    elif args.estimate_pointmass:
        expectation =  expectation * (1-paramdic['pm_mass']) + paramdic['pm_mass'] * paramdic['pm_val']
        pmmasstempval = paramdic['pm_mass']
        pmvaltempval = paramdic['pm_val']
    AIC = 2*numparams - 2*likelihood
    paramwritestr = ["AIC\t{:.3f}".format(AIC)] 
    paramwritestr+=[resultformatstrs[0].format(resultlabels[0],likelihood)]
    paramwritestr += [resultformatstrs[i+1].format(val,resultx[i],confidence_intervals[i][0],confidence_intervals[i][1]) for i,val in enumerate(resultlabels[1:-2])]
    if expectation == expectation: # this is how you check if it is not nan (i.e. if a is nan  then a != a)
        paramwritestr += [resultformatstrs[-2].format(resultlabels[-2],expectation)]
    if mode == mode: # this is how you check if it is not nan (i.e. if a is nan  then a != a)
        paramwritestr += [resultformatstrs[-1].format(resultlabels[-1],mode)]
    
    outf = open(outfilename, "a")
    outf.write(message)
    if densityof2Nsadjust:
        outf.write("\tNumerical integration of {} distribution check: {:.2g}  (near 1 is good, large departures from 1 suggest a numerical integration difficulty)\n".format(args.densityof2Ns,densityof2Nsadjust))
    outf.write("\n".join(paramwritestr)+"\n")
           
    outf.close()
    return pm0tempval,pmmasstempval,pmvaltempval,paramdic,expectation,mode

def generate_confidence_intervals(func,p_est, arglist, maxLL,bounds,alpha=0.05):
    """
    Generates 95% confidence intervals for each parameter in p_est.
    Find bounds where the likelihood drops by an ammount given by chi-square distribution, 1df 
    Find confidence interval bounds by searching for values that cross the threshold

    Args:
        p_est: List of estimated parameter values.
        f: Function that returns the log-likelihood for a given parameter set.
        maxLL:  the log likelihood at p_est 
        arglist : a tuple or list with all the rest of the arguments,  after the parameter values, to pass to the f 
        alpha: Significance level for confidence intervals (default: 0.05).

    Returns:
        List of tuples, where each tuple contains (lower bound, upper bound) for a parameter.
    """

    confidence_intervals = []
    chiinterval = chi2.ppf(1 - alpha, df=1)/2  
    likelihood_threshold = maxLL - chiinterval
    # checkLL = -func(p_est,*arglist)
    for i in range(len(p_est)):
        # Create a minimization function with fixed parameters except for the i-th one
        def neg_log_likelihood_fixed(p_i):
            p_fixed = p_est.copy()
            p_fixed[i] = p_i
            return -func(p_fixed, *arglist)  # Minimize negative log-likelihood
        if miscDebug:
            confidence_intervals.append((np.nan, np.nan))
            continue
        # widen the bounds of the original search in case CI falls outside the original bounds 
        lbtemp = bounds[i][0]/2
        if lbtemp  <   p_est[i] and  maxLL - neg_log_likelihood_fixed(lbtemp) > chiinterval:
            lower_bound = brentq(lambda p_i: neg_log_likelihood_fixed(p_i) - likelihood_threshold, lbtemp, p_est[i])
        else:
            lower_bound = np.nan # this should still be able to be written to the file
        ubtemp = bounds[i][1]*2
        if ubtemp  >   p_est[i] and  maxLL - neg_log_likelihood_fixed(ubtemp) > chiinterval:
            upper_bound = brentq(lambda p_i: neg_log_likelihood_fixed(p_i) - likelihood_threshold, p_est[i], ubtemp)
        else:
            upper_bound = np.nan # this should still be able to be written to the file

        confidence_intervals.append((lower_bound, upper_bound))
    return confidence_intervals

def calcdistances(X):
    sumsq = 0.0
    c = 0
    for i,row in enumerate(X):
        if i > 0:
            c += 1
            sumsq += pow(float(row[5]) -float(row[2]),2)
    RMSE = math.sqrt(sumsq/c)
    EucDis = math.sqrt(sumsq)
    return EucDis,RMSE

def run(args):
 
    # SET UP ARGS
    isfolded = args.foldstatus == "isfolded" 
    args.dofolded = args.foldstatus == "isfolded"  or args.foldstatus == "foldit"
    temp_estimatemax2Ns = False
    if args.densityof2Ns in ("lognormal","gamma") and args.fixmode0 == False:
        temp_estimatemax2Ns = args.fixedmax2Ns is None
    args.estimatemax2Ns = temp_estimatemax2Ns # True if the max of 2Ns is being estimated 
    args.numparams = countparameters(args)
    args.optimizemethod="Nelder-Mead" # this works better, more consistently than Powell
    
    #GET RATIO DATA 
    nc,neusfs,selsfs,ratios,thetaNest,thetaSest,thetaNspace = getSFSratios(args.sfsfilename,args.dofolded,isfolded=isfolded)

    # START RESULTS FILE
    outfilename,plotfn = buildoutpaths(args)
    if miscDebug:
        debugoutfilename = outfilename + "debug.out"
        debugoutfile = open(debugoutfilename,"w")
    outf = open(outfilename, "w")

    outf.write("PRF-R.py results\n================\n")
    outf.write("Command line: " + args.commandstring + "\n")
    outf.write("Arguments:\n")
    for key, value in vars(args).items():
        outf.write("\t{}: {}\n".format(key,value))
    outf.write("\n")
    outf.close()

    #SET STUFF UP FOR RECORDING RESULTS
    X = [] # Initialize the table 
    headers = [] # Initialize the table headers
    X, headers = update_table(X, headers,[neusfs,selsfs,ratios], ["DataN","DataS","DataRatio"])
    resultlabels,resultformatstrs,paramlabels = makeresultformatstrings(args)    
    # temp = PRF_Ratios_functions_new.NegL_SFSRATIO_estimate_thetaratio([1.3276,1.9209,1.7553,0,-0.061321],160,True,False,"lognormal",False,1,True,False,False,thetaNspace,ratios)
    #SET OPTIMIZATION FUNCTIONS AND TERMS
    boundsarray,startvals = set_bounds_and_start_possibilities(args,thetaNest,thetaSest,args.optimizetries)
    if args.use_theta_ratio:
        func = PRF_Ratios_functions_new.NegL_SFSRATIO_estimate_thetaratio
        arglist = (nc,args.dofolded,args.includemisspec,args.densityof2Ns,args.fix_theta_ratio,args.fixedmax2Ns,args.estimate_pointmass,args.estimate_pointmass0,args.fixmode0,thetaNspace,ratios)
    else:
        func = PRF_Ratios_functions_new.NegL_SFSRATIO_estimate_thetaS_thetaN
        arglist = (nc,args.dofolded,args.includemisspec,args.densityof2Ns,False,args.fixedmax2Ns,args.estimate_pointmass,args.estimate_pointmass0,args.fixmode0,ratios)     
    if miscDebug:
        arglist += (debugoutfile,)
    #RUN MINIMIZE TRIALS
    outf = open(outfilename, "a")
    outf.write("trial\tlikelihood\t" + "\t".join(paramlabels) + "\n")

    for ii in range(args.optimizetries):
        if miscDebug:
            debugoutfile.write("\n\nOptimization trial: {}\n\n".format(ii))
        if ii == 0:
            resvals = []
            rfunvals = []
        startarray = startvals[ii]
        result = minimize(func,np.array(startarray),args=arglist,bounds = boundsarray,method=args.optimizemethod,options={"disp":False,"maxiter":1000*4})    
        # from scipy.optimize import OptimizeResult
        # result = OptimizeResult(x=[1.3276,1.9209,1.7553,0,-0.061321],fun=-1105,message = "debugging")
        
        resvals.append(result)
        rfunvals.append(-result.fun)
        outf.write("{}\t{:.5g}\t{}\n".format(ii,-result.fun," ".join(f"{num:.5g}" for num in result.x)))
    outf.close()

    if args.optimizetries > 0:
        besti = rfunvals.index(max(rfunvals))
        OPTresult = resvals[besti]
        OPTlikelihood = rfunvals[besti]
    else:
        OPTlikelihood = -np.inf
    #basinhopping
    if args.basinhoppingopt:
        if miscDebug:
            debugoutfile.write("\n\nOptimization trial: {}\n\n".format("Basinhopping"))
        if args.optimizetries>0:
            startarray = startvals[besti] # start at the best value found previously,  is this helpful ? works better than an arbitrary start value 
        else:
            startarray = [(boundsarray[i][0] + boundsarray[i][1])/2.0 for i in range(len(boundsarray))]
        try:
            # args.optimizemethod="Powell"
            BHresult = basinhopping(func,np.array(startarray),T=10.0,
                                    minimizer_kwargs={"method":args.optimizemethod,"bounds":boundsarray,"args":tuple(arglist)})
            # BHresult = OptimizeResult(x=[7.5858,10.0,4.2989,0.03835,-0.99699],fun=-1279.792,message = "debugging")
        # print("done",result.x)
            BHlikelihood = -BHresult.fun
            outf = open(outfilename, "a")
            outf.write("BH\t{:.5g}\t{}\n".format(BHlikelihood," ".join(f"{num:.5g}" for num in BHresult.x)))
            outf.close()
        except Exception as e:
            BHlikelihood = -np.inf
            outf = open(outfilename, "a")
            outf.write("\nbasinhopping failed with message : {}\n".format(e))
            outf.close()
    else:
        BHlikelihood = -np.inf
    # print(OPTlikelihood,BHlikelihood)
    if BHlikelihood >= OPTlikelihood:
        result = BHresult
        likelihood = BHlikelihood
        outstring = "Basinhopping"
    elif OPTlikelihood > BHlikelihood:
        result = OPTresult
        likelihood = OPTlikelihood
        outstring = "Optimize"
    else:
        print("problem comparing likelihoods")
        exit()
    confidence_intervals = generate_confidence_intervals(func,list(result.x), arglist, likelihood,boundsarray)
    pm0tempval,pmmasstempval,pmvaltempval,paramdic,expectation,mode = writeresults(args,args.numparams,thetaNest,paramlabels,resultlabels,resultformatstrs,result.x,likelihood,confidence_intervals,outfilename,"\nMaximized Likelihood and Parameter Estimates:\n\tresult.message: {}\n".format(outstring,result.message))
    X,headers, params = buildSFStable(args,paramdic,pm0tempval,pmmasstempval,pmvaltempval,X,headers,nc)
    EucDis, RMSE = calcdistances(X)
    # WRITE a TABLE OF DATA and RATIOS UNDER ESTIMATED MODELS        
    outf = open(outfilename, "a")
    outf.write("\nCompare data and optimization estimates\n\tEuclidean Distance: {:.4f} RMSE: {:.5f}\n".format(EucDis,RMSE))
    if args.use_theta_ratio:
        outf.write("\t*Expected counts generated with Wright-Fisher theta estimates (Poisson-Random-Field-Ratio does not provide theta estimates)\n")
        outf.write("---------------------------------------------------------------------------------------------------------------------------\n") 
    else:
        outf.write("------------------------------------------------------------------------------------------\n") 
    outf.write("i\t" + "\t".join(headers) + "\n")
    for i,row in enumerate(X):
        outf.write("{}\t".format(i) +"\t".join(row) + "\n")            
    #clear caches and write cache usage to
    PRF_Ratios_functions_new.clear_cache(outf)
    # WRITE THE TIME AND CLOSE
    endtime = time.time()
    total_seconds = endtime-starttime
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    outf.write(f"\nTime taken: {hours} hours, {minutes} minutes, {seconds:.2f} seconds\n")            
    outf.close()
    if miscDebug:
        debugoutfile.close()
    print("done ",outfilename)


def parsecommandline():
    global miscDebug
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", dest="sfsfilename",required=True,type = str, help="Path for SFS file")
    parser.add_argument("-c",dest="fix_theta_ratio",default=None,type=float,help="set the fixed value of thetaS/thetaN")
    if JH_turn_off_options_for_release == False:
        parser.add_argument("-d",dest="densityof2Ns",default = "fixed2Ns",type=str,help="gamma, lognormal, normal, discrete3,fixed2Ns")
        parser.add_argument("-D",dest ="debugmode",default = False,action="store_true", help = "turn on debug mode,  only works if JH_turn_off_options_for_release == True")
        parser.add_argument("-q",dest="thetaNspacerange",default=100,type=int,help="optional setting for the range of thetaNspace, alternatives e.g. 25, 400")
        parser.add_argument("-M",dest="maxi",default=None,type=int,help="optional setting for the maximum bin index to include in the calculations")
        parser.add_argument("-o",dest="fixmode0",action="store_true",default=False,help="fix the mode of 2Ns density at 0, only works for lognormal and gamma")    
        parser.add_argument("-z",dest="estimate_pointmass0",action="store_true",default=False,help="include a proportion of the mass at zero in the density model, requires normal, lognormal or gamma")    
        parser.add_argument('--profile', action='store_true', help="Enable profiling")    
    else:
        parser.add_argument("-d",dest="densityof2Ns",default = "fixed2Ns",type=str,help="gamma, lognormal, normal, fixed2Ns")
    parser.add_argument("-e",dest="includemisspec",action="store_true",default=False,help=" for unfolded, include a misspecification parameter") 
    parser.add_argument("-f",dest="foldstatus",required=True,help="usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded' ")    
    parser.add_argument("-g",dest="basinhoppingopt",default=False,action="store_true",help=" turn on global optimzation using basinhopping (quite slow, sometimes improves things)") 
    parser.add_argument("-i",dest="optimizetries",type=int,default=0,help="run the minimize optimizer # times")
    parser.add_argument("-m",dest="fixedmax2Ns",default=None,type=float,help="optional setting for 2Ns maximum, use with -d lognormal or -d gamma, if None 2Ns is estimated ")
    parser.add_argument("-p",dest="poplabel",default = "", type=str, help="a population name or other label for the output filename and for the chart")    
    parser.add_argument("-r",dest="outdir",default = "", type=str, help="results directory")    
    parser.add_argument("-w",dest="use_theta_ratio",action="store_true",default=False,help="do not estimate both thetas, just the ratio") 
    parser.add_argument("-y",dest="estimate_pointmass",action="store_true",default=False,help="include a proportion of the mass at some point in the density model, requires normal, lognormal or gamma") 
    parser.add_argument("-x",dest="filecheck",action="store_true",default=False,help=" if true and output file already exists, the run is stopped, else a new numbered output file is made") 
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])

    if JH_turn_off_options_for_release == True: # add the things not included in args by parser.parse_args() when this is set 
        args.maxi = None
        args.fixmode0 = False
        args.estimate_pointmass0 = False
        args.thetaNspacerange = 100
        args.profile == False
    else:
        if args.debugmode: # use miscDebug because it is global and don't have to pass around args.debugmode 
            miscDebug = True 
            createdebugglobals()
        

    if args.densityof2Ns not in ("lognormal","gamma"):
        if (args.maxi is not None or args.fixedmax2Ns is not None):
            parser.error('cannot use -M or -m with -d normal or fixed 2Ns value')
        if (args.fixmode0):
            parser.error('cannot use -o, fixmode0,  with normal density or fixed 2Ns value')
    if args.densityof2Ns=="discrete3" and (args.fixmode0 or args.estimate_pointmass0):
        parser.error('cannot use -o or -z with discrete3 model')
    if args.densityof2Ns=="discrete3" and JH_turn_off_options_for_release:
        parser.error('cannot use -d discrete3 model')
    if args.fixmode0:
        if args.fixedmax2Ns:
            parser.error(' cannot use -o, fixmode0,  and also fix max2Ns (-m)')
        if args.estimate_pointmass0:
            parser.error(' cannot have point mass at 0 and fixed mode at 0')
    if args.estimate_pointmass0 and args.estimate_pointmass:
        parser.error(' cannot have both pointmass (-y) and pointmass0 (-z)')
    if args.fix_theta_ratio and args.use_theta_ratio == False:
        parser.error(' cannot use -c fix_theta_ratio without -w use_theta_ratio ')
    if args.foldstatus != "unfolded" and args.includemisspec == True:
        parser.error(' cannot include a misspecification term (-e) when SFS is not folded (-f)')
    if args.optimizetries == 0 and args.basinhoppingopt == False:
        parser.error(' either -i must be greater than 0, or -g  or both')
    
    return args


if __name__ == '__main__':
    """

    """
   
    args = parsecommandline()
    ## if --profile
    if args.profile:
        import cProfile
        import pstats
        import io
        from pstats import SortKey

        # Set up the profiler
        profiler = cProfile.Profile()
        profiler.enable()     
    
    run(args)

    ## if --profile
    if args.profile:
        profiler.disable()
        
        # Write full program profile stats
        prffilename = 'PRF_Ratios_stats_{}.prof'.format(args.poplabel)
        with open(prffilename, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats()
        print("profile stats written too {}".format(prffilename))
        
        # Filter and write myptools profile stats
        prffilename = 'PRF_Ratios_functions_stats_{}.prof'.format(args.poplabel)
        with open(prffilename, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats('PRF_Ratios_functions')
        print("PRF_Ratios_functions profile stats written too {}".format(prffilename))

