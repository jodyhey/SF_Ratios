import sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import chi2
sys.path.append("./")
sys.path.append("../")
import SF_Ratios_functions 
import math
import argparse
import warnings
import os.path as op 
from tabulate import tabulate 
warnings.filterwarnings("ignore")
JH_turn_off_options_for_release = True  # use this to turn off obscure options not for the release of this program. 

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

def valstoscreenandfile(label,tabledic,args,writetoscreen=False,AUC = None):
    if writetoscreen:
        print(label)
        print("Command line: {}\n".format(args.commandstring))
        if AUC is not None:
            print("\nAUC: {}\n".format(AUC))
        print(tabulate(tabledic,headers="keys",tablefmt="plain"))
    if args.outfilelabel == None:
        outfilename = "{}_values.txt".format(args.analysis)
    else:
        outfilename = "{}_values_{}.txt".format(args.analysis,args.outfilelabel)
    outfilepath = op.join(args.outdir,outfilename)
    _,outfilepath = check_filename_make_new_numbered_as_needed(outfilepath,True)
    f = open(outfilepath,'w')
    f.write(label + "\n")
    f.write("\nCommand line: {}\n".format(args.commandstring))
    f.write("\nAUC: {}\n".format(AUC))
    f.write(tabulate(tabledic,headers="keys",tablefmt="plain"))
    f.write("\n")
    f.close()

def runROC(args):
    np.random.seed(args.seed)
    dobasicPRF = args.do_standard_PRF # create ROC for basic poisson random field likelihood ratio test with and without selection
    doSFratio = not args.dontdoratio # create ROC for ratio of poisson variables likelihood ratio test with and without selection 
    dofolded = True # use folded distribution
    theta = args.thetaS
    misspec = args.misspec
    includemisspec = True if misspec > 0.0 else False
    n = args.n # sample size 
    if args.n <= 50:
        max2Ns = 10 # larger values result in two few values in the selected SFS
    else:
        max2Ns = 100 
    ntrials = args.ntrials
    foldstring = "folded" if dofolded else "unfolded"
    pvalues = [1,0.999, 0.995, 0.99, 0.9, 0.5, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 
        0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001,0]
    x2thresholds = [0.0,1.5708e-6, 0.0000392704, 0.000157088, 0.0157908, 0.454936, 
            2.70554, 3.84146, 5.41189, 6.6349, 7.87944, 9.54954, 10.8276, 
            12.1157, 13.8311, 15.1367, 16.4481, 18.1893, 19.5114,math.inf]
    x2thresholds.reverse()
    pvalues.reverse()
    if dobasicPRF:
        numg0 = ntrials//2
        
        rocfilename = op.join(args.outdir,'ROC_basicPRF_{}_theta{}_gmax{}_n{}_{}.pdf'.format(args.outfilelabel,theta,-max2Ns,n,foldstring))
        gvals_and_results = [[0.0,0] for i in range(numg0)] # half the g=2Ns values are 0
        for i in range(ntrials-numg0):
            gvals_and_results.append([-np.random.random()*max2Ns,1]) # half the g=2Ns values are nonzero 
        for i in range(ntrials):
            g = gvals_and_results[i][0]
            sfs,sfsfolded = SF_Ratios_functions.simsfs(theta,g,n,misspec,None, False)
            thetastart = theta
            gstart = -1.0
            if dofolded:
                thetagresult = minimize(SF_Ratios_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,includemisspec,sfsfolded),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
                g0result = minimize_scalar(SF_Ratios_functions.NegL_SFS_Theta_Ns,bracket=(thetastart/10,thetastart*10),args = (n,dofolded,includemisspec,sfsfolded),method='Brent')
            else:
                thetagresult = minimize(SF_Ratios_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,includemisspec,sfs),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
                g0result = minimize_scalar(SF_Ratios_functions.NegL_SFS_Theta_Ns,bracket=(thetastart/10,thetastart*10),args = (n,dofolded,includemisspec,sfs),method='Brent')

            thetagdelta = 2*(-thetagresult.fun + g0result.fun)
            # if g < 0:
            #     print(thetagdelta)
            for t in x2thresholds:
                if thetagdelta > t:
                    gvals_and_results[i].append(1)
                else:
                    gvals_and_results[i].append(0)
        TPrate = []
        FPrate = []
        for ti,t in enumerate(x2thresholds):
            tc = 0
            tpc = 0
            fc = 0
            fpc = 0
            for r in gvals_and_results:
                tc += r[1] == 1
                tpc += r[1]==1 and r[2+ti]==1
                fc += r[1] == 0
                fpc += r[1] == 0 and r[2+ti] == 1
            TPrate.append(tpc/tc)
            FPrate.append(fpc/fc)
        AUC = sum([(FPrate[i+1]-FPrate[i])*(TPrate[i+1]+TPrate[i])/2 for i in range(len(TPrate)-1) ])
        plt.plot(FPrate,TPrate)
        # plt.title("Poisson Random Field Likelihood Ratio Test ROC ({} SFS)".format("folded" if dofolded else "unfolded"))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.set_ylim(top=1.0,bottom=0.0)
        # plt.set_xlim(left=0.0,right=1.0)
        plt.text(0.1,0.1,"AUC = {:.3f}".format(AUC),fontsize=14)

        plt.savefig(rocfilename)
        # plt.show()
        plt.clf()
        tabledic = {"chi2":x2thresholds,"prob":pvalues,"TruePosRate":TPrate,"FalsePosRate":FPrate}
        valstoscreenandfile("basic PRF",tabledic,args,AUC=AUC)

    if doSFratio:
        numg0 = ntrials//2
        thetaS = args.thetaS
        thetaN = args.thetaN
        usethetaratio = args.usethetaratio
        if usethetaratio: 
            rocfilename = op.join(args.outdir,'ROC_SFratio_{}_estimate_thetaratio_{}_gmax{}_n{}_{}.pdf'.format(args.outfilelabel,thetaS/thetaN,-max2Ns,n,foldstring))
        else:
            rocfilename = op.join(args.outdir,'ROC_SFratio_{}_estimate_both_thetas_thetaN_{}_thetaS_{}_gmax{}_n{}_{}.pdf'.format(args.outfilelabel,thetaN,thetaS,-max2Ns,n,foldstring))

        gvals_and_results = [[0.0,0] for i in range(numg0)] # half the g=2Ns values are 0
        for i in range(ntrials-numg0):
            gvals_and_results.append([-np.random.random()*max2Ns,1]) # half the g=2Ns values are negative 
        for i in range(ntrials):
            g = gvals_and_results[i][0]
            nsfs,ssfs,ratios = SF_Ratios_functions.simsfsratio(thetaN,thetaS,1.0,n,None,dofolded,misspec,"fixed2Ns",[g],None,False,None)
            thetastart = 100.0
            gstart = -1.0
            if usethetaratio:
                thetaratiostart = thetaS/thetaN
                ratiothetagresult =  minimize(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaratio,
                    np.array([thetaratiostart,gstart]),args=(n,dofolded,includemisspec,"fixed2Ns",False,None,False,False,False,ratios),method="Powell",bounds=[(thetaratiostart/10,thetaratiostart*10),(10*gstart,-10*gstart)])
                ratiothetag0result = minimize_scalar(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaratio,
                    bracket=(thetaratiostart/10,thetaratiostart*10),args = (n,dofolded,includemisspec,"fix2Ns0",False,None,False,False,False,ratios),method='Brent')   
            else:
                thetastart = thetaN
                ratiothetagresult =  minimize(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaS_thetaN,
                    np.array([thetastart,thetastart,gstart]),args=(n,dofolded,includemisspec,"fixed2Ns",False,None,False,False,False,ratios),method="Powell",bounds=[(thetastart/10,thetastart*10),(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
                ratiothetag0result =  minimize(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaS_thetaN,
                    np.array([thetastart,thetastart]),args=(n,dofolded,includemisspec,"fix2Ns0",False,None,False,False,False,ratios),method="Powell",bounds=[(thetastart/10,thetastart*10),(thetastart/10,thetastart*10)])                    
            thetagdelta = 2*(-ratiothetagresult.fun + ratiothetag0result.fun)
            for t in x2thresholds:
                if thetagdelta > t:
                    gvals_and_results[i].append(1)
                else:
                    gvals_and_results[i].append(0)
        TPrate = []
        FPrate = []
        for ti,t in enumerate(x2thresholds):
            tc = 0
            tpc = 0
            fc = 0
            fpc = 0
            for r in gvals_and_results:
                tc += r[1] == 1
                tpc += r[1]==1 and r[2+ti]==1
                fc += r[1] == 0
                fpc += r[1] == 0 and r[2+ti] == 1
            TPrate.append(tpc/tc)
            FPrate.append(fpc/fc)
        AUC = sum([(FPrate[i+1]-FPrate[i])*(TPrate[i+1]+TPrate[i])/2 for i in range(len(TPrate)-1) ])
        plt.plot(FPrate,TPrate)
        # plt.title("Site Frequency Ratio Likelihood Ratio Test ROC ({} SFS)".format("folded" if dofolded else "unfolded"))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.set_ylim(top=1.0,bottom=0.0)
        # plt.set_xlim(left=0.0,right=1.0)
        plt.text(0.1,0.1,"AUC = {:.3f}".format(AUC),fontsize=14)
        plt.savefig(rocfilename)
        # plt.show()
        tabledic = {"chi2":x2thresholds,"prob":pvalues,"TruePosRate":TPrate,"FalsePosRate":FPrate}
        valstoscreenandfile("ratio PRF",tabledic,args,AUC=AUC)    

def runpower(args):
    np.random.seed(args.seed)
    dobasicPRF = args.do_standard_PRF
    doSFratio = not args.dontdoratio
    n=args.n
    dofolded = args.dofolded
    foldstring = "folded" if dofolded else "unfolded"
    misspec = args.misspec
    includemisspec = True if misspec > 0.0 else False
    pvalues = [0.05, 0.01, 0.001]
    x2thresholds = [3.84146,6.6349, 10.8276]
    plabels = ["{:.3f}".format(p) for p in pvalues]    
    gvals = [4,3,2.5,2.25,2,1.75,1.5,1.25,1,0.75,0.5,0.25,0.15,0.1,0.075,0.05,0.025,0.0,-0.025,-0.05,-0.075,-0.1,-0.15,-0.5,-0.5,-0.75,-1,-1.25,-1.5,-1.75,-2,-2.25,-2.5,-3,-4]
    if args.thetaS <= 100:
        gvals = list(np.array(gvals)*5)
    gvals.reverse()

    ntrialsperg = args.ntrials
    if dobasicPRF:
        theta = args.thetaS
        plotfilename = op.join(args.outdir,'POWER_basicPRF_{}_thetaS{}_n{}_{}.pdf'.format(args.outfilelabel,theta,n,foldstring))
        results = [[0]*len(pvalues) for i in range(len(gvals))] #results[i][j] has the count of significant results for gval[i] and pval[j]
        for gj,g in enumerate(gvals):
            for i in range(ntrialsperg):
                sfs,sfsfolded = SF_Ratios_functions.simsfs(theta,g,n,misspec,None, False)
                thetastart = theta 
                gstart = -1.0
                try:
                    thetagresult = minimize(SF_Ratios_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,includemisspec,sfsfolded),method="Nelder-Mead",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
                except Exception as e:
                    print([thetastart,gstart])
                    print([(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
                    exit()
                g0result = minimize_scalar(SF_Ratios_functions.NegL_SFS_Theta_Ns,bracket=(thetastart/10,thetastart*10),args = (n,dofolded,includemisspec,sfsfolded),method='Brent')
                thetagdelta = 2*(-thetagresult.fun + g0result.fun)
                for ti,t in enumerate(x2thresholds):
                    if thetagdelta > t:
                        results[gj][ti] += 1
        for ti,t in enumerate(x2thresholds):
            for gj,g in enumerate(gvals):
                results[gj][ti]/= ntrialsperg
        tabledic = {"2Ns_vals":gvals}
        for ti,t in enumerate(x2thresholds):
            temp = []
            for gj,g in enumerate(gvals):
                temp.append(results[gj][ti])
            plt.plot(gvals,temp,label=plabels[ti])
            tabledic[plabels[ti]] = temp
        plt.xlabel("g (selection)")
        plt.ylabel("Probability of rejecting Null")
        plt.legend(title="$\\alpha$")
        plt.savefig(plotfilename)
        plt.clf()
        valstoscreenandfile("Power_basicPRF",tabledic,args,writetoscreen=False)
        # plt.show()
    if doSFratio:
        thetaS = args.thetaS
        thetaN = args.thetaN
        thetaNspacerange=100
        thetaNspace = np.logspace(np.log10(thetaN/math.sqrt(thetaNspacerange)), np.log10(thetaN*math.sqrt(thetaNspacerange)), num=101) 
        usethetaratio = args.usethetaratio
        if usethetaratio:        
            plotfilename = op.join(args.outdir,'POWER_SFratio_{}_estimate_thetaratio_{}_n{}_{}.pdf'.format(args.outfilelabel,thetaS/thetaN,n,foldstring))
        else:
            plotfilename = op.join(args.outdir,'POWER_SFratio_{}_estimate_thetaN{}_thetaS_n{}_{}.pdf'.format(args.outfilelabel,thetaN,thetaS,n,foldstring))
        # print(plotfilename)
        # exit()
        results = [[0]*len(pvalues) for i in range(len(gvals))] #results[i][j] has the count of significant results for gval[i] and pval[j]
        for gj,g in enumerate(gvals):
            for i in range(ntrialsperg):
                nsfsfolded,ssfsfolded,ratios = SF_Ratios_functions.simsfsratio(thetaN,thetaS,1.0,n, None, dofolded,misspec, "fixed2Ns",[g],None, False,None)
                thetastart = thetaS #100.0
                gstart = -1.0
                if usethetaratio:
                    thetaratiostart = thetaS/thetaN
                    ratiothetagresult =  minimize(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaratio,
                        np.array([thetaratiostart,gstart]),args=(n,dofolded,includemisspec,"fixed2Ns",False,None,False,False,False,thetaNspace,ratios),method="Powell",bounds=[(thetaratiostart/10,thetaratiostart*10),(10*gstart,-10*gstart)])
                    ratiothetag0result = minimize_scalar(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaratio,
                        bracket=(thetaratiostart/10,thetaratiostart*10),args = (n,dofolded,includemisspec,"fix2Ns0",False,None,False,False,False,thetaNspace,ratios),method='Brent')   
                else:
                    ratiothetagresult =  minimize(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaS_thetaN,
                        np.array([thetastart,thetastart,gstart]),args=(n,dofolded,includemisspec,"fixed2Ns",False,None,False,False,False,ratios),method="Powell",bounds=[(thetastart/10,thetastart*10),(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
                    ratiothetag0result =  minimize(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaS_thetaN,
                        np.array([thetastart,thetastart]),args=(n,dofolded,includemisspec,"fix2Ns0",False,None,False,False,False,ratios),method="Powell",bounds=[(thetastart/10,thetastart*10),(thetastart/10,thetastart*10)])                                        
                thetagdelta = 2*(-ratiothetagresult.fun + ratiothetag0result.fun)
                for ti,t in enumerate(x2thresholds):
                    if thetagdelta > t:
                        results[gj][ti] += 1
        for ti,t in enumerate(x2thresholds):
            for gj,g in enumerate(gvals):
                results[gj][ti]/= ntrialsperg
        tabledic = {"2Ns_vals":gvals}
        for ti,t in enumerate(x2thresholds):
            temp = []
            for gj,g in enumerate(gvals):
                temp.append(results[gj][ti])
            plt.plot(gvals,temp,label=plabels[ti])
            tabledic["P="+plabels[ti]] = temp
        plt.xlabel("g (selection)")
        plt.ylabel("Probability of rejecting Null")
        plt.legend(title="$\\alpha$")
        plt.savefig(plotfilename)
        # plt.show()
        valstoscreenandfile("Power_SFratio",tabledic,args,writetoscreen=False)

def runchi2(args):
    dobasicPRF = args.do_standard_PRF
    doSFratio = not args.dontdoratio
    misspec = args.misspec
    includemisspec = True if misspec > 0.0 else False
    n=args.n
    numdf = 1 # all models are compared differ by one parameter 
    np.random.seed(args.seed)
    ntrials = args.ntrials
    g = 0.0 # fixed at 0,  we assume the null model is true 
    dofolded = args.dofolded #True # False

    foldstring = "folded" if dofolded else "unfolded"
    if dobasicPRF: # regular PRF LLR stuff
        theta = args.thetaS
        figfn = op.join(args.outdir,"CHISQcheck_basicPRF_{}_thetaS{}_g{}_n{}_{}.pdf".format(args.outfilelabel,theta,g,n,foldstring))
        thetagcumobsv = []
        numnegdeltas = 0

        #check chi^2 approximation of LLR for basic FW PRF 
        for i in range(ntrials):
            sfs,sfsfolded = SF_Ratios_functions.simsfs(theta,g,n,misspec,None,False)
            sfs = sfsfolded if dofolded else sfs
            thetastart = theta
            gstart = -1.0
            thetagresult = minimize(SF_Ratios_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,includemisspec,sfs),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
            g0result = minimize_scalar(SF_Ratios_functions.NegL_SFS_Theta_Ns,bracket=(thetastart/10,thetastart*10),args = (n,dofolded,includemisspec,sfs),method='Brent')

            thetagdelta = -thetagresult.fun + g0result.fun
            if thetagdelta < 0:
                thetagdelta = 0.0
                numnegdeltas +=1        
            thetagcumobsv.append(2*thetagdelta)
        thetagcumobsv.sort()
        thetagcumobsv = np.array(thetagcumobsv)
        cprob = np.array([i/len(thetagcumobsv) for i in range(1,len(thetagcumobsv)+1)])
        x = np.linspace(0, max(thetagcumobsv), 200)
        # plt.plot(thetagcumobsv,cprob)
        plt.plot(thetagcumobsv,cprob, label=r'Observed: cumulative distribution of $-2\Delta$')
        plt.ylabel('Cumulative Probability', fontsize=14)
        # Setting the X-axis label with Greek letters and superscript
        plt.xlabel(r'$-2\Delta $, $\chi^2 \, 1\mathrm{df}$', fontsize=14)
        plt.plot(x, chi2.cdf(x, df=numdf), label=r'Expected: $\chi^2$ 1df')
        plt.legend(fontsize=14)
        plt.savefig(figfn)
        # plt.show()
        plt.clf()
        tabledic = {"CumProb":cprob,"2Delta":thetagcumobsv,"chi^2":np.linspace(min(thetagcumobsv), max(thetagcumobsv), len(thetagcumobsv)),"chi^2_Prob":[chi2.cdf(x,df=numdf) for x in np.linspace(0, max(thetagcumobsv), len(thetagcumobsv))]}
        valstoscreenandfile("CHISQcheck_basicPRF",tabledic,args,writetoscreen=False)



    if doSFratio: #check chi^2 approximation of LLR for ratio of PRF, uses a single theta
        thetaS = args.thetaS
        thetaN = args.thetaN
        usethetaratio = args.usethetaratio
        if usethetaratio:
            figfn = op.join(args.outdir,"CHISQcheck_RATIO_{}_estimate_thetaratio_{}_g{}_n{}_{}.pdf".format(args.outfilelabel,thetaS/thetaN,g,n,foldstring))
        else:
            figfn = op.join(args.outdir,"CHISQcheck_RATIO_{}_estimate_thetaN_{}_thetaS_{}_g{}_n{}_{}.pdf".format(args.outfilelabel,thetaN,thetaS,g,n,foldstring))
        ratiothetacumobsv = []
        numnegdeltas = 0
        for i in range(ntrials):
            nsfs,ssfs,ratios = SF_Ratios_functions.simsfsratio(thetaN,thetaS,1.0,n,None,dofolded,misspec,"fixed2Ns",[g],None,False,None)
            gstart = -1.0
            if usethetaratio:
                thetaratiostart = thetaS/thetaN
                ratiothetagresult =  minimize(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaratio,
                    np.array([thetaratiostart,gstart]),args=(n,dofolded,includemisspec,"fixed2Ns",False,None,False,False,False,ratios),method="Powell",bounds=[(thetaratiostart/10,thetaratiostart*10),(10*gstart,-10*gstart)])
                ratiothetag0result = minimize_scalar(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaratio,
                    bracket=(thetaratiostart/10,thetaratiostart*10),args = (n,dofolded,includemisspec,"fix2Ns0",False,None,False,False,False,ratios),method='Brent')   
            else:
                ratiothetagresult =  minimize(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaS_thetaN,
                    np.array([thetastart,thetastart,gstart]),args=(n,dofolded,includemisspec,"fixed2Ns",False,None,False,False,False,ratios),method="Powell",bounds=[(thetastart/10,thetastart*10),(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
                ratiothetag0result =  minimize(SF_Ratios_functions.NegL_SFSRATIO_estimate_thetaS_thetaN,
                    np.array([thetastart,thetastart]),args=(n,dofolded,includemisspec,"fix2Ns0",False,None,False,False,False,ratios),method="Powell",bounds=[(thetastart/10,thetastart*10),(thetastart/10,thetastart*10)])                    
            thetagdelta = -ratiothetagresult.fun + ratiothetag0result.fun
            if thetagdelta < 0:
                thetagdelta = 0.0
                numnegdeltas +=1
            ratiothetacumobsv.append(2*thetagdelta)
        ratiothetacumobsv.sort()
        ratiothetacumobsv = np.array(ratiothetacumobsv)
        cprob = np.array([i/len(ratiothetacumobsv) for i in range(1,len(ratiothetacumobsv)+1)])
        x = np.linspace(0, max(ratiothetacumobsv), 200)
        plt.plot(ratiothetacumobsv,cprob,label=r'Observed: cumulative distribution of $-2\Delta$')
        plt.ylabel('Cumulative Probability',fontsize=14)
        # Setting the X-axis label with Greek letters and superscript
        plt.xlabel(r'$-2\Delta $, $\chi^2 \, 1\mathrm{df}$', fontsize=14)
        plt.plot(x, chi2.cdf(x, df=numdf), label=r'Expected: $\chi^2$ 1df')
        plt.legend(fontsize=14)
        plt.savefig(figfn)
        # print(numnegdeltas)
        # plt.show()
        tabledic = {"CumProb":cprob,"2Delta":ratiothetacumobsv,"chi^2":np.linspace(min(ratiothetacumobsv), max(ratiothetacumobsv), len(ratiothetacumobsv)),"chi^2_Prob":[chi2.cdf(x,df=numdf) for x in np.linspace(0, max(ratiothetacumobsv), len(ratiothetacumobsv))]}
        valstoscreenandfile("CHISQcheck_SFratio",tabledic,args,writetoscreen=False)


def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b",dest="do_standard_PRF",action="store_true",default = False,help = "do basic PRF, not SF PRF")   
    parser.add_argument("-d",dest="outdir",type = str, default = "./Results_WrightFisher_PRF_simulations",help=" ouput directory, optional") 
    parser.add_argument("-e",dest="misspec",type=float,default = 0.0,help = " rooting misspecification rate for unfolded distribution") 
    parser.add_argument("-f",dest="dofolded",action="store_true",default = False,help = "do folded")    
    parser.add_argument("-k",dest="ntrials",type = int, default = 1000,help="# of trials")
    parser.add_argument("-n",dest="n",type = int, default = 50,help="# of sampled chromosomes  i.e. 2*(# diploid individuals)  ")
    parser.add_argument("-o",dest="outfilelabel",type = str, default = None,help="label to go in output text file name")
    parser.add_argument("-p",dest="thetaS",type=float,default = 100,help = "thetaS ")    
    parser.add_argument("-q",dest="thetaN",type=float,help = "thetaN ")        
    parser.add_argument("-r",dest="dontdoratio",action="store_true",default = False,help = "do NOT do ratio PRF, default is to do it ")    
    parser.add_argument("-s",dest="seed",type = int,default = 1,help = "random number seed")  
    if JH_turn_off_options_for_release == False:
        parser.add_argument("-w",dest="usethetaratio",action="store_true",default = False,help = "use theta ratio rather than both thetaS and thetaN")    
    parser.add_argument("-y",dest="analysis",type = str,required=True, help="ROC, power, or chi2")
    
        
    args  =  parser.parse_args(sys.argv[1:])   
    args.commandstring = " ".join(sys.argv[1:])
    if args.thetaN is None:
        args.thetaN = args.thetaS
    if args.misspec != 0.0 and args.dofolded != False:
        parser.error ("-e conflicts with -f ")       
    args.analysis = args.analysis.upper()  
    if JH_turn_off_options_for_release:   
        args.usethetaratio = True 
    return args

    # return parser

if __name__ == '__main__':
    """

    """
    args = parsecommandline()
    if args.analysis == "ROC":
        runROC(args)
    elif args.analysis == "POWER":
        runpower(args)
    elif args.analysis == "CHI2":
        runchi2(args)
    else:
        print("invalid analysis option (valid options are ROC, power, and chi2): {}".format(args.analysis))
        exit()
