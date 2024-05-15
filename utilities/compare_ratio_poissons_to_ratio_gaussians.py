"""
simulate distribution of ratio Z=X/Y  where X and Y are poisson rvs

Also plot function for the density of the ratio of two normals
use expressions in Díaz-Francés, E. and F. J. Rubio (2013). "On the existence of a normal approximation to the distribution of the ratio of two independent normal random variables." Statistical Papers 54: 309-323.
to approximate the density	

usage: compare_ratio_poissons_to_ratio_gaussians.py [-h] [-f PLOTFILENAME] [-n N] -x UX -y UY

options:
  -h, --help       show this help message and exit
  -f PLOTFILENAME  plot file name, default = compare_distributions_poisson_gaussian_ratios.pdf
  -n N             number of simulated ratios, default = 100000
  -x UX            mean of numerator term
  -y UY            mean of denominator term


"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import sys
import argparse

def f(z,beta,rho,deltay):
    rho2 = rho*rho
    z2 = z*z
    q = (1+beta*rho2*z)/(deltay*pow(1+rho2*z2,1/2))
    erfq = math.erf(q/pow(2,1/2))
    temp1 = rho/(math.pi * (1+rho2*z2))
    temp2 = math.exp(-(rho2*beta*beta + 1)/(2*deltay*deltay))
    if temp2 > 0:
        temp3 = 1+ pow(math.pi/2,1/2)*q*erfq*math.exp(q*q/2)
        return temp1*temp2*temp3
    else:
        return 0

def run(args):
    ux = args.ux
    uy = args.uy
    sigmax = pow(ux,1/2)
    sigmay = pow(uy,1/2)
    deltay = 1/sigmay
    beta = ux/uy
    rho = sigmay/sigmax
    sigmax = pow(ux,1/2)
    n = args.n
    xvals = np.array([np.random.poisson(ux) for i in range(n)])
    yvals = np.array([np.random.poisson(uy) for i in range(n)])
    zvals = np.array([xvals[i]/yvals[i] for i in range(n) if yvals[i] > 0])
    zspace = np.linspace(zvals.min(), zvals.max(), 200)
    fzspace = np.array([f(z,beta,rho,deltay) for z in zspace])

    # print("mean",zvals.mean(),"var",zvals.var())
    # Plot the histogram.
    plt.hist(zvals, bins=200, density=True, alpha=0.6, color='g')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    plt.plot(zspace, fzspace, color='red')
    title = "Fit results: ux = %.1f,  uy = %.1f" % (ux,uy)
    plt.title(title)
    plt.savefig(args.plotfilename)
    # plt.show()


def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",dest="plotfilename",default = "compare_distributions_poisson_gaussian_ratios.pdf",help="plot file name, default = compare_distributions_poisson_gaussian_ratios.pdf")    
    parser.add_argument("-n",dest="n",type = int, default = 100000,help="number of simulated ratios, default = 100000")
    parser.add_argument("-x",dest="ux",type=float,required=True,help = "mean of numerator term")
    parser.add_argument("-y",dest="uy",type=float,required=True,help = "mean of denominator term")
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    return args

    # return parser

if __name__ == '__main__':
    """

    """
    args = parsecommandline()
    run(args)