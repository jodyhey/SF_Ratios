"""
    poisson random field SFS work 
    a module of various functions

    models:
        fix2Ns0,fixed2Ns,normal,lognormal,gamma,discrete3 

    sfs lists:
        all sfs lists begin with 0 in position 0 
        there is no position for a count where all chromosomes have the allele (i.e. fixed in the sample)


    counting with k chromosomes
        with unfolded, there are k - 1 values  so n_unf = k-1
            this means an unfolded list has length 1 + k - 1 == k 
        with folded it is more complicated, 
            if k is even,  a count of k//2  has no folding partner. e.g. if k is 4  then when folding the bins for counts 1, 3 are summed
                but the bin for 2 is not added to anything.  
                so n_f  has k//2 values  i.e. n_f = k//2  in a list that has length 1 + n_f
            if k is odd, a count of k//2 does have a folding partner, e.g. if k is 5 then bins 1,3 are summed, as are 2,4 
                so n_f has k//2 values,  i.e. n_f = k//2   in a list that has length 1 + n_f
            so the folded value of n_f for even k is the same as for odd count of k + 1 
    nc : # of chromosomes
    n_unf : nc - 1
    n_f : nc // 2 

        
"""
import numpy as np
import  mpmath 
import math
from scipy.optimize import minimize, brentq
import scipy
from scipy import integrate
from scipy.optimize import golden
from scipy.special import erf, gamma, gammainc,gammaincc
import warnings

warnings.simplefilter("error", RuntimeWarning) # make RuntimeWarnings errors so we can trap them

#for the Jacobian when using BFGS optimization in SFS_estimator_bias_variance.py
# from here  https://stackoverflow.com/questions/41137092/jacobian-and-hessian-inputs-in-scipy-optimize-minimize
# ended up mostly, not alwyas, crashing with nan errors 
# from numdifftools import Jacobian
# def NegL_SFSRATIO_Theta_Nsdensity_der(x, max2Ns,nc ,dofolded,densityof2Ns,zvals):
#     return Jacobian(lambda x: NegL_SFSRATIO_Theta_Nsdensity(x,max2Ns,nc ,dofolded,densityof2Ns,zvals))(x).ravel()
   

#constants

sqrt2 =pow(2,1/2)
sqrt_2_pi = np.sqrt(2 * np.pi)
sqrt_pi_div_2 = np.sqrt(np.pi/2)

discrete3_xvals = np.concatenate([np.linspace(-1000,-2,20),np.linspace(-1,1,20),np.linspace(1.1,10,20)]) # only for discrete3 distribution 

lowerbound_2Ns_integration = -100000 # exclude regions below this from integrations,  arbitrary but saves some time
minimum_2Ns_location = -10000 # the lowest value for the mode of a continuous distribution that will be considered,  lower than this an the likelihood just returns inf 

# thetaNspace = np.logspace(2,6,num=20) # used for integrating over thetaN, when calculating the probability of the ratio  100 <= thetaN <= 10000
thetaNspace = np.logspace(2,6,num=30) # used for integrating over thetaN, when calculating the probability of the ratio  100 <= thetaN <= 1000000

def coth(x):
    """
        save a bit of time for large values of x 
    """
    if abs(x) > 15: # save a bit of time 
        return -1.0 if x < 0 else 1.0
    else:
        return np.cosh(x)/np.sinh(x) 

def logprobratio(alpha,beta,z):
    """
        returns the log of the probability of a ratio z of two normal densities when for each normal density the variance equals the mean 
        is called from other functions,  where alpha, beta and the ratio z have been calculated for a particular frequency bin

        two versions of this function
       
        Díaz-Francés, E. and F. J. Rubio paper expression (1)

        Kuethe DO, Caprihan A, Gach HM, Lowe IJ, Fukushima E. 2000. Imaging obstructed ventilation with NMR using inert fluorinated gases. Journal of applied physiology 88:2279-2286.
        the gamma in the paper goes away because it is equal to 1/alpha^(1/2) when we assume the normal distributions have mean equal to variance 

        The functions are similar, but the Díaz-Francés, E. and F. J. Rubio  works much better overall,  e.g. LLRtest check and ROC curves are much better 
        However the Díaz-Francés and Rubio function gives -inf often enough to cause problems for the optimizer,  so we set final probability p = max(p,1e-50)

    """

    try:
        delta = beta
        beta = alpha
        z2 = z*z
        delta2 = delta*delta
        z1 = 1+z
        z2b1 = 1+z2/beta
        z2boverb = (z2+beta)/beta
        betasqroot = math.sqrt(beta)
        ratiotemp = -(1+beta)/(2*delta2)
        temp1 = mpmath.fdiv(mpmath.exp(ratiotemp),(math.pi*z2b1*betasqroot))
        try:
            ratiotemp2 =   (-pow(z-beta,2)/(2*delta2*(z2+beta)))
        except:
            ratiotemp2 = mpmath.fdiv(mpmath.fneg(mpmath.power(-z-beta,2)),(2*delta2*(z2+beta)))
        temp2num = mpmath.fmul(mpmath.fmul(mpmath.exp(ratiotemp2),z1), mpmath.erf(z1/(sqrt2 * delta * math.sqrt(z2boverb))))
        temp2denom = sqrt_2_pi *  betasqroot * delta*pow(z2boverb,1.5)
        temp2 = mpmath.fdiv(temp2num,temp2denom )
        p = mpmath.fadd(temp1,temp2)
        
        if p > 1e-307:
            logp = math.log(p)
        else:
            logp = float(mpmath.log(p))
        return logp 
    
    except Exception as e:
        print("Caught an exception in logprobratio: {}  p {}".format(e,p))  
        exit()

def intdeltalogprobratio(alpha,z,deltavals):
    """
        integrates over the delta term in the probability of a ratio of two normal distributions
    """
    def trapezoidal_integration(x_values, y_values): # not in use, use np.trapz() instead
        """Performs trapezoidal integration over lists of floats and function values."""
        integral = 0
        for i in range(1, len(x_values)):
            dx = x_values[i] - x_values[i - 1]
            y_avg = (y_values[i] + y_values[i - 1]) / 2
            integral += dx * y_avg
        return math.log(integral)
    
    def rprobdelta(z,beta,delta):

        delta2 = delta*delta
        try:
            ratiotemp = -(1+beta)/(2*delta2)
            temp1 = mpmath.fdiv(mpmath.exp(ratiotemp),(math.pi*z2b1*betasqroot))
            try:
                ratiotemp2 =   (-pow(z-beta,2)/(2*delta2*(z2+beta)))
            except:
                ratiotemp2 = mpmath.fdiv(mpmath.fneg(mpmath.power(-z-beta,2)),(2*delta2*(z2+beta)))
            temp2num = mpmath.fmul(mpmath.fmul(mpmath.exp(ratiotemp2),z1), mpmath.erf(z1/(sqrt2 * delta * math.sqrt(z2boverb))))
            try:
                temp2denom = sqrt_2_pi *  betasqroot * delta*pow(z2boverb,1.5)
            except RuntimeWarning:
                temp2denom = mpmath.fmul(sqrt_2_pi,mpmath.fmul(betasqroot,mpmath.fmul(delta,mpmath.power(z2boverb,1.5))))
            temp2 = mpmath.fdiv(temp2num,temp2denom )
            p = float(mpmath.fadd(temp1,temp2))
            # print("{:.1f} {}".format(z,float(p)))
            # p = mpmath.fadd(temp1,temp2)
            return p
        except Exception as e:
            print(f"Caught an exception in rprobdelta: {e}")  
            exit()

    beta = alpha
    z2 = z*z
    z1 = 1+z
    z2b1 = 1+z2/beta
    z2boverb = (z2+beta)/beta
    betasqroot = math.sqrt(beta)
    rprob_density_values = np.array([rprobdelta(z,beta,delta) for delta in deltavals])
    rprob=np.trapz(np.flip(rprob_density_values),np.flip(deltavals)) # have to flip as trapz expects increasing values 
    if rprob > 1e-307:
        logrprob = math.log(rprob)
    else:
        logrprob = float(mpmath.log(rprob))
    return logrprob

def ratio_expectation(p,i,max2Ns,nc,dofolded,misspec,densityof2Ns):
    """
    get the expected ratio for bin i given a set of parameter values 
    """
    
    def ztimesprobratio(z,alpha,beta,doneg):
        """
            called by ratio_expectation()
            for getting the expected ratio
            doneg is True when called by golden()
            False when doing the integration
        """

        # Díaz-Francés, E. and F. J. Rubio 
        try:
            delta = beta
            beta = alpha
            z2 = z*z
            delta2 = delta*delta
            z1 = 1+z
            z2b1 = 1+z2/beta
            z2boverb = (z2+beta)/beta
            betasqroot = math.sqrt(beta)
            ratiotemp = -(1+beta)/(2*delta2)
            temp1 = mpmath.fdiv(mpmath.exp(ratiotemp),(math.pi*z2b1*betasqroot))
            ratiotemp2 =   (-pow(z-beta,2)/(2*delta2*(z2+beta)))
            temp2num = mpmath.fmul(mpmath.fmul(mpmath.exp(ratiotemp2),z1), mpmath.erf(z1/(sqrt2 * delta * math.sqrt(z2boverb))))
            # temp2denom = sqrt_2_pi *  betasqroot * delta*pow(z2boverb,1.5)
            try:
                temp2denom = sqrt_2_pi *  betasqroot * delta*pow(z2boverb,1.5)
            except RuntimeWarning:
                temp2denom = mpmath.fmul(sqrt_2_pi,mpmath.fmul(betasqroot,mpmath.fmul(delta,mpmath.power(z2boverb,1.5))))
            temp2 = mpmath.fdiv(temp2num,temp2denom )
            p = float(mpmath.fadd(temp1,temp2))
        except Exception as e:
            print(f"Caught an exception in ztimesprobratio: {e}")  
        if p < 0.0:
            return 0.0
        if doneg:
            return -p*z
        return  p*z    

    foldxterm = dofolded and i < nc //2 # True if summing two bins, False if not, assumes nc is even, in which case the last bin is not folded 
    thetaN = p[0]
    thetaS = p[1]
    g = (p[2],p[3])
    ex,mode,sd,densityadjust,g_xvals = getXrange(densityof2Ns,g,max2Ns)
    intval = trapzint2Ns(densityof2Ns,max2Ns,g,nc,i,foldxterm,misspec,g_xvals,densityadjust)
    ux = thetaS*intval 
    uy = thetaN*nc /(i*(nc -i)) if foldxterm else thetaN/i    
    alpha = ux/uy
    sigmay = math.sqrt(uy)
    beta = 1/sigmay
    peak = golden(ztimesprobratio,args=(alpha,beta,True), brack = (0,1000)) # use ratio interval of 0 1000 to start 
    x = integrate.quad(ztimesprobratio,-10,peak*10,args=(alpha,beta, False)) # use interval of 0 to 10* the peak location 
    return x[0]

def prf_selection_weight(nc,i,g,dofolded,misspec):
    """
        Poisson random field selection weight for g=2Ns for bin i  (folded or unfolded)
        this is the function you get when you integrate the product of two terms:
             (1) WF term for selection    (1 - E^(-2 2 N s(1 - q)))/((1 - E^(-2 2 N s)) q(1 - q))  
             (2) bionomial sampling formula for i copies,  given allele frequency q 
        over the range of allele frequencies 

        the hyp1f1 function can easily fail.
        It is always when the function is trying to return a positive value near 0
        mpmath does not help as they values are effectively 0
        So for dofolded, so just set them to 0
        use mpmath when dofolded = False,  as there is only one hpy1f1 call
    """
    minselectweight = 1e-307 # small float,  use in place of 0's when they come up

    if abs(g)<1e-3: # neutral or close enough 
        if dofolded:
            us = nc/(i*(nc -i))
        else:
            if misspec:
                us = (1-misspec)/i + misspec/(nc-i)
            else:
                us = 1/i
        return us
    tempc = coth(g)
    if tempc==1.0: # happens if g is high and positive 
        if dofolded:
            us = 2*(nc /(i*(nc -i)))
        else:
            us = (nc /(i*(nc -i))) # same whether or not misspec is included 
        return us
    if dofolded:
        
        try:
            temph1 = scipy.special.hyp1f1(i,nc ,2*g)
        except: # Exception as e:
            temph1 = float( mpmath.hyp1f1(i,nc,2*g))
        try: 
            temph2 = scipy.special.hyp1f1(nc - i,nc ,2*g)
        except: # Exception as e:
            temph2 = float(mpmath.hyp1f1(nc-i,nc,2*g))
        temph = temph1+temph2
        us = (nc /(2*i*(nc -i)))*(2 +2*tempc - (tempc-1)*temph)
        #check some things 
        # if math.isnan(temph) or math.isnan(temph1) or math.isnan(temph2) or  math.isnan(us) or math.isnan(tempc):
        #     print("NAN",i,nc,g,temph, temph1,temph2,us,tempc)
        #     exit()
        # if us == 0.0:
        #     use = minselectweight
            # print("us<0 ",i,nc,g,temph, temph1,temph2,us,tempc)
            # exit()      
        return us      
    else: # unfolded
        if misspec in (None,False,0.0):
            try:
                temph = scipy.special.hyp1f1(i,nc ,2*g)
            except:
                temph = float(mpmath.hyp1f1(i,nc ,2*g))
            us = (nc /(2*i*(nc -i)))*(1 + tempc - (tempc-1)* temph)       
            #check some things 
            # if math.isnan(temph) or math.isnan(us) or math.isnan(tempc):
            #     print("NAN",i,nc,g,temph,us,tempc)
            #     exit()
            # if us == 0.0:
            #     us = minselectweight
                # print("us=0 ",i,nc,g,temph,us,tempc)
                # exit()
            return us
        else: #include misspecification term in unfolded 
            try:
                temph1 = scipy.special.hyp1f1(i,nc ,2*g)
            except: # Exception as e:
                temph1 = float( mpmath.hyp1f1(i,nc,2*g))
            try: 
                temph2 = scipy.special.hyp1f1(nc - i,nc ,2*g)
            except: # Exception as e:
                temph2 = float(mpmath.hyp1f1(nc-i,nc,2*g))
            temph = (1-misspec)*temph1 + misspec * temph2 
            us = (1+tempc + (1-tempc)*temph)  * (nc /(2*i*(nc -i)))          
            #check some things 
            # if math.isnan(temph) or math.isnan(temph1) or math.isnan(temph2) or  math.isnan(us) or math.isnan(tempc):
            #     print("misspec, NAN",i,nc,g,temph, temph1,temph2,us,tempc)
            #     exit()
            # if us == 0.0:
            #     us = minselectweight
                # print("misspec, us<0 ",i,nc,g,temph, temph1,temph2,us,tempc)
                # exit() 
            return us

def getXrange(densityof2Ns,g,max2Ns):
    """
        get the range of integration by going +4 and -4 standard deviations away from the mode 
    """
    numSDs = 5 
    def prfdensity(xval,g):
        if densityof2Ns=="lognormal":   
            mean = g[0]
            std_dev = g[1]
            x = float(max2Ns-xval)
            p = (1 / (x * std_dev * sqrt_2_pi)) * np.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2))
            if p==0.0:
                p= float(mpmath.fmul(mpmath.fdiv(1, (x * std_dev * sqrt_2_pi)), mpmath.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2))))
        elif densityof2Ns=="gamma":
            alpha = g[0]
            beta = g[1]
            x = float(max2Ns-xval)
            p = ((x**(alpha-1))*np.exp(-x / beta))/((beta**alpha)*math.gamma(alpha))
        elif densityof2Ns == "normal":
            mean = g[0]
            std_dev = g[1]
            p = (1 / (std_dev * sqrt_2_pi)) * math.exp(-(1/2)*((xval- mean)/std_dev)**2)
        return p
    
    listofarrays = []
    if densityof2Ns=="normal":
        mode = ex = g[0]
        sd = g[1]
    elif densityof2Ns == "lognormal":
        stdev_squared = g[1]*g[1]
        sd = math.sqrt((math.exp(stdev_squared) - 1) * math.exp(2*g[0] + stdev_squared))
        ex =  -math.exp(g[0] + stdev_squared/2)
        mode = -math.exp(g[0] - stdev_squared)
        if max2Ns:
            ex += max2Ns
            mode += max2Ns         
    elif densityof2Ns == "gamma":
        sd = math.sqrt(g[0]*g[1]*g[1])
        ex = -g[0]*g[1]
        mode = 0.0 if g[0] < 1 else -(g[0]-1)*g[1]
        if max2Ns:
            mode += max2Ns 
            ex += max2Ns
    #build an array of 2Ns values for integration
    #build using np.linspace over chunks of the range. The closer to mode, the more finely spaced
    sd10 = sd/10
    sd100 = sd/100
    listofarrays = [np.linspace(mode-sd,mode-sd10,10),np.linspace(mode-sd10,mode-sd100,10),np.linspace(mode-sd100,mode,10),np.linspace(mode,mode+sd100,10),np.linspace(mode+sd100,mode+sd10,10),np.linspace(mode+sd10,mode+sd,10)]
    for i in range(2,numSDs+1):
        listofarrays.insert(0,np.linspace(mode-i*sd,mode-(i-1)*sd,8))
        listofarrays.append(np.linspace(mode+(i-1)*sd,mode+ i*sd,8))
    # Concatenate the arrays
    xvals = np.concatenate(listofarrays)
    #stick in zero 
    xvals = np.append(xvals, 0.0)
    # Sort the concatenated array
    xvals = np.sort(xvals)
    # Remove duplicates using np.unique
    xvals = np.unique(xvals)
    if densityof2Ns in ("lognormal","gamma"):
        # Filter for values less than or equal to max2Ns
        xvals = xvals[xvals <= max2Ns]
        # Check if the highest value is equal to max2Ns
        if xvals[-1] == max2Ns:  # Replace the highest value with max2Ns - 1e-8
            xvals[-1] = max2Ns - 1e-8
        else: # append max2Ns - 1e-8
            xvals = np.append(xvals, max2Ns - 1e-8)
    # remove any values less than lowerbound_2Ns_integration
    xvals = xvals[xvals > lowerbound_2Ns_integration]

    density_values = np.array([prfdensity(x,g) for x in xvals])

    # check the integration.  Should be close to 1 
    densityadjust = float(np.trapz(density_values,xvals))
    return ex,mode,sd,densityadjust,xvals
 
def prfdensityfunction(g,densityadjust,nc ,i,arg1,arg2,max2Ns,densityof2Ns,foldxterm,misspec):
    """
    returns the product of poisson random field weight for a given level of selection (g) and a probability density for g 
    used for integrating over g 
    if foldxterm is true,  then it is a folded distribution AND two bins are being summed
    """
    us = prf_selection_weight(nc ,i,g,foldxterm,misspec)
    if densityof2Ns=="lognormal":   
        mean = arg1
        std_dev = arg2
        x = float(max2Ns-g)
        p = ((1 / (x * std_dev * sqrt_2_pi)) * np.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2)))/densityadjust
        if p==0.0:
           p= float(mpmath.fmul(mpmath.fdiv(1, (x * std_dev * sqrt_2_pi)), mpmath.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2))))/densityadjust
    elif densityof2Ns=="gamma":
        alpha = arg1
        beta = arg2
        x = float(max2Ns-g)
        p = (((x**(alpha-1))*np.exp(-x / beta))/((beta**alpha)*math.gamma(alpha)))/densityadjust
    elif densityof2Ns=="normal": # shouldn't need densityadjust for normal
        mu = arg1
        std_dev= arg2
        p = np.exp((-1/2)* ((g-mu)/std_dev)**2)/(std_dev * math.sqrt(2*math.pi))
    elif densityof2Ns=="discrete3":
        if g < -1:
            p=arg1/999
        elif g<= 1:
            p = arg2/2
        else:
            p = (1-arg1-arg2)/9
     
    if p*us < 0.0 or np.isnan(p):
        return 0.0
    return p*us

def trapzint2Ns(densityof2Ns,max2Ns,g,nc,i,foldxterm,misspec,xvals,densityadjust):
    """
        xvals is a numpy array 
    """
    density_values = np.array([prfdensityfunction(x,densityadjust,nc ,i,g[0],g[1],max2Ns,densityof2Ns,foldxterm,misspec) for x in xvals])
    intval = float(np.trapz(density_values,xvals))
    return intval
    
def NegL_SFS_Theta_Ns(p,nc,dofolded,includemisspec,counts): 
    """
        for fisher wright poisson random field model,  with with selection or without
        if p is a float,  then the only parameter is theta and there is no selection
        else p is a list (2 elements) with theta and Ns values 
        counts begins with a 0
        returns the negative of the log of the likelihood for a Fisher Wright sample 
    """
    # def L_SFS_Theta_Ns_bin_i(p,i,nc,dofolded,count): 
    def L_SFS_Theta_Ns_bin_i(i,count): 
        if isinstance(p,(float, int)): # p is simply a theta value,  no g  
            theta = p
            if theta <= 0:
                return -math.inf
            un = theta*nc/(i*(nc - i)) if dofolded else theta/i
            temp = -un +  math.log(un)*count - math.lgamma(count+1)
        else:
            theta = p[0]
            if theta <= 0:
                return -math.inf
            g = p[1]
            misspec = p[2] if includemisspec else 0.0 
            us = theta * prf_selection_weight(nc,i,g,dofolded,misspec)
            try:
                temp = -us +  math.log(us)*count - math.lgamma(count+1)
            except Exception as e:
                print("L_SFS_Theta_Ns_bin_i problem ",nc,i,g,us,theta,count)
                exit()

        return temp     
    assert(counts[0]==0)
    sum = 0
    for i in range(1,len(counts)):
        # sum += L_SFS_Theta_Ns_bin_i(p,i,nc,dofolded,counts[i])
        sum += L_SFS_Theta_Ns_bin_i(i,counts[i])
    return -sum 


def NegL_SFS_ThetaS_densityNs(p,max2Ns,nc ,dofolded,includemisspec,densityof2Ns,counts):
    """
        basic PRF likelihood
        returns negative of likelihood for the SFS 
        unknowns:
            thetaS
            terms for 2Ns density
    """
    sum = 0
    thetaS = p[0]
    term1 = p[1]
    term2 = p[2]
    misspec = p[3] if includemisspec else 0.0 
    ex,mode,sd,densityadjust,g_xvals = getXrange(densityof2Ns,(term1,term2),max2Ns)
    for i in range(1,len(counts)):
        intval = trapzint2Ns(densityof2Ns,max2Ns,(term1,term2),nc,i,dofolded,misspec,g_xvals,densityadjust)
        us = thetaS*intval 
        sum += -us + math.log(us)*counts[i] - math.lgamma(counts[i]+1)        
    return -sum    
 
def NegL_SFSRATIO_estimate_thetaS_thetaN(p,nc,dofolded,includemisspec,densityof2Ns,onetheta,max2Ns,usepm,usepm0,fix_mode_0,zvals): 
    """
        returns the negative of the log of the likelihood for the ratio of two SFSs
        estimates Theta values,  not their ratio

        densityof2Ns in fix2Ns0,fixed2Ns,normal,lognormal,gamma,discrete3 
        onetheta in True, False
        max2Ns  is either None,  or a fixed max value 
        usepm0 in True, False
        fix_mode_0 in True, False 


        replaces:
            def NegL_SFSRATIO_thetaS_thetaN_fixedNs(p,nc ,dofolded,zvals,nog,usepm0)
            def NegL_SFSRATIO_thetaS_thetaN_densityNs_max2Ns(p,max2Ns,nc ,maxi,dofolded,densityof2Ns,zvals)
            
        returns negative of likelihood using the probability of the ratio 
        unknown     # params
        thetaN,thetaS 1 if onetheta else 2 
        Ns terms    2 if densityNs is not fixed2Ns 1 
        max2Ns      1 if densityof2Ns is in ("lognormal","gamma") and max2Ns is None else 0 
        pointmass0  1 if usepm0 else 0 

        handles fix_mode_0 i.e. setting max2Ns so the mode of the distribution is 0 
        handles dofolded 
    """
    def calc_bin_i(i,z): 
        if densityof2Ns in ("fix2Ns0","fixed2Ns"):
            try:
                if z==math.inf or z==0.0:
                    return 0.0
                uy = thetaN*nc /(i*(nc -i)) if dofolded else thetaN/i     
                if g == 0:
                    alpha = thetaS/thetaN
                else:
                    ux = thetaS*prf_selection_weight(nc,i,g,dofolded,misspec)
                    if densityof2Ns == "fixed2Ns":
                        if usepm0:
                            ux = pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*ux
                        elif usepm:
                            ux = thetaS * pmass * prf_selection_weight(nc,i,pval,dofolded,misspec) +  (1-pmass)*ux

                    alpha = ux/uy
                sigmay = math.sqrt(uy)
                beta = 1/sigmay
                return logprobratio(alpha,beta,z)
            except:
                return -math.inf                
        else:
            try:
                if densityof2Ns == "discrete3":
                    # density_values = np.array([prfdensityfunction(x,None,nc ,i,g[0],g[1],None,densityof2Ns,foldxterm,misspec) for x in discrete3_xvals])
                    # ux = thetaS*float(np.trapz(density_values,discrete3_xvals))
                    ux = thetaS*trapzint2Ns(densityof2Ns,max2Ns,g,nc,i,foldxterm,misspec,discrete3_xvals,densityadjust)
                else:
                    ux = thetaS*trapzint2Ns(densityof2Ns,max2Ns,g,nc,i,foldxterm,misspec,g_xvals,densityadjust)
                    if usepm0:
                        ux = pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*ux
                    elif usepm:
                        ux = thetaS * pmass * prf_selection_weight(nc,i,pval,dofolded,misspec) +  (1-pmass)*ux
                uy = thetaN*nc /(i*(nc -i)) if foldxterm else thetaN/i    
                alpha = ux/uy
                sigmay = math.sqrt(uy)
                beta = 1/sigmay
                return logprobratio(alpha,beta,z)   
            except:
                return -math.inf 
            
    assert zvals[0]==0 or zvals[0] == math.inf
    try:
        p = list(p)
    except :
        p = [p]
        pass  # if this function is called by scipy minimize then p is just a scalar 
    if onetheta:
        thetaN = thetaS = p[0]
        unki = 1
    else:
        thetaN = p[0]
        thetaS = p[1]
        unki = 2
    if densityof2Ns == "fixed2Ns":
        g = p[unki]
        unki += 1
    elif densityof2Ns == "fix2Ns0":
        g = 0
    else:
        g = (p[unki],p[unki+1])
        if densityof2Ns=="discrete3":
            if ((0 < g[0] < 1) == False) or ((0 < g[1] < 1) == False) or ((g[0] + g[1]) >= 1):
                return math.inf
        holdki = unki
        unki += 2
    if usepm0:
        pm0 = p[unki]
        unki += 1
       
    if usepm:
        pmass = p[unki]
        pval = p[unki+1]
        unki += 2          
    if max2Ns==None and densityof2Ns in ("lognormal","gamma"):
        if fix_mode_0:
            if densityof2Ns=="lognormal":
                max2Ns = math.exp(p[holdki] - pow(p[holdki],2))
            if densityof2Ns=="gamma":
                if p[holdki] < 1:
                    max2Ns = 0.0
                else:
                    max2Ns = (p[holdki] - 1)*p[holdki+1]
        else:
            max2Ns = p[unki]
            unki += 1
      
    if includemisspec:
        misspec = p[unki] 
        unki += 1
    else:
        misspec = 0.0        
    if densityof2Ns not in ("fixed2Ns","discrete3","fix2Ns0"):
        ex,mode,sd,densityadjust,g_xvals = getXrange(densityof2Ns,g,max2Ns)
        if mode < minimum_2Ns_location or (mode - sd) < minimum_2Ns_location: # distribution is located way out in highly negative values,  
            return math.inf


    sum = 0
    for i in range(1,len(zvals)):
        foldxterm = dofolded and i < nc //2 # True if summing two bins, False if not 
        temp =  calc_bin_i(i,zvals[i])
        sum += temp
        if sum == -math.inf:
            return math.inf
            # raise Exception("sum==-math.inf in NegL_SFSRATIO_Theta_Nsdensity_given_thetaN params: " + ' '.join(str(x) for x in p))
    # print(-sum,p)
    return -sum   


def NegL_SFSRATIO_estimate_thetaratio(p,nc,dofolded,includemisspec,densityof2Ns,fix_theta_ratio,max2Ns,usepm,usepm0,fix_mode_0,zvals): 
    """
        returns the negative of the log of the likelihood for the ratio of two SFSs
        first parameter is the ratio of mutation rates
        sidesteps the theta terms by integrating over thetaN in the probability of the ratio (i.e. calls intdeltalogprobratio())

        densityof2Ns in fix2Ns0,fixed2Ns,normal,lognormal,gamma,discrete3 
        fixthetaratio is either None, or a fixed value for the ratio 
        max2Ns  is either None,  or a fixed max value 
        usepm0 in True, False
        fix_mode_0 in True, False 

        replaces:
            NegL_SFSRATIO_ratio_fixedNs
            NegL_SFSRATIO_ratio_densityNs
            NegL_SFSRATIO_ratio_densityNs_pointmass0

        returns negative of likelihood using the probability of the ratio 
        unknown     # params
        ratio       0 if fix_theta_ratio is not None else 1 
        Ns terms    2 if densityNs is not None else 1 
        max2Ns      1 if densityof2Ns is in ("lognormal","gamma") and max2Ns is None else 0 
        pointmass0  1 if usepm0 else 0 

        handles fix_mode_0 i.e. setting max2Ns so the mode of the distribution is 0 
        handles dofolded 
    """
    def calc_bin_i(i,z): 
        if densityof2Ns in ("fixed2Ns","fix2Ns0"):
            try:
                if z==math.inf or z==0.0:
                    return 0.0
                if g == 0:
                    alpha = thetaratio
                else:
                    sint = prf_selection_weight(nc,i,g,foldxterm,misspec)
                    if densityof2Ns == "fixed2Ns":
                        if usepm0:
                            sint = pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*sint
                        elif usepm:
                            sint = pmass*prf_selection_weight(nc,i,pval,foldxterm,misspec) + (1-pmass) * sint

                    alpha = thetaratio*sint/(nc /(i*(nc -i)) if foldxterm else 1/i )
                uy = thetaNspace * nc /(i*(nc -i)) if foldxterm else thetaNspace/i    
                deltavals = 1/np.sqrt(uy)
                return intdeltalogprobratio(alpha,z,deltavals)        
            except:
                return -math.inf                
        else:
            try:
                if densityof2Ns == "discrete3":
                    # density_values = np.array([prfdensityfunction(x,None,nc ,i,g[0],g[1],None,densityof2Ns,foldxterm,misspec) for x in discrete3_xvals])
                    # ux = thetaS*float(np.trapz(density_values,discrete3_xvals))
                    ux = trapzint2Ns(densityof2Ns,max2Ns,g,nc,i,foldxterm,misspec,discrete3_xvals,densityadjust)
                else:
                    ux = trapzint2Ns(densityof2Ns,max2Ns,g,nc,i,foldxterm,misspec,g_xvals,densityadjust)
                    if usepm0:
                        ux = pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*ux
                    elif usepm:
                        ux =  (1-pmass)*ux + pmass* prf_selection_weight(nc,i,pval,foldxterm,misspec)
                alpha = thetaratio*ux/(nc /(i*(nc -i)) if foldxterm else 1/i )
                uy = thetaNspace * nc /(i*(nc -i)) if foldxterm else thetaNspace/i    
                deltavals = 1/np.sqrt(uy)
                return intdeltalogprobratio(alpha,z,deltavals)        
            except:
                return -math.inf 
            
    assert zvals[0]==0 or zvals[0] == math.inf
    if isinstance(p,(int,float)):
        p = [p]
    else:
        p = list(p)
    unki = 0
    if fix_theta_ratio in (None,False):
        thetaratio = p[0]
        unki = 1
    else:
        thetaratio = fix_theta_ratio
    if densityof2Ns == "fixed2Ns":
        g = p[unki]
        unki += 1
    elif densityof2Ns  == "fix2Ns0":
        g = 0.0
    else:
        g = (p[unki],p[unki+1])
        if densityof2Ns=="discrete3":
            if ((0 < g[0] < 1) == False) or ((0 < g[1] < 1) == False) or ((g[0] + g[1]) >= 1):
                return math.inf
        unki += 2
    if usepm0:
        pm0 = p[unki]
        unki += 1
    if usepm:
        pmass = p[unki]
        pval = p[unki+1]
        unki += 2        
    if max2Ns==None and densityof2Ns in ("lognormal","gamma"):
        if fix_mode_0:
            if densityof2Ns=="lognormal":
                max2Ns = math.exp(p[1] - p[2]*p[2])
            if densityof2Ns=="gamma":
                if p[1] < 1:
                    max2Ns = 0.0
                else:
                    max2Ns = (p[1] - 1)*p[2]
        else:
            max2Ns = p[unki]
            unki += 1

    if includemisspec:
        misspec = p[unki] 
        unki += 1
    else:
        misspec = 0.0        
    if densityof2Ns in ("normal","lognormal","gamma"):
        ex,mode,sd,densityadjust,g_xvals = getXrange(densityof2Ns,g,max2Ns)
        if mode < minimum_2Ns_location or (mode - sd) < minimum_2Ns_location: # distribution is located way out in highly negative values,  
            # print(g,mode)
            return math.inf
    sum = 0
    for i in range(1,len(zvals)):
        foldxterm = dofolded and i < nc //2 # True if summing two bins, False if not 
        temp =  calc_bin_i(i,zvals[i])
        sum += temp
        if sum == -math.inf:
            return math.inf
            # raise Exception("sum==-math.inf in NegL_SFSRATIO_Theta_Nsdensity_given_thetaN params: " + ' '.join(str(x) for x in p))
    # print(-sum,p)
    return -sum   

def simsfs_continuous_gdist(theta,max2Ns,nc,misspec,maxi,densityof2Ns, params,pm0, returnexpected,pmmass = None,pmval = None):
    """
    nc  is the # of sampled chromosomes 

    simulate the SFS under selection, assuming a PRF Wright-Fisher model 
    uses a distribution of g (2Ns) values 
    gdist is "lognormal" or "gamma" ,params is two values

    return folded and unfolded    
    """
    sfs = [0]*nc 
    for i in range(1,nc):
        if densityof2Ns in ("normal","lognormal","gamma"):
            ex,mode,sd,densityadjust,g_xvals = getXrange(densityof2Ns,params,max2Ns)
            sint = trapzint2Ns(densityof2Ns,max2Ns,(params[0],params[1]),nc,i,False,misspec,g_xvals,densityadjust)
        elif densityof2Ns=="discrete3":
            sint = trapzint2Ns(densityof2Ns,max2Ns,(params[0],params[1]),nc,i,False,misspec,discrete3_xvals,None)
        
        if pm0 is not None:
            sint = pm0/i + (1-pm0)*sint
        elif pmmass is not None:
            sint = pmmass * prf_selection_weight(nc ,i,pmval,False,misspec) + (1-pmmass) * sint
        sfsexp = theta*sint
        assert sfsexp>= 0
        if returnexpected:
            sfs[i] = sfsexp
        else:
            sfs[i] = np.random.poisson(sfsexp)

    sfsfolded = [0] + [sfs[j]+sfs[nc -j] for j in range(1,nc //2)] + [sfs[nc //2]]
    if maxi:
        assert maxi < nc , "maxi setting is {} but nc  is {}".format(maxi,nc )
        sfs = sfs[:maxi+1]
        sfsfolded = sfsfolded[:maxi+1]            
    return sfs,sfsfolded

def simsfs(theta,g,nc , misspec,maxi, returnexpected):
    """
        nc  is the # of sampled chromosomes 

        simulate the SFS under selection, assuming a PRF Wright-Fisher model 
        uses just a single value of g (2Ns), not a distribution
        if returnexpected,  use expected values, not simulated
        generates,  folded and unfolded for Fisher Wright under Poisson Random Field
        return folded and unfolded 
    """
    # g = -10000
    if g==0:
        if misspec in (None,False, 0.0):
            sfsexp = [0]+[theta/i for i in range(1,nc )]
        else:
            sfsexp = [0]+[theta*((1-misspec)/i + misspec/(nc-i)) for i in range(1,nc )]
    else:
        sfsexp = [0]
        for i in range(1,nc ):
            u = prf_selection_weight(nc ,i,g,False,misspec)
            sfsexp.append(u*theta)    
    if returnexpected:
        sfs = sfsexp
    else:    
        try:
            sfs = [np.random.poisson(expected) for expected in sfsexp]
        except Exception as e:
            print(e)
            print(g)
            print(sfsexp)
            exit()
    sfsfolded = [0] + [sfs[j]+sfs[nc -j] for j in range(1,nc //2)] + [sfs[nc //2]]
    if maxi:
        assert maxi < nc , "maxi setting is {} but nc  is {}".format(maxi,nc )
        sfs = sfs[:maxi+1]
        sfsfolded = sfsfolded[:maxi+1]            
    return sfs,sfsfolded


def simsfsratio(thetaN,thetaS,max2Ns,nc ,maxi,dofolded,misspec,densityof2Ns,params,pm0, returnexpected, thetaratio,pmmass = None,pmval = None):
    """
     nc  is the # of sampled chromosomes 

    simulate the ratio of selected SFS to neutral SFS
    if returnexpected,  use expected values, not simulated
    if gdist is None,  params is just a g value,  else it is a list of distribution parameters
    if a bin of the neutral SFS ends up 0,  the program stops

    if ratio is not none, thetaS = thetaratio*thetaN

    pm0 is point mass 0,  as of 2/4/2024 used only by run_one_pair_of_SFSs.py
    """
    
    nsfs,nsfsfolded = simsfs(thetaN,0,nc ,misspec,maxi,returnexpected)

    if thetaratio is not None:
        thetaS = thetaN*thetaratio
    if densityof2Ns == "fixed2Ns":
        ssfs,ssfsfolded = simsfs(thetaS,params[0],nc ,misspec,maxi,returnexpected)
    else:
        ssfs,ssfsfolded = simsfs_continuous_gdist(thetaS,max2Ns,nc ,misspec,maxi,densityof2Ns,params,pm0,returnexpected,pmmass = pmmass,pmval = pmval)
    if dofolded:
        ratios = [math.inf if nsfsfolded[j] <= 0.0 else ssfsfolded[j]/nsfsfolded[j] for j in range(len(nsfsfolded))]
        return nsfsfolded,ssfsfolded,ratios
    else:
        ratios = [math.inf if nsfs[j] <= 0.0 else ssfs[j]/nsfs[j] for j in range(len(nsfs))]
        return nsfs,ssfs,ratios



