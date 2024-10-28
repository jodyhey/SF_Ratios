
import sys
import argparse
import os.path as op
try:
    import numpy as np
    from scipy.stats import lognorm, gamma, norm
    import matplotlib.pyplot as plt
except Exception as e:
    print(e)
    exit()


def parse_file(filepath):
    """
        crappy code
        files can vary a lot in terms of their model parameters 
    """
    Keys = [
        "File","fileMtime","IntCheck","NonS/Syn", "numparams","FixQR", "Density", "SetMax2Ns", "FixMode0", "PMmode",
        "Lklhd","AIC", "Qratio","Qratio_CI", "p1","p1_CI", "p2","p2_CI","estMax2Ns","estMax2Ns_CI",
        "pm0_mass","pm0_mass_CI","pm_mass","pm_mass_CI","pm_val","pm_val_CI","Mean", "Mode"
        ]

    data = {key: None for key in Keys}
    with open(filepath, 'r') as file:
        lines = file.readlines()
    # print(filepath)
    # last_modified_datetime = op.getmtime(filepath)
    # last_modified_datetime = datetime.fromtimestamp(last_modified_datetime)
    # data["fileMtime"] = f"{last_modified_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    li = 0
    while True:

        if "Command line arguments:" in lines[li] or lines[li][:10] == "Arguments:":
            li += 1
            break
        li += 1
    data["PMmode"]  = "FALSE"
    while len(lines[li]) > 2 :
        if "sfsfilename:" in lines[li]:
            data["File"] = op.split(filepath)[-1]
            data["NonS/Syn"] = "NONSYN" if "NONSYN" in data["File"].upper() else "SYN"
        if "fix_theta_ratio" in lines[li]:
            data["FixQR"] = lines[li].split(":")[1].strip()
        if "densityof2Ns" in lines[li]:
            data["densityof2Ns"] = lines[li].split(":")[1].strip()
        if "max2Ns" in lines[li] and "estimatemax2Ns" not in lines[li]:
            data["SetMax2Ns"] = lines[li].split(":")[1].strip()
        if "fixmode0" in lines[li]:
            data["FixMode0"] = lines[li].split(":")[1].strip()
        if "estimate_pointmass0:" in lines[li]:
            temp = lines[li].split(":")[1].strip()
            if temp == "True":
                data["PMmode"] = "PM_at_0"
        if "estimate_pointmass:" in lines[li]:
            temp = lines[li].split(":")[1].strip()
            if temp == "True":
                data["PMmode"] = "PM_M_and_V"
        if "numparams" in lines[li]:
            data["numparams"] = int(lines[li].split(":")[1].strip())
        li += 1

    while "trial\t" not in lines[li]:
        if li >= len(lines):
            break
        li += 1
    li += 1

# Qratio	p1	p2	estMax2Ns	pm0_mass	pm_mass	pm_val
    tempcounter = 0
    tempatlike = False
    while data["Mode"] == None: #"Mode is the last thing read here "
        if li >= len(lines):
            break
        line = lines[li]
        if "AIC\t" in  line:
            data["AIC"] = float(line.strip().split()[1])
        if "likelihood" in line and "thetaratio" not in line:
            data["Lklhd"] = line.split('\t')[1].strip() if '\t' in line else None
            tempcounter = 0  # First four lines after "basinhopping" are liklihood, thetaratio, and then the two parameters
            tempatlike = True
        elif "distribution check" in line:
            start = line.find("distribution check:") + len("distribution check: ")
            end = line.find(" ", start)
            data["IntCheck"] = float(line[start:end])
        if tempatlike:
            if tempcounter == 1:
                if  data["FixQR"] in (None,"None"):
                    data["Qratio"] = line.split('\t')[1].strip() if '\t' in line else None
                    data["Qratio_CI"] = line.split('\t')[2].strip() if '\t' in line else None
                else:
                    data["Qratio"] = None
                    data["Qratio_CI"] = None
                    tempcounter += 1
            if tempcounter == 2:
                data["p1"] = line.split('\t')[1].strip() if '\t' in line else None
                data["p1_CI"] = line.split('\t')[2].strip() if '\t' in line else None                
            if tempcounter == 3:
                if data["densityof2Ns"] != 'single2Ns':
                    data["p2"] = line.split('\t')[1].strip() if '\t' in line else None
                    data["p2_CI"] = line.split('\t')[2].strip() if '\t' in line else None                
                else:
                    data["p2"] = data["p2_CI"] = None
        if "max2Ns" in line:
            data["estMax2Ns"] = line.split('\t')[1].strip() if '\t' in line else None
            data["estMax2Ns_CI"] = line.split('\t')[2].strip() if '\t' in line else None
        elif "pm0" in line:
            data["pm0_mass"] = line.split('\t')[1].strip() if '\t' in line else None
            data["pm0_mass_CI"] = line.split('\t')[2].strip() if '\t' in line else None
        elif "pm_mass" in line:
            data["pm_mass"] = line.split('\t')[1].strip() if '\t' in line else None
            data["pm_mass_CI"] = line.split('\t')[2].strip() if '\t' in line else None
        elif "pm_val" in line:
            data["pm_val"] = line.split('\t')[1].strip() if '\t' in line else None
            data["pm_val_CI"] = line.split('\t')[2].strip() if '\t' in line else None
        elif "expectation" in line or "Mean" in line:
            data["Mean"] = line.split('\t')[1].strip() if '\t' in line else None
            # if data["Mean"] == 'nan':
            #     if data["densityof2Ns"]=='single2Ns':
            #         if data["PMmode"] == "PM_at_0":
            #             data["Mean"] = (1- float(data["pm0_mass"] ))* float(data["p1"]) 
            #             if float(data["pm0_mass"]) > 0.5:
            #                 data["Mode"] = 0.0
            #             else:
            #                 data["Mode"] = data["p1"]
            #         elif data["PMmode"] == "PM_M_and_V":
            #             data["Mean"] = (1- float(data["pm_mass"] ))* float(data["p1"])  + float(data["pm_mass"] ) * float(data["pm_val"]) 
            #             if float(data["pm_mass"]) > 0.5:
            #                 data["Mode"] = data["pm_val"]
            #             else:
            #                 data["Mode"] = data["p1"]
            #         else:
            #             data["Mean"] = data["p1"]
            #             data["Mode"] = data["p1"]
            #     elif data['densityof2Ns'] == 'discrete3':
            #         data["Mean"] = -(11/2)* (-1 + 92*float(data["p1"]) + float(data["p2"]))
        elif "mode" in line:
            data["Mode"] = line.split('\t')[1].strip() if '\t' in line else None
        tempcounter += 1  
        li += 1
    return data
    
def check_filename_make_new_numbered_as_needed(filename,filecheck):
    """
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

# Example usage
# base = "NONSYN_Samp_gamma"
# new_filename = generate_filename(base)
# print(new_filename)

# def generate_distribution_plot(densityof2Ns, params,args, Max2Ns, point_mass, mode, expectation, likelihood, poplabel,sfsfilename, filename):
def generate_distribution_plot(data,poplabel,filename,xliml=None,xlimr = None):
    """
    Generate a plot of a specified distribution with an optional point mass.
    Only works for continuous distributions, with or without a point mass 
    """

    def plot_main(ax, x, y, label=None, color='blue'):
        """
        Helper function to plot the main distribution.
        """
        ax.plot(x, y, label=label, color=color)
        ax.set_ylabel('Density', fontsize=14)

    def plot_point_mass(ax, height, x_loc=0, width=1, label='Point Mass'):
        """
        Helper function to plot the point mass as a vertical bar.
        """
        ax.bar(x_loc, height, width=width, color='red', label=label, align='center')
        ax.set_ylim(bottom=0)  # Ensure the bottom starts at 0


    if data["SetMax2Ns"] not in (None,"None"):
        max2Ns = float(data["SetMax2Ns"])
    elif data["estMax2Ns"] not in (None,"None"):
        max2Ns = float(data["estMax2Ns"])
    else:
        max2Ns = None
    
    # tempMax2Ns = None if data["Density"]=="normal" else (float(data["estMax2Ns"]) if data["estMax2Ns"]  is not None else float(data["Max2Ns"]))
    # densityof2Ns, params,estimatemax2Ns, M, point_mass,mode,expectation,likelihood,poplabel, filename)
    try:
        data["pm0_mass"] = float(data["pm0_mass"])
    except:
        pass
    try:
        data["pm_mass"] = float(data["pm_mass"])
    except:
        pass
    try:
        data["pm_val"] = float(data["pm_val"])
    except:
        pass
    try:
        data["Mode"] = float(data["Mode"])
    except:
        pass 

    Keys = [
        "File","fileMtime","IntCheck","NonS/Syn", "numparams","FixQR", "densityof2Ns", "SetMax2Ns", "FixMode0", "PMmode",
        "Lklhd","AIC", "Qratio","Qratio_CI", "p1","p1_CI", "p2","p2_CI","estMax2Ns","estMax2Ns_CI",
        "pm0_mass","pm0_mass_CI","pm_mass","pm_mass_CI","pm_val","pm_val_CI","Mean", "Mode"
        ]

    # Set up figure and main axis
    fig, ax_main = plt.subplots(figsize=(8, 6))
    ax_main.set_xlabel('2$N_e s$', fontsize=14, fontstyle='italic')
    if xliml:
        ax_main.set_xlim(left = xliml)
    if xlimr:
        ax_main.set_xlim(right = xlimr)
    # Determine distribution type and calculate y values
    x, y = np.linspace(-10, 10, 1000), None  # Default x values
    if data["densityof2Ns"] == 'lognormal':
        scale, std_dev = np.exp(float(data["p1"])), float(data["p2"])
        x = np.linspace(50, 1e-5, 1000)
        y = lognorm.pdf(x, scale=scale, s=std_dev)

        x = -x if max2Ns is None else max2Ns - x
    elif data["densityof2Ns"] == 'gamma':
        alpha, scale = float(data["p1"]),float(data["p2"])
        x = np.linspace(50, 1e-5, 1000)
        y = gamma.pdf(x, a=alpha, scale=scale)
        x = -x if max2Ns is None else max2Ns - x
    elif data["densityof2Ns"] == 'normal':
        bounds = (float(data["p1"]) - 3 * float(data["p2"]), float(data["p1"]) + 3 * float(data["p2"]))
        x = np.linspace(bounds[0], bounds[1], 1000)
        y = norm.pdf(x, loc=float(data["p1"]), scale=float(data["p2"]))
    else:
        raise ValueError("Unsupported distribution type")

    # Plot main distribution
    plot_main(ax_main, x, y)
    poplabel += "\nNonSynonymous" if "nons" in data["File"].lower() else "\nSynonymous"
    if data["densityof2Ns"]=="normal":
        infostr = "{}\nGaussian\nMean: {:.3f}\nStd Dev: {:.3f}".format(poplabel,float(data["p1"]),float(data["p2"]))
        if data["pm0_mass"] not in (None,0.0) or data["pm_mass"] not in (None,0.0):
            infostr += "\nMean with point mass: {:0.3f}".format(float(data["Mean"]))            
    elif data["densityof2Ns"]=="lognormal":
        infostr = "{}\nLogNormal\n".format(poplabel) + r'$\mu$: {:.3f}'.format(float(data["p1"])) +"\n" + r'$\sigma$: {:.3f}'.format(float(data["p2"])) + "\n" + "Mode: {:.3f}\nMean: {:.3f}".format(float(data["Mode"]),float(data["Mean"]))
    elif data["densityof2Ns"]=="gamma":
        infostr = "{}\nGamma\n".format(poplabel) + r'$\alpha$: {:.3f}'.format(float(data["p1"])) +"\n" + r'$\beta$: {:.3f}'.format(float(data["p2"])) + "\n" + "Mode: {:.3f}\nMean: {:.3f}".format(float(data["Mode"]),float(data["Mean"]))
    if data["SetMax2Ns"] not in ("None",None):
        infostr += "\nFixed max 2$N_e s$: {}".format(float(data["SetMax2Ns"]))
    elif data["estMax2Ns"]  not in ("None",None):
        infostr += "\nMax 2$N_e s$: {:.3f}".format(float(data["estMax2Ns"]))
    if data["pm0_mass"] not in (None,0.0):
        infostr += "\nPoint mass at 0: {:0.3f}".format(float(data["pm0_mass"]))            
    elif data["pm_mass"] not in (None,0.0):
        infostr += "\nPoint mass {:0.3f} at {:0.3f}".format(float(data["pm_mass"]),float(data["pm_val"]))

    infostr += "\nLikelihood: {:.2f}".format(float(data["Lklhd"]))
    infostr += "\nAIC: {:.2f}".format(float(data["AIC"]))

# Adding the text box in the upper left corner of the plot
    plt.text(0.05, 0.95, infostr, transform=plt.gca().transAxes, fontsize=14, 
         verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.12))

    # Check if we need to plot point mass
    # if data["pm_mass"] not in (None, 0):
    if data["PMmode"] != "FALSE":
        # if isinstance(data["pm_mass"], (list, tuple)) and len(data["pm_mass"]) == 2:
        if data["PMmode"] == "PM_M_and_V":
            point_height, point_location = float(data["pm_mass"]),float(data["pm_val"])
        elif data["PMmode"] == "PM_at_0":
            point_height, point_location = float(data["pm0_mass"]), 0
        else:
            print("point mass error")
            exit()
        # Determine if a split axis is needed
        if point_height > max(y) * 10:  # Arbitrary threshold for split axis
            # Create a split y-axis
            ax_split = ax_main.twinx()
            plot_point_mass(ax_split, point_height, x_loc=point_location, width=1)
            ax_split.set_ylim(bottom=min(y), top=point_height * 1.2)  # Adjust top for visibility
            ax_split.set_yticks([])  # Hide y-ticks for split axis
        else:
            plot_point_mass(ax_main, point_height, x_loc=point_location, width=1)

    # Setting the x and y axis labels with the specified formatting
    # plt.xlabel(r'$\textit{2Ns}$', fontsize=14, fontstyle='italic')
    plt.xlabel('2$N_e s$', fontsize=14, fontstyle='italic')
    plt.ylabel('Probability', fontsize=14)
    plt.savefig(filename, format='png')
    plt.close()

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="filename",required=True,type = str, help="file path")
    parser.add_argument("-p",dest="poplabel",default = "",type=str,help="populuation/dataset name")
    parser.add_argument("-l",dest="xliml",default = None,type=float,help="xlim left")
    parser.add_argument("-r",dest="xlimr",default = None,type=float,help="xlim right")
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    return args

args = parsecommandline()

if "_estimates.out" in args.filename:
    _,pngname = check_filename_make_new_numbered_as_needed(args.filename[:args.filename.find("_estimates.out")] + ".png",True)
else:
    pngname = args.filename + ".png"


data = parse_file(args.filename)

#densityof2Ns, params,args, Max2Ns, point_mass, mode, expectation, likelihood, poplabel,sfsfilename, filename
# generate_distribution_plot(data["Density"],(float(data["p1"]),float(data["p2"])),estimateMax2Ns,data["Max2Ns"],data["estPM0"],float(data["Mode"]),float(data["Mean"]),float(data["Lklhd"]),plotlabel,pngname)
generate_distribution_plot(data,args.poplabel,pngname,xliml=args.xliml,xlimr = args.xlimr)



