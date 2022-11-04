import numpy as np 
import pylab as py 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from matplotlib import rcParams
import time
import matplotlib
rcParams["font.size"] = 11.5
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import StrMethodFormatter
rcParams["font.size"] = 11.5
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
matplotlib.rcParams['text.usetex']=True
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
#from sklearn.tree import export_graphviz 
from sklearn.tree import export_text
from sklearn import tree
cmap=plt.get_cmap('viridis')



f1=open("./forest1.txt","r")
nm= sum(1 for line in f1) 
par1=np.zeros((nm,5)) 
par1= np.loadtxt("./forest1.txt")



f2=open("./forest2.txt","r")
nm= sum(1 for line in f2) 
par2=np.zeros((nm,5)) 
par2= np.loadtxt("./forest2.txt")



plt.clf()
fig= plt.figure(figsize=(8,6)) 
plt.plot(par1[:,0]*1.0, np.log10(par1[:,4]), "b--", lw=2.0, label=r"$\rm{Without}~\rm{limb}-\rm{darkening}$")
plt.plot(par2[:,0]*1.0, np.log10(par2[:,4]), "m--", lw=2.0, label=r"$\rm{With}~\rm{limb}-\rm{darkening}$")
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$\rm{Number}~\rm{of}~\rm{Trees}$", fontsize=17)
plt.ylabel(r"$\log_{10}[\rm{RMSE}]$", fontsize=17)
#plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.legend(loc='best',fancybox=True, shadow=True, fontsize=18)
plt.grid(True)
plt.grid(linestyle='dashed')
fig.tight_layout()
plt.savefig("Forest_both.jpg", dpi=200)
 
