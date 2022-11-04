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
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
#from sklearn.tree import export_graphviz 
from sklearn.tree import export_text
from sklearn import tree

cmap=plt.get_cmap('viridis')
labell=[r"$m_{\rm{base}}$",r"$\Delta m(t_{0})$",r"$\rm{FWHM}$",r"$T_{\rm{max}}$",r"$f_{\rm{ws}}$", r"$f_{\rm{pl}}$",r"$\rho_{\star}$",r"$t_{\star}$", r"$u_{\rm{r}}$",r"$f_{\rm{b}}$",r"$m_{\star}$", r"$\Gamma$"]
#######################################################################
df= pd.read_csv("./efsf2.csv", sep=",",  skipinitialspace = True)
print("describe:  ",  df.describe() )
print("*****************************************")
print("Columns:  ",  df.columns, "len(columns):  ",  len(df.columns)   )
print("*****************************************")

corrM = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
corrM.style.background_gradient(cmap='coolwarm').set_precision(2)
ax= sns.heatmap(corrM, annot=True, xticklabels=labell, yticklabels=labell,annot_kws={"size": 16}, square=True, linewidth=1.0, cbar_kws={"shrink": .99}, linecolor="k",fmt=".2f", cbar=True, vmax=1, vmin=-1, center=0.0, ax=None, robust=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
plt.xticks(rotation=45,horizontalalignment='right',fontweight='light', fontsize=18)
plt.yticks(rotation=0, horizontalalignment='right',fontweight='light', fontsize=18)
plt.title(r"$\rm{Correlation}~\rm{Matrix}$", fontsize=19)
fig.tight_layout()
plt.savefig("corr2.jpg", dpi=200)
print("**** Correlation matrix was calculated ******** ")
input("Enter a number ")

###########################################################################
'''
f1=open("./param2.txt","r")
nm= sum(1 for line in f1) 
par=np.zeros((nm,20)) 
par= np.loadtxt("./param2.txt")
#icon, mbase, fabs(magm),fabs(FWHM), fabs(tmax1), fws, fli, rho, tstar, u0/rho, fb, mstar, limb, tim1, tmm, tim2, twing, tshol, error,N);
# 0     1       2          3            4           5   6    7    8       9     10    11     12   13    14   15    16      17     18  19
'''
#######################################################
'''
plt.clf()
fig= plt.figure(figsize=(8,6))
ax= plt.gca()              
plot=plt.scatter(abs(par[:,8]),abs(par[:,3]),  c=par[:,9],  cmap=cmap, s=1.5 )
cb=plt.colorbar(plot)
cb.ax.tick_params(labelsize=16)
cb.set_label(r"$u_{0}/\rho_{\star}$", fontsize=17)
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$t_{\star}$",fontsize=18)
plt.ylabel(r"$FWHM$",fontsize=18)
plt.grid("True")
plt.grid(linestyle='dashed')
fig=plt.gcf()
fig.savefig("./lights/scatter1b.jpg",dpi=200)
#######################################################
plt.clf()
fig= plt.figure(figsize=(8,6))
ax= plt.gca()              
plot=plt.scatter(abs(par[:,8]),abs(par[:,4]), c=par[:,9], cmap=cmap, s=1.4)
cb=plt.colorbar(plot)
cb.ax.tick_params(labelsize=16)
cb.set_label(r"$ u_{0}/\rho_{\star}$", fontsize=17)
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$t_{\star}$",fontsize=18)
plt.ylabel(r"$T_{max}$",fontsize=18)
plt.grid("True")
plt.grid(linestyle='dashed')
fig=plt.gcf()
fig.savefig("./lights/scatter2b.jpg",dpi=200)
'''
#######################################################
'''
plt.clf()
fig= plt.figure(figsize=(8,6))
ax= plt.gca()              
plot= plt.scatter(abs(par[:,3]),abs(par[:,4]),c=par[:,9], cmap=cmap, s=1.4 )##3.5
cb=plt.colorbar(plot)
cb.ax.tick_params(labelsize=16)
cb.set_label(r"$u_{0}/\rho_{\star}$", fontsize=17)
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$\rm{FWHM}$",fontsize=18)
plt.ylabel(r"$t_{\rm{max}}$",fontsize=18)
plt.grid("True")
plt.grid(linestyle='dashed')
fig=plt.gcf()
fig.savefig("./lights/scatter3b.jpg",dpi=200)
'''
#######################################################
'''
plt.clf()
fig= plt.figure(figsize=(8,6))
ax= plt.gca()              
plot=plt.scatter(abs(par[:,9]),abs(abs(par[:,3])-abs(par[:,4])), c=np.log10(par[:,7]), cmap=cmap)
cb=plt.colorbar(plot)
cb.ax.tick_params(labelsize=16)
cb.set_label(r"$\log_{10}[\rho_{\star}]$", fontsize=17)
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.ylabel(r"$|FWHM -T_{max}|$",fontsize=18)
plt.xlabel(r"$u_{r}$",fontsize=18)
plt.grid("True")
plt.grid(linestyle='dashed')
fig=plt.gcf()
fig.savefig("./lights/scatter4b.jpg",dpi=200)
print("Scatter plots were made *****************************")
###################################################################

#icon, mbase, fabs(magm),fabs(FWHM), fabs(tmax1), fws, fli, rho, tstar, u0/rho, fb, mstar, limb, tim1, tmm, tim2, twing, tshol, error,N);

for i in range(nm): 
    icon, mbase, magm, FWHM, tmax1, fws, ratio=par[i,0], par[i,1], par[i,2], par[i,3], par[i,4], par[i,5], par[i,6]
    rho, tstar, ur, fb, mstar, limb=          par[i,7], par[i,8], par[i,9], par[i,10], par[i,11], par[i,12]
    tim1, tmm, tim2, twing, tshol,error, N=   par[i,13], par[i,14], par[i,15], par[i,16], par[i,17], par[i,18], par[i, 19]
    
    if(error==1):  
        print("observable features:  ", icon, mbase, magm,  FWHM, tmax1,  fws  )
        print("Lensing parameters:  ",  rho,  tstar/rho, ur*rho,  fb, mstar )
        print("extra parameters:  ",   tim1,  tmm, tim2,  twing, tshol, error )
        f2=open("./files/m_{0:d}.dat".format(i),"r")
        nd= sum(1 for line in f2)
        print ("No.  data:  ",  nd)
        if(nd>0):  
            dat=np.zeros((nd,4)) 
            dat=np.loadtxt("./files/m_{0:d}.dat".format(i))
            plt.clf()
            fig= plt.figure(figsize=(8,6)) 
            plt.scatter(dat[:,0], dat[:,1], color= "m", s=1.0)       
            plt.xlabel(r"$time(t_{\star})$",fontsize=18,labelpad=0.1)
            plt.ylabel(r"$\Delta m(\rm{mag})$",fontsize=18,labelpad=0.1)
            plt.xticks(fontsize=16, rotation=0)
            plt.yticks(fontsize=16, rotation=0)
            py.xlim([ -5.0 , 5.0 ])
            plt.gca().invert_yaxis()
            plt.grid("True")
            plt.grid(linestyle='dashed')
            fig=plt.gcf()
            fig.savefig("./lights/lerrorb_{0:d}.jpg".format(i),dpi=200)
            ###########################################################3
            plt.clf()
            fig= plt.figure(figsize=(8,6)) 
            plt.scatter(dat[:,0], dat[:,2], color="m", s=1.0)       
            plt.xlabel(r"$time(t_{\star})$",fontsize=18,labelpad=0.1)
            plt.ylabel(r"$\Delta m(\rm{mag})$",fontsize=18,labelpad=0.1)
            plt.xticks(fontsize=16, rotation=0)
            plt.yticks(fontsize=16, rotation=0)
            py.xlim([ -5.0 , 5.0 ])
            #plt.gca().invert_yaxis()
            plt.yscale('log')
            plt.grid("True")
            plt.grid(linestyle='dashed')
            fig=plt.gcf()
            fig.savefig("./lights/derrorb_{0:d}.jpg".format(i),dpi=200)            
            print("Lightcurves are plotted ************************",   i)
            input("Enter a number ") 


'''
fif=open("./result2c.txt","a+")
#fif.write("888888888888888888888888888888888888888888888888\n")
fif.close()
##########################################################################
#mbase, Deltam, FWHM, Tmax, fws, frac, rho, tstar, ur,  fb,  mstar, limb
print("**************  ONE DECISION TREE ***************")
x=np.zeros(( len(df.tstar) , 6))
y=np.zeros(( len(df.tstar) , 6))
for i in range(len(df.tstar)):  
    x[i,0], x[i,1], x[i,2], x[i,3], x[i,4], x[i,5]= df.mbase[i], df.Deltam[i], df.FWHM[i], df.Tmax[i], df.fws[i], df.frac[i]
    y[i,0], y[i,1], y[i,2], y[i,3], y[i,4], y[i,5]= df.rho[i],   df.tstar[i],  df.ur[i],   df.fb[i],   df.mstar[i], df.limb[i] 

tre=DecisionTreeRegressor(max_depth=13) 
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=0.2, random_state=0)    
tre.fit(xtrain, ytrain)
ypred=tre.predict(xtest);


#######################################################################
'''
mape =np.abs(np.mean(np.abs((ytest-ypred)/ytest) ))
mse= metrics.mean_squared_error( ytest, ypred)  
r2s= metrics.r2_score(ytest, ypred)
rmse=np.sqrt(mse)

print("************* PREDICTED VALUES **********************") 
print("r2s,   mape,    mse,    rmse):        ", r2s,   mape,    mse,    rmse)
print("************* ONE TREE IS MADE ************************")
resu=np.array([r2s,   mape,    mse,    rmse ])
fif=open("./result2c.txt","a+")
np.savetxt(fif,resu.reshape((1, 4)),fmt="ONE_TREE      $%.3f$ &      $%.3f$   &    $%.3f$   &   $%.3f$ ") 
fif.write("=====================================================\n")
fif.close();
'''
#######################################################################

'''
array=np.zeros((len(xtest) , 3))
print (len(xtest), xtest[1][0], xtest[0][1] )
for i in range(len(xtest)): 
    array[i,0]= float(xtest[i][0])
    array[i,1]= float(ytest[i][4])#[i]
    array[i,2]= float(ypred[i][4])#[i]
plt.clf()
plt.scatter(array[:,0], array[:,1],c="m", s=3.0, label="original")
plt.scatter(array[:,0], array[:,2],c="b", s=3.5, label="predicted")
plt.title("y-test and y-predicted data")
plt.xlabel(r"$m_{\rm{base}}$", fontsize=18)
plt.ylabel(r"$m_{\star}$",     fontsize=18)
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.grid(linestyle='dashed')
plt.savefig("Example_tree2.jpg",  dpi=200) 
'''
##################################################################
'''
for i in range(1):
    i=5 
    test2=np.zeros((10,4))
    model = RandomForestRegressor(n_estimators=120, n_jobs=1, random_state=65)
    cv = model_selection.KFold(n_splits=10)
    nkf=0; 
    for traini, testi in cv.split(x):
        xtrain=np.zeros((len(traini),6))    
        ytrain=np.zeros((len(traini) ))
        xtest= np.zeros((len(testi), 6))      
        ytest= np.zeros((len(testi) ))
        for j in range(len(traini)):
            xtrain[j,:] = x[int(traini[j]),:] 
            ytrain[j] = y[int(traini[j]),i]
        for j in range(len(testi)):
            xtest[j,:] = x[int(testi[j]),:]
            ytest[j] = y[int(testi[j]),i]
         
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        mape= np.abs(np.mean(np.abs((ytest-ypred)/ytest) ))
        mse= metrics.mean_squared_error(ytest, ypred)
        r2s=   metrics.r2_score(ytest, ypred)
        rmse=np.sqrt(mse)

        test2[nkf,0], test2[nkf,1], test2[nkf,2], test2[nkf,3]  = r2s,   mape,    mse,    rmse
        print("r2s,   mape,    mse,    rmse :    ",   r2s,   mape,    mse,    rmse)
        
        #fif=open("./result2b.txt","a+")
        #np.savetxt(fif,test2[nkf,:].reshape((1,4)),fmt="KFOLD_i    %.7f     %.7f     %.7f    %.7f") 
        #fif.close();   
        nkf+=1
    ave=np.array([np.mean(test2[:,0]), np.mean(test2[:,1]),   np.mean(test2[:,2]),  np.mean(test2[:,3]) ])
    print(ave)
    fif=open("./result2c.txt","a+")
    np.savetxt(fif,ave.reshape((-1,4)),fmt="AVE_i     $%.3f$ &      $%.3f$   &    $%.3f$  &  $%.3f$ ") 
    fif.write("=====================================================\n\n")
    fif.close();
print("importance:  ", model.feature_importances_ )
'''
################################################################### 
xt=np.zeros((2,6))
for i in range(2): 
    xt[i,0], xt[i,1], xt[i,2],xt[i,3],xt[i,4],xt[i,5]=14.909246044752612,1.040446734231793, 1.656004461881821, 0.7225816, 1.8360354448822076 , 0.4618452798070645
#df.mbase[i], df.Deltam[i], df.FWHM[i], df.Tmax[i], df.fws[i], df.frac[i]


for i in range(1): 
    test2=np.zeros((10,4))
    model = RandomForestRegressor(n_estimators=120, n_jobs=4, random_state=65)
    cv = model_selection.KFold(n_splits=10)
    nkf=0; 
    for traini, testi in cv.split(x):
        xtrain=np.zeros((len(traini),6))    
        ytrain=np.zeros((len(traini),6))
        xtest= np.zeros((len(testi), 6))      
        ytest= np.zeros((len(testi), 6))
        for j in range(len(traini)):
            xtrain[j,:] = x[int(traini[j]),:] 
            ytrain[j,:] = y[int(traini[j]),:]
        for j in range(len(testi)):
            xtest[j,:] = x[int(testi[j]),:]
            ytest[j,:] = y[int(testi[j]),:]
         
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        ypred1 = model.predict(xt)
        print("***ypred1:  ",    ypred1)
        mape= np.abs(np.mean(np.abs((ytest-ypred)/ytest) ))
        mse= metrics.mean_squared_error(ytest, ypred)
        r2s=   metrics.r2_score(ytest, ypred)
        rmse=np.sqrt(mse)

        test2[nkf,0], test2[nkf,1], test2[nkf,2], test2[nkf,3]  = r2s,   mape,    mse,    rmse
        print("r2s,   mape,    mse,    rmse :    ",   r2s,   mape,    mse,    rmse)
        
        #fif=open("./result2b.txt","a+")
        #np.savetxt(fif,test2[nkf,:].reshape((1,4)),fmt="KFOLD_TOT    %.7f     %.7f     %.7f    %.7f") 
        #fif.close();   
        nkf+=1
    ave=np.array([np.mean(test2[:,0]), np.mean(test2[:,1]),   np.mean(test2[:,2]),  np.mean(test2[:,3]) ])
    print (ave)
    fif=open("./result2c.txt","a+")
    np.savetxt(fif,ave.reshape((-1,4)),fmt="AVE_TOT    $%.3f$ &      $%.3f$   &    $%.3f$  &  $%.3f$ ") 
    fif.write("=====================================================\n\n")
    fif.close();
print("importance:  ", model.feature_importances_ )
##################################################################################  

#rep=tree.export_text(tre,feature_names= labell[:5])
#with open("decistion_tree.txt", "w") as fout:
#    fout.write(rep)   
#fig = plt.figure(figsize=(25,20))
#_ =tree.plot_tree(tre, filled=True) ## feature_names=labell[:5],
#fig.savefig("DTC"+ ".jpg", format="jpg", dpi=250, bbox_inches='tight') 
'''
nd=int(30)
err=np.zeros((nd,3))
for j in range(nd):   
    err[j,0]=j+1
    tree=DecisionTreeRegressor(max_depth=j+1)   
    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.2, random_state=0)    
    tree.fit(xtrain, ytrain)
    ypred= tree.predict(xtest)
    err[j,1]= metrics.mean_squared_error(ytrain , tree.predict(xtrain))
    err[j,2]= metrics.mean_squared_error(ytest ,  ypred)
    #print ("Maximum Depth of tree: ",   j+1)
    print ("Error(Training Set):  ",  err[j,1],  "   Error(test set):  ",  err[j,2])
    print ("*********************************************")

###################################################################    
plt.clf()
plt.plot(err[:,0], err[:,1], "r-",  lw=2.0, label=r"$\rm{Training}~\rm{Set}$")
plt.plot(err[:,0], err[:,2], "b--", lw=2.0, label=r"$\rm{Test}~\rm{Set}$")
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$\rm{Maximum}~\rm{Tree}~\rm{Depth}$", fontsize=17)
plt.ylabel(r"$\rm{Mean}~\rm{Squared}~\rm{Error}$", fontsize=17)
plt.legend()
plt.savefig("ErrorDepth2.jpg", dpi=200)
'''
###################################################################    

start_time =time.time()
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
forest_importances = pd.Series(importances, index=labell[:6])

plt.clf()
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_ylabel(r"$\rm{Feature}~\rm{importance}$", fontsize=18)
plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16, rotation=0)
plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'dashed')
fig.tight_layout()
fig.savefig("import2a.jpg", dpi=200)


###################################################################      

start_time = time.time()
result = permutation_importance(model, xtest, ytest, n_repeats=10, random_state=42, n_jobs=1)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
plt.clf()
fig, ax = plt.subplots()
forest_importances = pd.Series(result.importances_mean, index=labell[:6] )
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_ylabel(r"$\rm{Feature}~\rm{importance}$", fontsize=18)
plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'dashed')
plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16, rotation=0)
fig.tight_layout()
fig.savefig("import2b.jpg", dpi=200)


###################################################################
     


