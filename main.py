#%% import corpora and define P, P_bar, Pset and Pset_bar
def make_P():
    def load_data(i):
        import os
        import pandas as pd
        
        directory = os.getcwd()
        
        s = directory+'\\datasets\\set'+str(i)+'.xlsx'
        
        data = pd.read_excel(s)
        doc = data.doc.tolist()
        label_long = data.label.tolist()
        label = [int(l) for l in label_long]
        P = [doc,label]
    
        return P
    
    Pset = []
    for i in range(2,15):           #first set is confidential
        Pset.append(load_data(i))   
    Pset_bar = Pset
    
    return Pset, Pset_bar
[Pset,Pset_bar] = make_P()

#%% cross domain error
import numpy as np
from classify import cross_error
[Pset,Pset_bar] = make_P()

xi_cross    = np.zeros((len(Pset),len(Pset_bar)))
    
for i in range(len(Pset)):
    for j in range(len(Pset_bar)):
        P = Pset[i]
        P_bar = Pset_bar[j]
        
        if P != P_bar:
            xi_cross[i,j] = cross_error(P,P_bar)
    
        print('xi_cross for P' + str(i) + ' and P_bar' + str(j) + ' calculated')

#%% inner domain error
import numpy as np
from classify import inner_error
[Pset,Pset_bar] = make_P()

xi_inner    = np.zeros((len(Pset),len(Pset_bar)))
k           = 10

for i in range(len(Pset)):
    e = inner_error(Pset[i],k)
    for j in range(len(Pset_bar)):
        xi_inner[i,j] = e
    print('xi_inner for P' + str(i))

#%% calculate error when training on all data
import numpy as np
from classify import cross_error
[Pset,Pset_bar] = make_P()

xi_cross_all    = np.zeros((len(Pset),))

for i in range(len(Pset)):
    r = range(len(Pset))
    del r[i]
    
    docs, labels = [], []
    for j in r:
        docs = docs + Pset[j][0]
        labels = labels + Pset[j][1]
    labels, docs = (list(t) for t in zip(*sorted(zip(labels, docs),reverse=True)))
    P_all = [docs,labels]
    P_bar = Pset[i]
    
    xi_cross_all[i] = cross_error(P_all,P_bar)

    print('xi_cross_all for P = P_all' + ' and P_bar' + str(i) + ' calculated')

#%% calculate chi2
from measures import chi2measure
import numpy as np
[Pset,Pset_bar] = make_P()

chi = np.zeros((len(Pset),len(Pset_bar)))

for i in range(len(Pset)):
    for j in range(len(Pset_bar)):
        P = Pset[i]
        P_bar = Pset_bar[j]  
        chi[i,j] = chi2measure(corpusa = P[0], corpusb = P_bar[0], N=1000, lamb=0.05)
        print(j)

#%% calculate MMD
from measures import MMDmeasure
import numpy as np
[Pset,Pset_bar] = make_P()

mmd = np.zeros((len(Pset),len(Pset_bar)))

for i in range(len(Pset)):
    for j in range(i+1,len(Pset_bar)):
        P = Pset[i]
        P_bar = Pset_bar[j] 
        mmd[i,j] = MMDmeasure(corpusa = P[0][:5000], corpusb = P_bar[0][:5000],N=1000)
        print(j)

mmd = mmd + mmd.transpose()

#%% calculate EMD
from measures import emd_dist
import numpy as np
[Pset,Pset_bar] = make_P()

emd = np.zeros((len(Pset),len(Pset_bar)))

for i in range(len(Pset)):
    for j in range(i+1,len(Pset_bar)):
        P = Pset[i]
        P_bar = Pset_bar[j] 
        emd[i,j] = emd_dist(P[0],P_bar[0],1000)
        print(j)
    print('emd')

emd = emd + emd.transpose()

#%% calculate KLD
from measures import kldiv
import numpy as np
[Pset,Pset_bar] = make_P()

kld = np.zeros((len(Pset),len(Pset_bar)))
kld_special = np.zeros((len(Pset),len(Pset_bar)))

for i in range(len(Pset)):
    for j in range(len(Pset_bar)):
        P = Pset[i]
        P_bar = Pset_bar[j]         
        kld[i,j] = kldiv(P[0],P_bar[0],N=1000,lamb=0.00001)   
        print(j)    
    print('kld')
    
#%% save results in pickle
import pickle
pickle.dump([xi_inner, xi_cross, xi_cross_all, chi, mmd, emd, kld], open( "measures_and_accuracies.p", "wb" ))

#%% Plot fig 1
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
import matplotlib.pylab as plt
import math
from classify import cross_error
[Pset,Pset_bar] = make_P()

# regression parameters 
alpha = 0               #regularization strength
loss = 'L1'             #type of loss for constructing the linear source domain selection model parameters
Nsets = 1               #number of sources domains to select
B = []
# load and shape data
[xi_inner, xi_cross, xi_cross_all, chi, mmd, emd, kld] = pickle.load(open("measures_and_accuracies.p", "rb"))

m  = chi.shape[0]                                       #number of datasets
s0 = chi.flatten()                                      #array of chi-squared values (m*m x 1)
s1 = mmd.flatten()                                      #array of MMD values (m*m x 1)
s2 = emd.flatten()                                      #array of EMD values (m*m x 1)
s3 = kld.flatten()                                      #array of KLD values (m*m x 1)
s4 = xi_inner.flatten()                                 #array of inner error values (m*m x 1)
X = normalize(np.stack((s0,s1,s2,s3,s4), axis=1),axis=0)             

xi_cross_hat_list = []
n = int(X.shape[1])                                     #number of measures   
y = xi_cross.flatten()                                  #value to be predicted (m*m x 1)

index_diag = range(0,m**2+1,m+1)                        #the indices of all domain pairs with two same domains
                                                        #the pairs (imdb-imdb) or (amazon-amazon) for example
                                                        #we exclude this domains later on
predicted_min, true_min, error = [], [], []             #initializing

# function to calculate the optimal linear regression model  
def func(beta0,X_train,y_train,alpha):
    if loss == 'L2':
        error = sum((np.dot(X_train,beta0[0:n])+beta0[n]-y_train)**2 + alpha*sum(beta0))
    if loss == 'L1':
        error = sum(abs(np.dot(X_train,beta0[0:n])+beta0[n]-y_train) + alpha*sum(beta0))
    return error


# ----------------------------------------------------------------------------
# make regression model for leave-one-out setup
# ----------------------------------------------------------------------------

for s_val in range(0,m):
    # make train-test split
    index_train = list(set(range(s_val*m,s_val*m+m) \
                 + range(s_val,m*m,m) \
                 + index_diag))                         #these indices represent same domain pairs
                                                        #and the pairs were the target domain is used
                                                        #to train or test on. In other words, each run
                                                        #we exclude all pairs with the new domain that
                                                        #is used as fictive target domain. The fictive
                                                        #target domain is used as domain to test on.

    index_test = range(s_val,m*m,m)                     #the indices represent domain pairs of which 
    index_test.remove(s_val+m*s_val)                    #the testing domain is the target domain
                                                                                  
    X_train = np.delete(X,index_train,0)                #The feature values of our training pairs
    y_train = np.delete(y,index_train,0)                #The labels of our training pairs
    X_test = X[index_test,:]                            #The feature values of our test pairs
    y_test = y[index_test]                              #The labels of our test pairs
    
    # fit linear regression model
    beta_i = minimize(func,x0=np.zeros((n+1)),args=(X_train,y_train,alpha),method='L-BFGS-B',bounds=n*[(0,None)]+[(None,None)],options={'disp': True})
    B.append(beta_i.x)
    
    xi_cross_hat = np.dot(X_test,beta_i.x[0:n])+beta_i.x[n]#the predicted values correspond to the predictions of 
    xi_cross_hat_list.append(xi_cross_hat)                  #the cross domain classification error (xi_cross)
       
    # remove target domain from candidate source domains
    xi_cross_source = np.delete(xi_cross[:,s_val],s_val)#This array states all the cross domain classification
                                                        #errors when training on each dataset, except for the 
                                                        #target domain itself    
    
    # train the model on the Nsets sets with lowest predicted xi_cross and apply on target domain
    index_min = xi_cross_hat.argsort()[:Nsets]          #the source domain that has the lowest Nsets predicted xi_cross
    rset = list(Pset)
    del rset[s_val]
    
    docs, labels, Ps = [], [], []
    for j in list(index_min):
        docs = docs + rset[j][0]
        labels = labels + rset[j][1]
    labels, docs = (list(t) for t in zip(*sorted(zip(labels, docs),reverse=True)))
    Ps = [docs,labels]
    del rset
    
    predicted_min.append(cross_error(Ps, Pset[s_val]))  #the true xi_cross when using the predicted best source domain(s) for training
    true_min.append(np.min(xi_cross_source) )           #the true lowest xi_cross (when the true best source domain is selected)
    error.append(xi_cross_source)                       #the average xi_cross when training on a random domain

# model selection relative error (xi_cross when using the model - xi_cross when training on true best singel source domain)
relative_error = []
for i in range(0,len(predicted_min)):
    relative_error.append(predicted_min[i]-true_min[i])

# random domain selection realative error distribution to the plot
random_selection_values = []
for i in range(0,m):
    random_selection_values = random_selection_values + list(error[i]-true_min[i])

B = np.mean(B,axis=0)

# ----------------------------------------------------------------------------
# plotting results
# ----------------------------------------------------------------------------

# plot parameters
spacing = .45                                           #spacing parameter (bar width)
delta   = 0.05                                          #bin width of histogram
xmax    = 0.4                                           #maximum bin

bins    = np.linspace(0,xmax,math.ceil(xmax/delta)+1)   #bin edges
offset  = delta*(.5-spacing)                            #need to center the bars in cluster

plt.figure(figsize=(6,4))
ax = plt.axes()


# plotting the relative error distribution for random domain selection
hist = np.histogram(random_selection_values,bins=bins,normed=True)
plt.bar(hist[1][:-1] + delta/2, hist[0]*delta ,delta*spacing ,align='edge', color=(.6,.6,.6), edgecolor='black')

# plotting the relative error distribution for Chi2, MMD, EMD & KLD selection model
hist = np.histogram(relative_error,bins=bins,normed=True)
plt.bar(hist[1][:-1]+offset , hist[0]*delta, delta*spacing, align='edge', color=(.9,.9,.9), edgecolor='black')



plt.title('Relative error distribution - selecting 1 source domain')
plt.xlim([0,xmax])
plt.xticks(bins)
plt.yticks(np.arange(0,max(plt.yticks()[0]),0.1))
plt.xlabel(r'$\xi ( \hat{\mathbb{P}},\bar{\mathbb{P}} ) - \xi ( \mathbb{P}^{\star},\bar{\mathbb{P}} )$')
plt.ylabel('probability')
plt.legend(('Random domain selection','Domain selection by CMEK'))
ax.yaxis.grid(linestyle=':')
ax.set_axisbelow(True)
plt.tight_layout()

plt.savefig('paper_fig3.png',transparent=True)

#%% Table 1
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
import matplotlib.pylab as plt
import math
from classify import cross_error
from astropy.table import Table

table1 = np.zeros((7,3))
significance = np.zeros((6,13))

# regression parameters 
alpha = 0               #regularization strength
loss = 'L1'             #type of loss for constructing the linear source domain selection model parameters
Nsets = 1               #number of sources domains to select

# load and shape data
[xi_inner, xi_cross, xi_cross_all, chi, mmd, emd, kld] = pickle.load(open("measures_and_accuracies.p", "rb"))

m  = chi.shape[0]                                       #number of datasets
s0 = chi.flatten()                                      #array of chi-squared values (m*m x 1)
s1 = mmd.flatten()                                      #array of MMD values (m*m x 1)
s2 = emd.flatten()                                      #array of EMD values (m*m x 1)
s3 = kld.flatten()                                      #array of KLD values (m*m x 1)
s4 = xi_inner.flatten()                                 #array of inner error values (m*m x 1)

X1 = normalize(np.stack((s0,s1,s2,s3,s4), axis=1),axis=0)                 
X2 = normalize(s0,axis=1).transpose() 
X3 = normalize(s1,axis=1).transpose()
X4 = normalize(s2,axis=1).transpose() 
X5 = normalize(s3,axis=1).transpose()

Xset = [X1,X2,X3,X4,X5] 

z = 0
for X in Xset:  
    z = z +1          
    n = int(X.shape[1])                                     #number of measures   
    y = xi_cross.flatten()                                  #value to be predicted (m*m x 1)
    
    index_diag = range(0,m**2+1,m+1)                        #the indices of all domain pairs with two same domains
                                                            #the pairs (imdb-imdb) or (amazon-amazon) for example
                                                            #we exclude this domains later on
    predicted_min, true_min, error = [], [], []             #initializing
    average_error, fifth_true      = [], []                 #initializing
    
    # function to calculate the optimal linear regression model  
    def func(beta0,X_train,y_train,alpha):
        if loss == 'L2':
            error = sum((np.dot(X_train,beta0[0:n])+beta0[n]-y_train)**2 + alpha*sum(beta0))
        if loss == 'L1':
            error = sum(abs(np.dot(X_train,beta0[0:n])+beta0[n]-y_train) + alpha*sum(beta0))
        return error
    
    
    # ----------------------------------------------------------------------------
    # make regression model for leave-one-out setup
    # ----------------------------------------------------------------------------
    
    for s_val in range(0,m):
        # make train-test split
        index_train = list(set(range(s_val*m,s_val*m+m) \
                     + range(s_val,m*m,m) \
                     + index_diag))                         #these indices represent same domain pairs
                                                            #and the pairs were the target domain is used
                                                            #to train or test on. In other words, each run
                                                            #we exclude all pairs with the new domain that
                                                            #is used as fictive target domain. The fictive
                                                            #target domain is used as domain to test on.
    
        index_test = range(s_val,m*m,m)                     #the indices represent domain pairs of which 
        index_test.remove(s_val+m*s_val)                    #the testing domain is the target domain
                                                                                      
        X_train = np.delete(X,index_train,0)                #The feature values of our training pairs
        y_train = np.delete(y,index_train,0)                #The labels of our training pairs
        X_test = X[index_test,:]                            #The feature values of our test pairs
        y_test = y[index_test]                              #The labels of our test pairs
        
        # fit linear regression model
        beta_i = minimize(func,x0=np.zeros((n+1)),args=(X_train,y_train,alpha),method='L-BFGS-B',bounds=n*[(0,None)]+[(None,None)],options={'disp': True})
        if z == 1:
            print(beta_i.x)
        xi_cross_hat = np.dot(X_test,beta_i.x[0:n])+beta_i.x[n]#the predicted values correspond to the predictions of 
                                                               #the cross domain classification error (xi_cross)
           
        # remove target domain from candidate source domains
        xi_cross_source = np.delete(xi_cross[:,s_val],s_val)#This array states all the cross domain classification
                                                            #errors when training on each dataset, except for the 
                                                            #target domain itself    
        
        # train the model on the Nsets sets with lowest predicted xi_cross and apply on target domain
        index_min = xi_cross_hat.argsort()[:Nsets]          #the source domain that has the lowest Nsets predicted xi_cross
        rset = list(Pset)
        del rset[s_val]
        
        docs, labels, Ps = [], [], []
        for j in list(index_min):
            docs = docs + rset[j][0]
            labels = labels + rset[j][1]
        labels, docs = (list(t) for t in zip(*sorted(zip(labels, docs),reverse=True)))
        Ps = [docs,labels]
        del rset
        
        predicted_min.append(cross_error(Ps, Pset[s_val]))  #the true xi_cross when using the predicted best source domain(s) for training
        true_min.append(np.min(xi_cross_source) )           #the true lowest xi_cross (when the true best source domain is selected)
        fifth_true.append(sorted(xi_cross_source, reverse=True)[4]) #value of fifth highest true error
        error.append(xi_cross_source)                       #the average xi_cross when training on a random domain
        average_error.append(np.mean(xi_cross_source))      #xi_cross when training on a each source domain individually
    
    significance[z-1,:] = predicted_min  
    n_optimal_predictions = np.sum(np.array(true_min)==np.array(predicted_min))*1./m
    n_5worst_predictions = np.sum(np.array(fifth_true)<=np.array(predicted_min))*1./m
                                 
    table1[z,:] = np.array([n_optimal_predictions,n_5worst_predictions,np.mean(predicted_min)])
table1[0,:] = np.array([1./(m-1), 5./(m-1), np.mean(average_error)])
xi_cross.sort(axis=0)
xi_opt = np.mean(xi_cross[1,:])
table1 = table1[table1[:-1,2].argsort()]                 #sort table
table1 = np.concatenate((np.reshape(np.array([1,0,xi_opt]),(1,3)),table1),axis=0)
table1_subset = table1[[0,4,6],:]                                                         #only display optimal, CMEK & random
np.savetxt('table1.txt',table1_subset,fmt='%.3f') 
                                                    
arr = {'Selection method': ('Optimal','EMD','Chi2','KLD','CMEK','MMD','Random'),
       'Probability selecting true best domain' : table1[:,0],
       'Probability selecting one of five worst domains' : table1[:,1],
       'Mean cross domain classification rate' : table1[:,2]}
Table(arr, names=('Selection method', 'Probability selecting true best domain', 'Probability selecting one of five worst domains','Mean cross domain classification rate'))

pickle.dump([significance], open( "significance.p", "wb" )) #store data for significante calculations

#%% fig 2
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
import matplotlib.pylab as plt
import math
from classify import cross_error
from astropy.table import Table


# regression parameters 
alpha = 0               #regularization strength
loss = 'L1'             #type of loss for constructing the linear source domain selection model parameters


# load and shape data
[xi_inner, xi_cross, xi_cross_all, chi, mmd, emd, kld] = pickle.load(open("measures_and_accuracies.p", "rb"))

m  = chi.shape[0]                                       #number of datasets
s0 = chi.flatten()                                      #array of chi-squared values (m*m x 1)
s1 = mmd.flatten()                                      #array of MMD values (m*m x 1)
s2 = emd.flatten()                                      #array of EMD values (m*m x 1)
s3 = kld.flatten()                                      #array of KLD values (m*m x 1)
s4 = xi_inner.flatten()   
X1 = normalize(np.stack((s0,s1,s2,s3,s4), axis=1),axis=0)                 
X2 = normalize(s0,axis=1).transpose() 
X3 = normalize(s1,axis=1).transpose() 
X4 = normalize(s2,axis=1).transpose() 
X5 = normalize(s3,axis=1).transpose() 
Xset = [X1,X2,X3,X4,X5] 

Q = np.zeros((m-1,len(Xset)))         #initializing (answer array)
Qraw = np.zeros((m-1,len(Xset),m))         #initializing (answer array)
k = -1
for Nsets in range(1,m):              #number of sources domains to select
    k = k + 1
    
    z = -1
    for X in Xset:  
        z = z +1          
        n = int(X.shape[1])                                     #number of measures   
        y = xi_cross.flatten()                                  #value to be predicted (m*m x 1)
        
        index_diag = range(0,m**2+1,m+1)                        #the indices of all domain pairs with two same domains
                                                                #the pairs (imdb-imdb) or (amazon-amazon) for example
                                                                #we exclude this domains later on
        predicted_min, true_min, error = [], [], []             #initializing
        average_error                  = []                     #initializing
        
        # function to calculate the optimal linear regression model  
        def func(beta0,X_train,y_train,alpha):
            if loss == 'L2':
                error = sum((np.dot(X_train,beta0[0:n])+beta0[n]-y_train)**2 + alpha*sum(beta0))
            if loss == 'L1':
                error = sum(abs(np.dot(X_train,beta0[0:n])+beta0[n]-y_train) + alpha*sum(beta0))
            return error
        
        
        # ----------------------------------------------------------------------------
        # make regression model for leave-one-out setup
        # ----------------------------------------------------------------------------
        
        for s_val in range(0,m):
            # make train-test split
            index_train = list(set(range(s_val*m,s_val*m+m) \
                         + range(s_val,m*m,m) \
                         + index_diag))                         #these indices represent same domain pairs
                                                                #and the pairs were the target domain is used
                                                                #to train or test on. In other words, each run
                                                                #we exclude all pairs with the new domain that
                                                                #is used as fictive target domain. The fictive
                                                                #target domain is used as domain to test on.
        
            index_test = range(s_val,m*m,m)                     #the indices represent domain pairs of which 
            index_test.remove(s_val+m*s_val)                    #the testing domain is the target domain
                                                                                          
            X_train = np.delete(X,index_train,0)                #The feature values of our training pairs
            y_train = np.delete(y,index_train,0)                #The labels of our training pairs
            X_test = X[index_test,:]                            #The feature values of our test pairs
            y_test = y[index_test]                              #The labels of our test pairs
            
            # fit linear regression model
            beta_i = minimize(func,x0=np.zeros((n+1)),args=(X_train,y_train,alpha),method='L-BFGS-B',bounds=n*[(0,None)]+[(None,None)],options={'disp': True})
            
            xi_cross_hat = np.dot(X_test,beta_i.x[0:n])+beta_i.x[n]#the predicted values correspond to the predictions of 
                                                                   #the cross domain classification error (xi_cross)
               
            # remove target domain from candidate source domains
            xi_cross_source = np.delete(xi_cross[:,s_val],s_val)#This array states all the cross domain classification
                                                                #errors when training on each dataset, except for the 
                                                                #target domain itself    
            
            # train the model on the Nsets sets with lowest predicted xi_cross and apply on target domain
            index_min = xi_cross_hat.argsort()[:Nsets]          #the source domain that has the lowest Nsets predicted xi_cross
            rset = list(Pset)
            del rset[s_val]
            
            docs, labels, Ps = [], [], []
            for j in list(index_min):
                docs = docs + rset[j][0]
                labels = labels + rset[j][1]
            labels, docs = (list(t) for t in zip(*sorted(zip(labels, docs),reverse=True)))
            Ps = [docs,labels]
            del rset
            
            predicted_min.append(cross_error(Ps, Pset[s_val]))  #the true xi_cross when using the predicted best source domain(s) for training
            true_min.append(np.min(xi_cross_source) )           #the true lowest xi_cross (when the true best source domain is selected)
            error.append(xi_cross_source)                       #the average xi_cross when training on a random domain
            average_error.append(np.mean(xi_cross_source))      #xi_cross when training on a each source domain individually
        
        Q[k,z]=np.mean(predicted_min)
        Qraw[k,z,:]=np.array(predicted_min)
        
#from significance_ttest_all import avg_Nset_error
#XI = avg_Nset_error()  #run to get cross domain classification error for Nsets = {1,2,...,13} for random domain selection.
#pickle.dump([XI], open( "XI.p", "wb" ))

#import pickle
pickle.dump([Q, Qraw], open( "backup.p", "wb" ))
[Q, Qraw] = pickle.load(open("backup.p", "rb"))
[XI] = pickle.load(open("XI.p", "rb"))
#
# ----------------------------------------------------------------------------
# Plotting results
# ----------------------------------------------------------------------------

plt.figure(figsize=(6,4))

plt.plot(range(1,m),Q[:,0],linestyle="-",color=(0,0,0),zorder=10)
#plt.plot(range(1,m),Q[:,1],linestyle="-",color=(0,0,0))
#plt.plot(range(1,m),Q[:,2],linestyle="-.",color=(0,0,0))
plt.plot(range(1,m),np.mean(XI,axis=1),linestyle="--",color=(0,0,0))
plt.plot(range(1,m),(m-1)*[np.mean(xi_cross_all)],linestyle=":",color=(0,0,0),zorder=1)

plt.title(r'Classification error selecting $n$ source domains')
plt.xlim([0.5,m-0.5])
plt.xticks(range(1,14,1))
plt.yticks(np.arange(min(plt.yticks()[0]),0.4,0.02))
plt.xlabel(r'$n$')
plt.ylabel(r'$ \xi(\hat{\mathcal{P}}, \bar{ \mathbb{P}}  ) $')
plt.legend(('CMEK selection','random selection','training on all domains'))

plt.tight_layout()
plt.savefig('paper_fig4.png', transparent=True)


