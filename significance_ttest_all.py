#%% significance t-test
import numpy as np
import scipy.stats

def sigp(X1,X2):
    d = X2-X1
    [stat, p_normal] = scipy.stats.normaltest(d)
    
    n = len(d)
    Xbar = np.mean(d)
    s = np.std(d)
    s = 635.4
    df = n-1
    
    SE = s/np.sqrt(n)
    mu = 0
    t = (Xbar - mu)/SE
    
    p_value = scipy.stats.t.sf(np.abs(t), df)
    
    return p_normal, p_value

#%% results for 1 domain selection
import pickle
[significance] = pickle.load(open("significance.p", "rb"))
[xi_inner, xi_cross, xi_cross_all, chi, mmd, emd, kld] = pickle.load(open("measures_and_accuracies.p", "rb"))
[Q, Qraw] = pickle.load(open("backup.p", "rb"))

n = xi_cross.shape[0]
xi_cross_sorted = 1*xi_cross
for i in range(0,n):
    xi_cross_sorted[:,i].sort()
random_error = np.mean(xi_cross_sorted[1:,:],axis=0)

sets = np.concatenate((np.reshape(random_error,(1,n)),significance[:-1,:]),axis=0)
m = sets.shape[0]

P_normal = np.zeros((m,m))
P_values = np.zeros((m,m))

for i in range(m):
    for j in range(m):
        if i != j:
            (P_normal[i,j],P_values[i,j]) = sigp(sets[i],sets[j])

#%% significance of results for P selecting best domain
import scipy.stats
n = 13
p = 1./(n-1)  #probability of random selection selecting the best domain
cdf = 0
for i in np.arange(1,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more best domains is " + str(cdf))

p = 0.308  #probability of EMD selection selecting the best domain
cdf = 0
for i in np.arange(1,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more best domains is " + str(cdf))
    
p = 0.308  #probability of KLD selection selecting the best domain
cdf = 0
for i in np.arange(1,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more best domains is " + str(cdf))
            
p = 0.385  #probability of CMEK selection selecting the best domain
cdf = 0
for i in np.arange(1,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more best domains is " + str(cdf))
    
p = 0.0769  #probability of MMD selection selecting the best domain
cdf = 0
for i in np.arange(1,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more best domains is " + str(cdf))

p = 0.154  #probability of MMD selection selecting the best domain
cdf = 0
for i in np.arange(1,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more best domains is " + str(cdf))
    
#%% significance of results for P not selecting one of 5 worst domains
import scipy.stats
n = 13
p = 7./(n-1)  #probability of random selection not selecting one of 5 worst domains
cdf = 0
for i in np.arange(0,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more not one of 5 worst domains is " + str(cdf))

p = 13./13-0.0000001  #probability of EMD selection not selecting one of 5 worst domains
cdf = 0
for i in np.arange(0,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more not one of 5 worst domains is " + str(cdf))
    
p = 12./13 #probability of KLD selection not selecting one of 5 worst domains
cdf = 0
for i in np.arange(0,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more not one of 5 worst domains is " + str(cdf))
    
p = 11./13  #probability of CMEK selection not selecting one of 5 worst domains
cdf = 0
for i in np.arange(0,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more not one of 5 worst domains is " + str(cdf))
    
p = 11./13  #probability of MMD selection not selecting one of 5 worst domains
cdf = 0
for i in np.arange(0,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more not one of 5 worst domains is " + str(cdf))
    
p = 9./13  #probability of Chi2 selection not selecting one of 5 worst domains
cdf = 0
for i in np.arange(0,n+1,1)[::-1]:
    pi = scipy.stats.binom.pmf(i,n,p)
    cdf = cdf + pi
    print("probability of selecting " + str(i) + " or more not one of 5 worst domains is " + str(cdf))
    
    
#%% training on multiple domains: significance assessment for comparison within defined number of domains
import pickle 
[XI] = pickle.load(open("XI.p", "rb"))
[Q, Qraw] = pickle.load(open("backup.p", "rb"))

nd =5    #number of domains evaluated -1
m = 13
sets = np.concatenate((np.reshape(XI,(m-1,1,m)),Qraw),axis=1)
P_normal = np.zeros((4,4))
P_values = np.zeros((4,4))
for i in range(0,4):
    for j in range(0,4):
        if i != j:
            (P_normal[i,j],P_values[i,j]) = sigp(sets[nd,i,:],sets[nd,j,:])


#%% training on multiple domains: significance assessment for comparison over all defined number of domains
import pickle 
[XI] = pickle.load(open("XI.p", "rb"))
[Q, Qraw] = pickle.load(open("backup.p", "rb"))
m = 13
XI_avg = np.mean(XI,axis=1)
sets = np.concatenate((np.reshape(XI_avg,(m-1,1)),Q),axis=1)

P_normal = np.zeros((4,4))
P_values = np.zeros((4,4))
for i in range(0,4):
    for j in range(0,4):
        if i != j:
            (P_normal[i,j],P_values[i,j]) = sigp(sets[:-1,i],sets[:-1,j])


#%% comparing model with training on all data
import pickle
[significance] = pickle.load(open("significance.p", "rb"))
[xi_inner, xi_cross, xi_cross_all, chi, mmd, emd, kld] = pickle.load(open("measures_and_accuracies.p", "rb"))
[Q, Qraw] = pickle.load(open("backup.p", "rb"))

P_normal = np.zeros((3,12))
P_values = np.zeros((3,12))

nd = 0 #number of domains evaluated -1
for nd in range(0,12):
    for k in range(0,3):
        (P_normal[k,nd], P_values[k,nd]) = sigp(Qraw[nd,k,:],xi_cross_all)          
            
#%% calculate the xi_error for nd domains for random domain selection
def avg_Nset_error():
    import numpy as np
    import pickle
    from classify import cross_error
    import random
    random.seed(a=3)
    
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
    
    iterations = [40,40,20,20,20,20,10,10,10,5,5,5]
    
    m  = len(xi_cross_all)
    XI = np.zeros((m-1,m))
    k = -1
    for Nsets in range(1,m):            #iteration for Nsets to select
        k = k + 1       
        for s_val in range(0,m):        #iteration for s_val as target domain
            runs = []
            for w in range(0,iterations[k]):
                target_domain = Pset_bar[s_val]
                source_domains = Pset[:]
                del source_domains[s_val]
                
                I = random.sample(range(0,12),Nsets)
                
                training_docs = []
                training_labels = []
                for i in I:
                    training_docs = training_docs + source_domains[i][0]
                    training_labels = training_labels + source_domains[i][1]
                    
                label_doc = zip(training_labels,training_docs)
                label_doc.sort(reverse=True)
                labels,docs = zip(*label_doc)
                training_domains_sorted = [list(docs),list(labels)]
                runs.append(cross_error(training_domains_sorted,target_domain))
                print("iteration = "+str(w))
            XI[k,s_val] = np.mean(runs)
            print("number of sets = " + str(Nsets)+ ", target domain = "+ str(s_val))
    
    pickle.dump([XI], open( "XI.p", "wb" ))

