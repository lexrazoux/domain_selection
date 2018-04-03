#%% function for removing punctuation
def remove_punct(corpus):
    stripped_corpus = []
    i = 0
    import string
    for mystring in corpus:
        try:
            stripped_corpus.append(str(mystring).translate(None, string.punctuation))
        except:
            i=i+1
    return stripped_corpus


#%% count the frequencies
def count_freq(corpus):
    import nltk
    voc = []
    for document in corpus:
        words = document.split()
        for word in words:
            voc.append(word.lower())

    frequencies = nltk.FreqDist(voc)
    
    words, freq = zip(*frequencies.most_common())
    return words, freq


#%% select words for measure
def get_eval_voc(wordsa,wordsb,N):
    n = 0
    na = 0
    nb = 0
    eval_voc = []
    
    la = len(wordsa)
    lb = len(wordsb)

    while n < min(N,len(list(set(wordsa).union(wordsb)))):
        
        if (na <= nb or nb >= lb) and na<la:
            if wordsa[na] not in eval_voc:
                eval_voc.append(wordsa[na])
                n = n + 1
            na = na + 1
            
        else:
            if wordsb[nb] not in eval_voc:
                eval_voc.append(wordsb[nb])
                n = n + 1
            nb = nb + 1

    
    return eval_voc, n


#%% Chi2
def chi2measure(corpusa, corpusb, N, lamb):
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy
    from scipy.special import kl_div
    from scipy.stats import entropy
    
    vectorizer = CountVectorizer(ngram_range=(1, 2),max_features=N)
    vectorizer.fit(corpusa+corpusb)
    
    X1 = vectorizer.transform(corpusa)
    X2 = vectorizer.transform(corpusb)
    
    x1 = np.asarray(scipy.sparse.csr_matrix.todense(X1))
    x2 = np.asarray(scipy.sparse.csr_matrix.todense(X2))
    n = max(np.max(x1),np.max(x2))
    bins = np.arange(0,n,1)
    
    si = []
    for i in range(0,x1.shape[1]):
        h1 = np.histogram(x1[:,i],bins=bins,normed=True)
        h2 = np.histogram(x2[:,i],bins=bins,normed=True)
        px1 = h1[0] + np.full(h1[0].shape, lamb,dtype='float64')
        px2 = h2[0] + np.full(h2[0].shape, lamb,dtype='float64')
        q = px1-px2
        si.append(sum(np.multiply(q,q)/px2))    
    s = sum(si)
                    
    return s

#%% calculate MMD
def MMDmeasure(corpusa,corpusb,N):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import scipy
    from mmdxx import mmd
    import random
    random.seed(a=3)
    
    if len(corpusa) > len(corpusb):
        I = random.sample(range(0, len(corpusa)), len(corpusb))
        corpusz = list(corpusa[z] for z in I)
        corpusa = corpusz
    else:
        I = random.sample(range(0, len(corpusb)), len(corpusa))
        corpusz = list(corpusb[z] for z in I)
        corpusb = corpusz    

    vectorizer = CountVectorizer(ngram_range=(1, 2),max_features=N)
    vectorizer.fit(corpusa+corpusb)
    
    X1 = vectorizer.transform(corpusa)
    X2 = vectorizer.transform(corpusb)
    x1 = np.asarray(scipy.sparse.csr_matrix.todense(X1))
    x2 = np.asarray(scipy.sparse.csr_matrix.todense(X2))
    
    if x1.size <= x2.size:
        x2 = x2[:x1.shape[0],:]
    else:
        x1 = x1[:x2.shape[0],:]
    
    [sigma, value] = mmd(x1, x2, sigma=None, verbose=False)

    return value


#%%
def emd_dist(corpusa, corpusb, N):
    from pyemd import emd
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy
    import random
    random.seed(a=3)
    np.random.seed(seed=3)
          
    corpusa.sort(key = lambda s: -len(s))
    corpusb.sort(key = lambda s: -len(s))
    
    na = [len(doc.split()) for doc in corpusa]
    nb = [len(doc.split()) for doc in corpusb]
    
    la = int(round(np.mean([len(doc.split()) for doc in corpusa])))
    lb = int(round(np.mean([len(doc.split()) for doc in corpusb])))
    

    corpus_new = []
    if la<lb:           
        nb_new = np.random.choice(na,size=len(nb),replace=True).tolist()
        nb_new.sort(key = lambda s: -s) 
        
        for i in range(len(corpusb)):
            doc = corpusb[i]
            doc_length = len(doc.split())
            new_length = nb_new[i]
            
            if doc_length > new_length:
                word_list = doc.split()[-new_length:]
                corpus_new.append(' '.join(word for word in word_list))
            else:
                corpus_new.append(doc)
        corpusb = corpus_new
        
    else:
        na_new = np.random.choice(nb,size=len(na),replace=True).tolist()
        na_new.sort(key = lambda s: -s) 
        
        for i in range(len(corpusa)):
            doc = corpusa[i]
            doc_length = len(doc.split())
            new_length = na_new[i]
            
            if doc_length > new_length:
                word_list = doc.split()[-new_length:]
                corpus_new.append(' '.join(word for word in word_list))
            else:
                corpus_new.append(doc)
        corpusa = corpus_new
            
    vectorizer = CountVectorizer(ngram_range=(1, 2),max_features=N)
    vectorizer.fit(corpusa+corpusb)
    
    X1 = vectorizer.transform(corpusa)
    X2 = vectorizer.transform(corpusb)
    x1 = np.asarray(scipy.sparse.csr_matrix.todense(X1))
    x2 = np.asarray(scipy.sparse.csr_matrix.todense(X2))
    n = max(np.max(x1),np.max(x2))
    bins = np.arange(0,n,1)       
      
    d = np.zeros((n,n))
    d[0,:] = range(0,n)
    for i in range(1,n):
        d[i,:]=d[i-1,:]-1
    d = abs(d)  
    
    s = []
    for i in range(0,x1.shape[1]):
        h1 = np.histogram(x1[:,i],bins=bins,normed=True)
        h2 = np.histogram(x2[:,i],bins=bins,normed=True)
        s.append(emd(h1[0],h2[0],d))
        
    z = sum(s)
    return z

#%% Kullback-Leibler divergence
def kldiv(corpusa, corpusb, N, lamb):
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy
    from scipy.special import kl_div
    from scipy.stats import entropy
    
    vectorizer = CountVectorizer(ngram_range=(1, 2),max_features=N)
    vectorizer.fit(corpusa+corpusb)
    
    X1 = vectorizer.transform(corpusa)
    X2 = vectorizer.transform(corpusb)
    
    x1 = np.asarray(scipy.sparse.csr_matrix.todense(X1))
    x2 = np.asarray(scipy.sparse.csr_matrix.todense(X2))
    n = max(np.max(x1),np.max(x2))
    bins = np.arange(0,n,1)
    
    si = []
    for i in range(0,x1.shape[1]):
        h1 = np.histogram(x1[:,i],bins=bins,normed=True)
        h2 = np.histogram(x2[:,i],bins=bins,normed=True)
        px1 = h1[0] + np.full(h1[0].shape, lamb,dtype='float64')
        px2 = h2[0] + np.full(h2[0].shape, lamb,dtype='float64')
        si.append(entropy(px1,px2))      
    s = sum(si)
    
    return s