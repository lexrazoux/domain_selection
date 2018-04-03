def inner_error(P,k):
    
    # Import librariesand functions
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    # constructing x and y
    l1 = sum(P[1])  
    l0 = len(P[1])-l1
    x  = P[0]
    y  = l1*[1]+l0*[0]
    
    # initializing error list
    error = []
    
    # splitting in training and test set              
    kf          = KFold(n_splits = k,shuffle=True,random_state=3)    
    for Itr, Its in kf.split(x):    
        x_tr    = [x[i] for i in Itr]
        y_tr    = [y[i] for i in Itr]
        x_ts    = [x[i] for i in Its]
        y_ts    = [y[i] for i in Its]   
        
        # transforming x to vectors
        vectorizer  = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df = .4, sublinear_tf=True,use_idf=True)
        tr_tfidf    = vectorizer.fit_transform(x_tr)
        ts_tfidf    = vectorizer.transform(x_ts)
    
        # Fit classifier  
        classifier = LogisticRegression(class_weight='balanced',random_state=43)
        classifier.fit(tr_tfidf,y_tr)
    
        # Test accuracy
        y_est               =   classifier.predict(ts_tfidf)
        error.append(sum(abs(y_est - y_ts))*1./len(y_est))
    
    # calculating the mean error and its variance over the k-fold evaluations
    mean    = np.mean(error)
    
    return mean     

#%%
def cross_error(P,P_bar):
    
    # Import librariesand functions
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    # constructing training and test sets
    l1    = sum(P[1])  
    l0    = len(P[1])-l1
    x_tr  = P[0]
    y_tr  = l1*[1]+l0*[0]
    
    l1    = sum(P_bar[1])  
    l0    = len(P_bar[1])-l1
    x_ts  = P_bar[0]
    y_ts  = l1*[1]+l0*[0]
    
    # transforming x to vectors
    vectorizer  = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df = .4, sublinear_tf=True,use_idf=True)
    tr_tfidf    = vectorizer.fit_transform(x_tr)
    ts_tfidf    = vectorizer.transform(x_ts)

    # Fit classifier  
    
    classifier = LogisticRegression(class_weight='balanced',random_state=43)
    #classifier = RidgeClassifier(alpha=1,class_weight='balanced',random_state=43)
    classifier.fit(tr_tfidf,y_tr)

    # Test accuracy
    y_est               =   classifier.predict(ts_tfidf)
    error               =   (sum(abs(y_est - y_ts))*1./len(y_est))
    
    return error
