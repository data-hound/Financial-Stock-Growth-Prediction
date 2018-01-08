def readtrain():
    import csv
    import numpy as np
    filename = 'train.csv'
    raw_data = open(filename, 'rb')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)

    #separating numeric data from non-numeric data
    lab=x[0]
    x=x[1:len(x)]

    data = np.array(x).astype('float')#converting numeric data to numpy array

    T,X,Y=np.hsplit(data,[1,101])#separating time and Y field from Attributes field

    return X,T,Y,lab

def readtest():
    import csv
    import numpy as np
    filename = 'test.csv'
    raw_data = open(filename, 'rb')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)

    #separating numeric data from non-numeric data
    lab=x[0]
    x=x[1:len(x)]

    data = np.array(x).astype('float')#converting numeric data to float

    T,X=np.hsplit(data,[1])

    return T,X,lab

def classify():
    X_raw1=[]
    X_raw2=[]

    #getting data
    import numpy as np
    X_raw,T_raw,Y_raw,lab=readtrain()
    X_raw1=np.append(T_raw,X_raw,axis=1)

    T_rawtest,X_rawtest,labtest=readtest()
    X_raw2=np.append(T_rawtest,X_rawtest,axis=1)

    #reshaping Y
    Y_train=np.reshape(Y_raw,3000)

    X_total=np.append(X_raw1,X_raw2,axis=0)


    #Applying PCA
    from sklearn.decomposition import PCA
    pca=PCA(n_components=15)
    pca.fit(X_total)
    X_train=pca.transform(X_raw1)
    X_test=pca.transform(X_raw2)



    #Ensemble of SVM
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    clf=AdaBoostClassifier(n_estimators=1000)
    clf.fit(X_train,Y_train)
    pred=clf.predict(X_test)

    return pred,T_rawtest,lab

def writedata():
    Y,T,lab=classify()
    T_=[]
    

    T_=[item for sublist in T for item in sublist]

    print T_#[item for sublist in T for item in sublist]

    t=zip(T_,Y)
    #print t

    import csv
    with open('predict.csv','wb') as fp:
        k=csv.writer(fp,delimiter=',')
        k.writerow(['Time','Y'])
        k.writerows(t)
    
