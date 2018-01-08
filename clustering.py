def readdata():
    import csv
    import numpy as np
    filename = 'train.csv'
    raw_data = open(filename, 'rb')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)

    #separating numeric data from non-numeric data
    lab=x[0]
    x=x[1:len(x)]
    
    data = np.array(x).astype('float')#converting numeric data to float
    

    T,X,Y=np.hsplit(data,[1,101])

    

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

    data = np.array(x).astype('float')

    T,X=np.hsplit(data,[1])

    return T,X,lab

def assetclstr():
    X1,T1,Y1,lab1=readdata()
    T2,X2,lab2=readtest()

    X=np.append(X1,X2,axis=0)

    feats=lab[1:101]
    print feats

    from sklearn.cluster import KMeans
    import numpy as np

    kmeans=KMeans(n_clusters=15,random_state=100,max_iter=1000).fit(np.transpose(X))
    clust= kmeans.labels_

    

    return feats,clust

def writedata1():
    feats,clust=assetclstr()

    import csv

    c=zip(feats,clust)
    

    with open('assetcluster.csv','wb') as fp:
        k=csv.writer(fp,delimiter=',')
        k.writerow(['Asset','Cluster'])
        k.writerows(c)


    
