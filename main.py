import metis
import scipy
from scipy.sparse import csr_matrix
import networkx as nx
from lxml import etree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
# import loadXES
from time import time
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn import decomposition
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import scipy.sparse
import collections
from scipy.sparse.csgraph import connected_components
from scipy.sparse import spmatrix


def get_sentences_XES(filename):
  texts = []

  tree = etree.parse(filename)
  root= tree.getroot()
  for element in root.iter():
      #print(element.tag)
      if('}' in element.tag):
          tag= element.tag.split('}')[1]
      else: tag= element.tag
      if(tag== "trace"):
          wordslist = []
          for childelement in element.iterchildren():
              if('}' in childelement.tag):
                  ctag= childelement.tag.split('}')[1]
              else: ctag= childelement.tag
              if (ctag=="event"):
                  for grandchildelement in childelement.iterchildren():
                      if(grandchildelement.get('key')=='concept:name'):
                          event_name=grandchildelement.get('value')
                      #    print(event_name)
                          wordslist.append(event_name.replace(' ',''))
          texts.append(' '.join(wordslist))
  return texts


def getVariants(logname):
    texts = get_sentences_XES(logname)
    print("len(trace)：", len(texts))
    new_texts = []
    for i in texts:
        if i not in new_texts:
            new_texts.append(i)
    print("len(variant)：", len(new_texts))
    dic = collections.OrderedDict((key, []) for key in tuple(new_texts))
    for i in range(len(texts)):
        dic[texts[i]].append(i)
    return dic

def direct_relation(trace,traceDS):
    """
    obtain DS.
    """
    for i in range(len(trace) - 1):
        name = '%s,%s' % (trace[i], trace[i + 1])
        traceDS.add(name)
    return traceDS


def getVariants1(logname):

    texts = get_sentences_XES(logname)
    print("len(trace)：", len(texts))
    new_texts = []
    for i in texts:
        activities = i.split()  # list
        act = set(activities)
        if act not in new_texts:
            new_texts.append(act)
    print("len(variant)：", len(new_texts))
    new_texts = [' '.join(list(item)) for item in new_texts]
    dic = dict((key, []) for key in tuple(new_texts))
    for i in range(len(texts)):
        act_i = set(activities)
        key1 = ' '.join(list(act_i))
        dic[key1].append(i)
    return dic


def getKNN(logname, name, kInit=100, pcaBool=False):


    dic = getVariants(logname)
    texts = list(dic)
    vectorizer = CountVectorizer(ngram_range=(
        1, 3))
    X = vectorizer.fit_transform(texts)
    binair = (X > 0).astype(int)

    pca = decomposition.TruncatedSVD(n_components=20)
    pca.fit(binair)
    Y = pca.transform(binair)

    if pcaBool:
        dataset = Y
    else:
        dataset = binair





    connect = False
    K = kInit
    while (connect == False):
        print("K:", K)
        nbrs = NearestNeighbors(n_neighbors=K).fit(dataset)
        distances, indices = nbrs.kneighbors(X)

        print(indices)

        A = nbrs.kneighbors_graph(X=dataset, n_neighbors=K, mode='distance')

        Asp = scipy.sparse.csr_matrix(A.toarray())
        A_coo = Asp.tocoo()

        rows, cols, vals = A_coo.row, A_coo.col, A_coo.data
        vals = 1.0 / vals


        w = csr_matrix((vals, (rows, cols)), shape=A.shape)


        n_components, labels = connected_components(w, directed=False, return_labels=True)

        if n_components == 1:
            connect = True
            print('connectivity graph!')
        else:
            K = K + 1

    scipy.sparse.save_npz("Data/npz/"+name + str(K) + "graph.npz",
                          w)
    print('Successfully saved to:Data/npz')
    return dic

def calMetricfunction(A, B, i, j, alpha=1):

    cluster_i_indices = np.where(np.array(B) == i)[0]
    cluster_j_indices = np.where(np.array(B) == j)[0]
    C_i = len(cluster_i_indices)
    C_j = len(cluster_j_indices)
    E_i = A[cluster_i_indices, :][:, cluster_i_indices].sum()
    non_zero_i=A[cluster_i_indices, :][:, cluster_i_indices]
    non_zero_count_i = non_zero_i.nnz



    E_j = A[cluster_j_indices, :][:, cluster_j_indices].sum()
    non_zero_j=A[cluster_j_indices, :][:, cluster_j_indices]
    non_zero_count_j = non_zero_j.nnz

    SE_ij=np.sum(A[np.ix_(cluster_i_indices, cluster_j_indices)])


    # RI
    if SE_ij==0:
      metricValue=0
      return metricValue
    else:
      relative_interconnectivity = 2*SE_ij / (E_i + E_j)
    # RC
    non_zero_count_ij = A[np.ix_(cluster_i_indices, cluster_j_indices)].count_nonzero()
    relative_closeness = (C_i+C_j)*SE_ij / non_zero_count_ij / (C_i*E_i/non_zero_count_i + C_j*E_j/non_zero_count_j)

    #metricValue = (relative_interconnectivity ** alpha)
    metricValue = relative_interconnectivity * (relative_closeness ** alpha)
    return metricValue


def update_cluster_membership(A, B, num_clusters, alpha=1):

    n = np.max(B) + 1
    arr = np.arange(n)
    metricValue = [[0 for j in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            metricValue[i][j]=calMetricfunction(A, B, i, j,alpha)
    print('Metrics matrix:',metricValue)
    while(len(arr) > num_clusters):

        max_val = np.max(metricValue)
        print("Max：",max_val)
        max_indices = np.argwhere(metricValue == max_val)


        for only in np.where(np.array(B) == int(max_indices[0][1])):
          for onlyone in only:
            B[onlyone] = max_indices[0][0]
        metricValue = np.array(metricValue)

        metricValue[max_indices[0][1], :] = 0
        metricValue[:, max_indices[0][1]] = 0

        arr = np.delete(arr, np.where(arr == max_indices[0][1]))
        print("len(clusterNum)：",len(arr),' Cluster number removed：',max_indices[0][1])
        for m in arr:
            if max_indices[0][0] < m:
                metricValue[max_indices[0][0]][m] = calMetricfunction(A, B, max_indices[0][0], m, alpha)
            if max_indices[0][0] > m:
                metricValue[m][max_indices[0][0]] = calMetricfunction(A, B, m, max_indices[0][0], alpha)
    return B

def update_cluster_membership1(dic, A, B, num_clusters, trace_num, clusters_num, alpha=1):

    n = np.max(B) + 1
    arr = np.arange(n)
    metricValue = [[0 for j in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            metricValue[i][j]=calMetricfunction(A, B, i, j, alpha)
    print('Metrics matrix:',metricValue)
    while(len(arr) > num_clusters):

        max_val = np.max(metricValue)
        max_indices = np.argwhere(metricValue == max_val)

        for only in np.where(np.array(B) == arr[max_indices[0][1]]):
            for onlyone in only:
                B[onlyone] = arr[max_indices[0][0]]



        arr = np.delete(arr, max_indices[0][1])
        print("len(clusterNum)",len(arr))
        if(len(arr) == clusters_num):

            result = getResult(dic, trace_num, B)
            print("Clustering results:", result)

        metricValue = [[0 for j in range(len(arr))] for i in range(len(arr))]
        time_i = 0
        for m in arr:
            time_j=time_i+1
            for mj in arr[time_i+1:]:
                metricValue[time_i][time_j] = calMetricfunction(A, B, m, mj, alpha)
                time_j=time_j+1
            time_i=time_i+1

    return B

def getResult(dic,trace_num,B):
    trace = [-1] * trace_num
    keys_list = list(dic.keys())
    for i in range(len(keys_list)):
        for j in dic[keys_list[i]]:
            trace[j] = B[i]
    return trace



def main():

    usage = """\
    usage:
        driftDetection.py [-l value] [-c value]
    options:
        -l -- the name of the event log
      	-c -- the number of clusters expected
        """
    import getopt, sys


    try:

        opts, args = getopt.getopt(sys.argv[1:], "l:c:")
        if len(args) == 0:
            print(usage)
            return

        logname = 'BPIC_2020'
        clusters_num = 2
        args=['BPIC_2020.xes']

        for opt, value in opts:
            if opt == '-l':
                logname = str(value)
            elif opt == '-c':
                clusters_num = int(value)

        print("--------------------------------------------------------------")
        print(" Log: ", args[0])
        print(" logname: ", logname)
        print(" clusters_num: ", clusters_num)
        print("--------------------------------------------------------------")


        dic = getKNN("Data/BPIC_xes/"+logname+'.xes', logname, 4, False)  # xes文件路径，k值

        trace_num = 7065

        A = scipy.sparse.load_npz("Data/npz/"+logname+"4graph.npz")

        G = nx.from_scipy_sparse_matrix(A)
        num_nodes = G.number_of_nodes()
        nx.set_node_attributes(G, {i: 1 for i in range(num_nodes)}, 'nvtxs')
        weights = A.data.tolist()

        edges = list(zip(A.nonzero()[0].tolist(), A.indices.tolist()))
        edge_weights = dict(zip(map(tuple, edges), weights))
        nx.set_edge_attributes(G, edge_weights, 'weight')


        (edgecuts, partitions) = metis.part_graph(G, 100)

        print(type(partitions))
        print("Graph partition:", partitions)


        B1 = update_cluster_membership1(dic, A, partitions, 2, trace_num, clusters_num, 1)
        print("Clustering result:", B1)

    except getopt.GetoptError:
        print(usage)
    except SyntaxError as error:
        print(error)
        print(usage)
    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main()



