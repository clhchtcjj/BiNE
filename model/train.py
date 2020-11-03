#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import numpy as np
from sklearn import preprocessing
from data_utils import DataUtils
from graph_utils import GraphUtils
import random
import math
import os
import pandas as pd
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,auc,precision_recall_fscore_support


def init_embedding_vectors(node_u, node_v, node_list_u, node_list_v, args):
    """
    initialize embedding vectors
    :param node_u:
    :param node_v:
    :param node_list_u:
    :param node_list_v:
    :param args:
    :return:
    """
    # user
    for i in node_u:
        vectors = np.random.random([1, args.d])
        help_vectors = np.random.random([1, args.d])
        node_list_u[i] = {}
        node_list_u[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm='l2')
        node_list_u[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm='l2')
    # item
    for i in node_v:
        vectors = np.random.random([1, args.d])
        help_vectors = np.random.random([1, args.d])
        node_list_v[i] = {}
        node_list_v[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm='l2')
        node_list_v[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm='l2')



def walk_generator(gul,args):
    """
    walk generator
    :param gul:
    :param args:
    :return:
    """
    gul.calculate_centrality(args.mode)
    if args.large == 0:
        gul.homogeneous_graph_random_walks(percentage=args.p, maxT=args.maxT, minT=args.minT)
    elif args.large == 1:
        gul.homogeneous_graph_random_walks_for_large_bipartite_graph(percentage=args.p, maxT=args.maxT, minT=args.minT)
    elif args.large == 2:
        gul.homogeneous_graph_random_walks_for_large_bipartite_graph_without_generating(datafile=args.train_data,percentage=args.p,maxT=args.maxT, minT=args.minT)
    return gul


def get_context_and_negative_samples(gul, args):
    """
    get context and negative samples offline
    :param gul:
    :param args:
    :return: context_dict_u, neg_dict_u, context_dict_v, neg_dict_v,gul.node_u,gul.node_v
    """
    if args.large == 0:
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        print("negative samples is ok.....")
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.G_u, gul.walks_u, args.ws, args.ns, neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.G_v, gul.walks_v, args.ws, args.ns, neg_dict_v)
    else:
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        # print len(gul.walks_u),len(gul.walks_u)
        print("negative samples is ok.....")
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.node_u, gul.walks_u, args.ws, args.ns, neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.node_v, gul.walks_v, args.ws, args.ns, neg_dict_v)

    return context_dict_u, neg_dict_u, context_dict_v, neg_dict_v,gul.node_u,gul.node_v


def skip_gram(center, contexts, negs, node_list, lam, pa):
    """
    skip-gram
    :param center:
    :param contexts:
    :param negs:
    :param node_list:
    :param lam:
    :param pa:
    :return:
    """
    loss = 0
    I_z = {center: 1}  # indication function
    for node in negs:
        I_z[node] = 0
    V = np.array(node_list[contexts]['embedding_vectors'])
    update = [[0] * V.size]
    for u in I_z.keys():
        if node_list.get(u) is  None:
            pass
        Theta = np.array(node_list[u]['context_vectors'])
        X = float(V.dot(Theta.T))
        sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))
        update += pa * lam * (I_z[u] - sigmod) * Theta
        node_list[u]['context_vectors'] += pa * lam * (I_z[u] - sigmod) * V
        try:
            loss += pa * (I_z[u] * math.log(sigmod) + (1 - I_z[u]) * math.log(1 - sigmod))
        except:
            pass
            # print "skip_gram:",
            # print(V,Theta,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
    return update, loss


def KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma):
    """
    KL-divergenceO1
    :param edge_dict_u:
    :param u:
    :param v:
    :param node_list_u:
    :param node_list_v:
    :param lam:
    :param gamma:
    :return:
    """
    loss = 0
    e_ij = edge_dict_u[u][v]

    update_u = 0
    update_v = 0
    U = np.array(node_list_u[u]['embedding_vectors'])
    V = np.array(node_list_v[v]['embedding_vectors'])
    X = float(U.dot(V.T))

    sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))

    update_u += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(math.e, math.e)) * V
    update_v += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(math.e, math.e)) * U

    try:
        loss += gamma * e_ij * math.log(sigmod)
    except:
        pass
        # print "KL:",
        # print(U,V,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
    return update_u, update_v, loss

def top_N(test_u, test_v, test_rate, node_list_u, node_list_v, top_n):
    recommend_dict = {}
    for u in test_u:
        recommend_dict[u] = {}
        for v in test_v:
            if node_list_u.get(u) is None:
                pre = 0
            else:
                U = np.array(node_list_u[u]['embedding_vectors'])
                if node_list_v.get(v) is None:
                    pre = 0
                else:
                    V = np.array(node_list_v[v]['embedding_vectors'])
                    pre = U.dot(V.T)[0][0]
            recommend_dict[u][v] = float(pre)

    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    for u in test_u:
        tmp_r = sorted(recommend_dict[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(recommend_dict[u]),top_n)]
        tmp_t = sorted(test_rate[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(test_rate[u]),top_n)]
        tmp_r_list = []
        tmp_t_list = []
        for (item, rate) in tmp_r:
            tmp_r_list.append(item)

        for (item, rate) in tmp_t:
            tmp_t_list.append(item)
        pre, rec = precision_and_recall(tmp_r_list,tmp_t_list)
        ap = AP(tmp_r_list,tmp_t_list)
        rr = RR(tmp_r_list,tmp_t_list)
        ndcg = nDCG(tmp_r_list,tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)
    precison = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    #print(precison, recall)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1,map,mrr,mndcg

def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i+1
        dcg += 1/ math.log(rank+1, 2)
    return dcg / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i+2, 2)
    return idcg

def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i+1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0

def RR(ranked_list, ground_list):

    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0

def precision_and_recall(ranked_list,ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits/(1.0 * len(ranked_list))
    rec = hits/(1.0 * len(ground_list))
    return pre, rec

def generateFeatureFile(filecase,filevector_u,filevector_v,fileout,factors):
    vectors_u = {}
    vectors_v = {}
    with open(filevector_u,'r') as fu:
        for line in fu.readlines():
            items = line.strip().split(' ')
            vectors_u[items[0]] = items[1:]
    with open(filevector_v,'r') as fv:
        for line in fv.readlines():
            items = line.strip().split(' ')
            vectors_v[items[0]] = items[1:]
    with open(filecase,'r') as fc, open(fileout,'w') as fo:
        for line in fc.readlines():
            items = line.strip().split('\t')
            if vectors_u.get(items[0]) == None:
                vectors_u[items[0]] = ['0'] * factors
            if vectors_v.get(items[1]) == None:
                vectors_v[items[1]] = ['0'] * factors
            if items[-1] == '1':
                fo.write('{}\t{}\t{}\n'.format('\t'.join(vectors_u[items[0]]),'\t'.join(vectors_v[items[1]]),1))
            else:
                fo.write('{}\t{}\t{}\n'.format('\t'.join(vectors_u[items[0]]),'\t'.join(vectors_v[items[1]]),0))

def link_prediction(args):
    filecase_a = args.case_train
    filecase_e = args.case_test
    filevector_u = args.vectors_u
    filevector_v = args.vectors_v
    filecase_a_c = r'../data/features_train.dat'
    filecase_e_c = r'../data/features_test.dat'
    generateFeatureFile(filecase_a,filevector_u,filevector_v,filecase_a_c,args.d)
    generateFeatureFile(filecase_e,filevector_u,filevector_v,filecase_e_c,args.d)

    df_data_train = pd.read_csv(filecase_a_c,header = None,sep='\t',encoding='utf-8')
    X_train = df_data_train.drop(len(df_data_train.keys())-1,axis = 1)
    y_train = df_data_train[len(df_data_train.keys())-1]

    df_data_test = pd.read_csv(filecase_e_c,header = None,sep='\t',encoding='utf-8')
    X_test = df_data_test.drop(len(df_data_train.keys())-1,axis = 1)
    X_test = X_test.fillna(X_test.mean())
    y_test = df_data_test[len(df_data_test.keys())-1]
    y_test_list = list(y_test)

    lg = LogisticRegression(penalty='l2',C=0.001)
    lg.fit(X_train,y_train)
    lg_y_pred_est = lg.predict_proba(X_test)[:,1]
    fpr,tpr,thresholds = metrics.roc_curve(y_test,lg_y_pred_est)
    average_precision = average_precision_score(y_test, lg_y_pred_est)
    os.remove(filecase_a_c)
    os.remove(filecase_e_c)
    return metrics.auc(fpr,tpr), average_precision

def train_by_sampling(args):
    model_path = os.path.join('../', args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    alpha, beta, gamma, lam = args.alpha, args.beta, args.gamma, args.lam
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, p : %0.4f, ws : %d, ns : %d, maxT : % d, minT : %d, max_iter : %d, d : %d' % (alpha, beta, gamma, lam, args.p, args.ws, args.ns,args.maxT,args.minT,args.max_iter, args.d))
    print('========== processing data ===========')
    dul = DataUtils(model_path)
    if args.rec:
        test_user, test_item, test_rate = dul.read_data(args.test_data)
    print("constructing graph....")
    gul = GraphUtils(model_path)
    gul.construct_training_graph(args.train_data)
    edge_dict_u = gul.edge_dict_u
    edge_list = gul.edge_list
    walk_generator(gul,args)
    print("getting context and negative samples....")
    context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, node_u, node_v = get_context_and_negative_samples(gul, args)
    node_list_u, node_list_v = {}, {}
    init_embedding_vectors(node_u, node_v, node_list_u, node_list_v, args)
    last_loss, count, epsilon = 0, 0, 1e-3
 
    print("============== training ==============")
    for iter in range(0, args.max_iter):
        s1 = "\r[%s%s]%0.2f%%"%("*"* iter," "*(args.max_iter-iter),iter*100.0/(args.max_iter-1))
        loss = 0
        visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))
        visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))
        random.shuffle(edge_list)
        for i in range(len(edge_list)):
            u, v, w = edge_list[i]
              
            length = len(context_dict_u[u])
            random.shuffle(context_dict_u[u])
            if visited_u.get(u) < length:
                # print(u)
                index_list = list(range(visited_u.get(u),min(visited_u.get(u)+1,length)))
                for index in index_list:
                    context_u = context_dict_u[u][index]
                    neg_u = neg_dict_u[u][index]
                    # center,context,neg,node_list,eta
                    for z in context_u:
                        tmp_z, tmp_loss = skip_gram(u, z, neg_u, node_list_u, lam, alpha)
                        node_list_u[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_u[u] = index_list[-1]+3

            length = len(context_dict_v[v])
            random.shuffle(context_dict_v[v])
            if visited_v.get(v) < length:
                # print(v)
                index_list = list(range(visited_v.get(v),min(visited_v.get(v)+1,length)))
                for index in index_list:
                    context_v = context_dict_v[v][index]
                    neg_v = neg_dict_v[v][index]
                    # center,context,neg,node_list,eta
                    for z in context_v:
                        tmp_z, tmp_loss = skip_gram(v, z, neg_v, node_list_v, lam, beta)
                        node_list_v[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_v[v] = index_list[-1]+3

            update_u, update_v, tmp_loss = KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma)
            loss += tmp_loss
            node_list_u[u]['embedding_vectors'] += update_u
            node_list_v[v]['embedding_vectors'] += update_v

        delta_loss = abs(loss - last_loss)
        if last_loss > loss:
            lam *= 1.05
        else:
            lam *= 0.95
        last_loss = loss
        if delta_loss < epsilon:
            break
        sys.stdout.write(s1)
        sys.stdout.flush()
    save_to_file(node_list_u,node_list_v,model_path,args)
    print("")
    if args.rec:
        print("============== testing ===============")
        f1, map, mrr, mndcg = top_N(test_user,test_item,test_rate,node_list_u,node_list_v,args.top_n)
        print('recommendation metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f' % (round(f1,4), round(map,4), round(mrr,4), round(mndcg,4)))
    if args.lip:
        print("============== testing ===============")
        auc_roc, auc_pr = link_prediction(args)
        print('link prediction metrics: AUC_ROC : %0.4f, AUC_PR : %0.4f' % (round(auc_roc,4), round(auc_pr,4)))

def train(args):
    model_path = os.path.join('../', args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    alpha, beta, gamma, lam = args.alpha, args.beta, args.gamma, args.lam
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, p : %0.4f, ws : %d, ns : %d, maxT : % d, minT : %d, max_iter : %d, d : %d' % (alpha, beta, gamma, lam, args.p, args.ws, args.ns,args.maxT,args.minT,args.max_iter, args.d))
    print('========== processing data ===========')
    dul = DataUtils(model_path)
    if args.rec:
        test_user, test_item, test_rate = dul.read_data(args.test_data)
    print("constructing graph....")
    gul = GraphUtils(model_path)
    gul.construct_training_graph(args.train_data)
    edge_dict_u = gul.edge_dict_u
    edge_list = gul.edge_list
    walk_generator(gul,args)

    print("getting context and negative samples....")
    context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, node_u, node_v = get_context_and_negative_samples(gul, args)
    node_list_u, node_list_v = {}, {}
    init_embedding_vectors(node_u, node_v, node_list_u, node_list_v, args)

    last_loss, count, epsilon = 0, 0, 1e-3
    print("============== training ==============")
    for iter in range(0, args.max_iter):
        s1 = "\r[%s%s]%0.2f%%"%("*"* iter," "*(args.max_iter-iter),iter*100.0/(args.max_iter-1))
        loss = 0
        num = 0
        visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))
        visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))

        random.shuffle(edge_list)
        for (u, v, w) in edge_list:
            if visited_u.get(u) == 0 or random.random() > 0.95:
                # print(u)
                length = len(context_dict_u[u])
                index_list = random.sample(list(range(length)), min(length, 1))
                for index in index_list:
                    context_u = context_dict_u[u][index]
                    neg_u = neg_dict_u[u][index]
                    # center,context,neg,node_list,eta
                    for k, z in enumerate(context_u):
                        tmp_z, tmp_loss = skip_gram(u, z, neg_u, node_list_u, lam, alpha)
                        node_list_u[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_u[u] = 1
            if visited_v.get(v) == 0 or random.random() > 0.95:
                # print(v)
                length = len(context_dict_v[v])
                index_list = random.sample(list(range(length)), min(length, 1))
                for index in index_list:
                    context_v = context_dict_v[v][index]
                    neg_v = neg_dict_v[v][index]
                    # center,context,neg,node_list,eta
                    for k,z in enumerate(context_v):
                        tmp_z, tmp_loss = skip_gram(v, z, neg_v, node_list_v, lam, beta)
                        node_list_v[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_v[v] = 1
            # print(len(edge_dict_u))
            update_u, update_v, tmp_loss = KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma)
            loss += tmp_loss
            node_list_u[u]['embedding_vectors'] += update_u
            node_list_v[v]['embedding_vectors'] += update_v
            count = iter
            num += 1
        delta_loss = abs(loss - last_loss)
        if last_loss > loss:
            lam *= 1.05
        else:
            lam *= 0.95
        last_loss = loss
        if delta_loss < epsilon:
            break
        sys.stdout.write(s1)
        sys.stdout.flush()
    save_to_file(node_list_u,node_list_v,model_path,args)
    print("")
    if args.rec:
        print("============== testing ===============")
        f1, map, mrr, mndcg = top_N(test_user,test_item,test_rate,node_list_u,node_list_v,args.top_n)
        print('recommendation metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f' % (round(f1,4), round(map,4), round(mrr,4), round(mndcg,4)))
    if args.lip:
        print("============== testing ===============")
        auc_roc, auc_pr = link_prediction(args)
        print('link prediction metrics: AUC_ROC : %0.4f, AUC_PR : %0.4f' % (round(auc_roc,4), round(auc_pr,4)))
    


def ndarray_tostring(array):
    string = ""
    for item in array[0]:
        string += str(item).strip()+" "
    return string+"\n"

def save_to_file(node_list_u,node_list_v,model_path,args):
    with open(args.vectors_u,"w") as fw_u:
        for u in node_list_u.keys():
            fw_u.write(u+" "+ ndarray_tostring(node_list_u[u]['embedding_vectors']))
    with open(args.vectors_v,"w") as fw_v:
        for v in node_list_v.keys():
            fw_v.write(v+" "+ndarray_tostring(node_list_v[v]['embedding_vectors']))


def main():
    parser = ArgumentParser("BiNE",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--train-data', default=r'../data/rating_train.dat',
                        help='Input graph file.')

    parser.add_argument('--test-data', default=r'../data/rating_test.dat')

    parser.add_argument('--model-name', default='default',
                        help='name of model.')

    parser.add_argument('--vectors-u', default=r'../data/vectors_u.dat',
                        help="file of embedding vectors of U")

    parser.add_argument('--vectors-v', default=r'../data/vectors_v.dat',
                        help="file of embedding vectors of V")

    parser.add_argument('--case-train', default=r'../data/wiki/case_train.dat',
                        help="file of training data for LR")

    parser.add_argument('--case-test', default=r'../data/wiki/case_test.dat',
                        help="file of testing data for LR")

    parser.add_argument('--ws', default=5, type=int,
                        help='window size.')

    parser.add_argument('--ns', default=4, type=int,
                        help='number of negative samples.')

    parser.add_argument('--d', default=128, type=int,
                        help='embedding size.')

    parser.add_argument('--maxT', default=32, type=int,
                        help='maximal walks per vertex.')

    parser.add_argument('--minT', default=1, type=int,
                        help='minimal walks per vertex.')

    parser.add_argument('--p', default=0.15, type=float,
                        help='walk stopping probability.')

    parser.add_argument('--alpha', default=0.01, type=float,
                        help='trade-off parameter alpha.')

    parser.add_argument('--beta', default=0.01, type=float,
                        help='trade-off parameter beta.')

    parser.add_argument('--gamma', default=0.1, type=float,
                        help='trade-off parameter gamma.')

    parser.add_argument('--lam', default=0.01, type=float,
                        help='learning rate lambda.')
    parser.add_argument('--max-iter', default=50, type=int,
                        help='maximal number of iterations.')

    parser.add_argument('--top-n', default=10, type=int,
                        help='recommend top-n items for each user.')

    parser.add_argument('--rec', default=0, type=int,
                        help='calculate the recommendation metrics.')

    parser.add_argument('--lip', default=0, type=int,
                        help='calculate the link prediction metrics.')

    parser.add_argument('--large', default=0, type=int,
                        help='for large bipartite, 1 do not generate homogeneous graph file; 2 do not generate homogeneous graph')

    parser.add_argument('--mode', default='hits', type=str,
                        help='metrics of centrality')

    args = parser.parse_args()
    train_by_sampling(args)

if __name__ == "__main__":
    sys.exit(main())

