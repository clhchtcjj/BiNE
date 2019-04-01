# BiNE: Bipartite Network Embedding

This repository contains the demo code of the paper: 

> BiNE: Bipartite Network Embedding. Ming Gao, Leihui Chen, Xiangnan He & Aoying Zhou

which has been accepted by **SIGIR2018**.

`Note`: Any problems, you can contact me at [leihuichen@gmail.com](mailto:leihuichen@gmail.com). Through email, you will get my rapid response.



## Environment settings

- python==2.7.11
- numpy==1.13.3
- sklearn==0.17.1
- networkx==1.11
- datasketch==1.2.5
- scipy==0.17.0
- six==1.10.0





## Basic Usage

**Main Parameters:**

```
Input graph path. Defult is '../data/rating_train.dat' (--train-data)
Test dataset path. Default is '../data/rating_test.dat' (--test-data)
Name of model. Default is 'default' (--model-name)
Number of dimensions. Default is 128 (--d)
Number of negative samples. Default is 4 (--ns)
Size of window. Default is 5 (--ws)
Trade-off parameter $\alpha$. Default is 0.01 (--alpha)
Trade-off parameter $\beta$. Default is 0.01 (--beta)
Trade-off parameter $\gamma$. Default is 0.1 (--gamma)
Learning rate $\lambda$. Default is 0.01 (--lam)
Maximal iterations. Default is 50 (--max-iters)
Maximal walks per vertex. Default is 32 (--maxT)
Minimal walks per vertex. Default is 1 (--minT)
Walk stopping probability. Default is 0.15 (--p)
Calculate the recommendation metrics. Default is 0 (--rec)
Calculate the link prediction. Default is 0 (--lip)
File of training data for LR. Default is '../data/wiki/case_train.dat' (--case-train)
File of testing data for LR. Default is '../data/wiki/case_test.dat' (--case-test)
File of embedding vectors of U. Default is '../data/vectors_u.dat' (--vectors-u)
File of embedding vectors of V. Default is '../data/vectors_v.dat' (--vectors-v)
For large bipartite, 1 do not generate homogeneous graph file; 2 do not generate homogeneous graph. Default is 0 (--large)
Mertics of centrality. Default is 'hits', options: 'hits' and 'degree_centrality' (--mode)
```

**Usage**

We provide two processed dataset:

- DBLP (for recommendation). It contains:

  - A training dataset     ./data/dblp/rating_train.dat 
  - A testing dataset      ./data/dblp/rating_test.dat

- Wikipedia (for link prediction). It contains:

  - A training dataset     ./data/wiki/rating_train.dat 
  - A testing dataset      ./data/wiki/rating_test.dat

- Each line is a instance: userID (begin with 'u')\titemID (begin with 'i') \t weight\n

  For example: u0\ti0\t1

Please run the './model/train.py' 

```
cd model
python train.py --train-data ../data/dblp/rating_train.dat --test-data ../data/dblp/rating_test.dat --lam 0.025 --max-iter 100 --model-name dblp --rec 1 --large 2 --vectors-u ../data/dblp/vectors_u.dat --vectors-v ../data/dblp/vectors_v.dat
```

The embedding vectors of nodes are saved in file '/model-name/vectors_u.dat' and '/model-name/vectors_v.dat', respectively.



## Example

### Recommendation

**Run**

```
cd model
python train.py --train-data ../data/dblp/rating_train.dat --test-data ../data/dblp/rating_test.dat --lam 0.025 --max-iter 100 --model-name dblp --rec 1 --large 2 --vectors-u ../data/dblp/vectors_u.dat --vectors-v ../data/dblp/vectors_v.dat
```

**Output** (training process)

```
======== experiment settings =========
alpha : 0.0100, beta : 0.0100, gamma : 0.1000, lam : 0.0250, p : 0.1500, ws : 5, ns : 4, maxT :  32, minT : 1, max_iter : 100
========== processing data ===========
constructing graph....
number of nodes: 6001
walking...
walking...ok
number of nodes: 1177
walking...
walking...ok
getting context and negative samples....
negative samples is ok.....
context...
context...ok
context...
context...ok
============== training ==============
[*************************************************************************************************** ]100.00%
```

**Output** (testing process)

```
============== testing ===============
recommendation metrics: F1 : 0.1132, MAP : 0.2041, MRR : 0.3331, NDCG : 0.2609
```



### Link Prediction

**Run**

```
cd model
python train.py --train-data ../data/wiki/rating_train.dat --test-data ../data/wiki/rating_test.dat --lam 0.01 --max-iter 100 --model-name wiki --lip 1 --large 2 --gamma 1 --vectors-u ../data/wiki/vectors_u.dat --vectors-v ../data/wiki/vectors_v.dat --case-train ../data/wiki/case_train.dat --case-test ../data/wiki/case_test.dat
```

**Output** (training process)

```
======== experiment settings =========
alpha : 0.0100, beta : 0.0100, gamma : 1.0000, lam : 0.0100, p : 0.1500, ws : 5, ns : 4, maxT :  32, minT : 1, max_iter : 100, d : 128
========== processing data ===========
constructing graph....
number of nodes: 15000
walking...
walking...ok
number of nodes: 2529
walking...
walking...ok
getting context and negative samples....
negative samples is ok.....
context...
context...ok
context...
context...ok
============== training ==============
[*************************************************************************************************** ]100.00%
```

**Output** (testing process)

```
============== testing ===============
link prediction metrics: AUC_ROC : 0.9468, AUC_PR : 0.9614
```

