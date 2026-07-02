# LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

**Authors:** Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang
* **University of Science and Technology of China** (Xiangnan He, Kuan Deng, Yongdong Zhang) - `{xiangnanhe@gmail.com, dengkuan@mail.ustc.edu.cn, zhyd73@ustc.edu.cn}`
* **National University of Singapore** (Xiang Wang) - `xiangwang@u.nus.edu`
* **Beijing Kuaishou Technology Co., Ltd.** (Yan Li) - `liyan@kuaishou.com`
* **Hefei University of Technology** (Meng Wang - Corresponding Author) - `eric.mengwang@gmail.com`

**Venue:** Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '20), Virtual Event, China.
**DOI:** [10.1145/3397271.3401063](https://doi.org/10.1145/3397271.3401063)

---

## Abstract
Graph Convolution Network (GCN) has become new state-of-the-art for collaborative filtering. Nevertheless, the reasons of its effectiveness for recommendation are not well understood. Existing work that adapts GCN to recommendation lacks thorough ablation analyses on GCN, which is originally designed for graph classification tasks and equipped with many neural network operations. However, we empirically find that the two most common designs in GCNs — feature transformation and nonlinear activation — contribute little to the performance of collaborative filtering. Even worse, including them adds to the difficulty of training and degrades recommendation performance.

In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named **LightGCN**, including only the most essential component in GCN — neighborhood aggregation — for collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings learned at all layers as the final embedding. Such simple, linear, and neat model is much easier to implement and train, exhibiting substantial improvements (about 16.0% relative improvement on average) over Neural Graph Collaborative Filtering (NGCF) — a state-of-the-art GCN-based recommender model — under exactly the same experimental setting. Further analyses are provided towards the rationality of the simple LightGCN from both analytical and empirical perspectives. Our implementations are available in both TensorFlow and PyTorch.

**CCS Concepts:** 
* **Information systems** $\rightarrow$ **Recommender systems**.

**Keywords:** Collaborative Filtering, Recommendation, Embedding Propagation, Graph Neural Network

---

## 1 Introduction
To alleviate information overload on the web, recommender system has been widely deployed to perform personalized information filtering [7, 45, 46]. The core of recommender system is to predict whether a user will interact with an item, e.g., click, rate, purchase, among other forms of interactions. As such, collaborative filtering (CF), which focuses on exploiting the past user-item interactions to achieve the prediction, remains to be a fundamental task towards effective personalized recommendation [10, 19, 28, 39].

The most common paradigm for CF is to learn latent features (a.k.a. embedding) to represent a user and an item, and perform prediction based on the embedding vectors [6, 19]. Matrix factorization is an early such model, which directly projects the single ID of a user to her embedding [26]. Later on, several research find that augmenting user ID with her interaction history as the input can improve the quality of embedding. For example, SVD++ [25] demonstrates the benefits of user interaction history in predicting user numerical ratings, and Neural Attentive Item Similarity (NAIS) [18] differentiates the importance of items in the interaction history and shows improvements in predicting item ranking. In view of user-item interaction graph, these improvements can be seen as coming from using the subgraph structure of a user — more specifically, her one-hop neighbors — to improve the embedding learning.

To deepen the use of subgraph structure with high-hop neighbors, Wang et al. [39] recently proposes NGCF and achieves state-of-the-art performance for CF. It takes inspiration from the Graph Convolution Network (GCN) [14, 23], following the same propagation rule to refine embeddings: feature transformation, neighborhood aggregation, and nonlinear activation. Although NGCF has shown promising results, we argue that its designs are rather heavy and burdensome — many operations are directly inherited from GCN without justification. As a result, they are not necessarily useful for the CF task. To be specific, GCN is originally proposed for node classification on attributed graph, where each node has rich attributes as input features; whereas in user-item interaction graph for CF, each node (user or item) is only described by a one-hot ID, which has no concrete semantics besides being an identifier. In such a case, given the ID embedding as the input, performing multiple layers of nonlinear feature transformation — which is the key to the success of modern neural networks [16] — will bring no benefits, but negatively increases the difficulty for model training.

To validate our thoughts, we perform extensive ablation studies on NGCF. With rigorous controlled experiments (on the same data splits and evaluation protocol), we draw the conclusion that the two operations inherited from GCN — feature transformation and nonlinear activation — has no contribution on NGCF’s effectiveness. Even more surprising, removing them leads to significant accuracy improvements. This reflects the issues of adding operations that are useless for the target task in graph neural network, which not only brings no benefits, but rather degrades model effectiveness.

Motivated by these empirical findings, we present a new model named **LightGCN**, including only the most essential component of GCN — neighborhood aggregation — for collaborative filtering. Specifically, after associating each user (item) with an ID embedding, we propagate the embeddings on the user-item interaction graph to refine them. We then combine the embeddings learned at different propagation layers with a weighted sum to obtain the final embedding for prediction. The whole model is simple and elegant, which not only is easier to train, but also achieves better empirical performance than NGCF and other state-of-the-art methods like Mult-VAE [28].

To summarize, this work makes the following main contributions:
* We empirically show that two common designs in GCN, feature transformation and nonlinear activation, have no positive effect on the effectiveness of collaborative filtering.
* We propose LightGCN, which largely simplifies the model design by including only the most essential components in GCN for recommendation.
* We empirically compare LightGCN with NGCF by following the same setting and demonstrate substantial improvements. In-depth analyses are provided towards the rationality of LightGCN from both technical and empirical perspectives.

---

## 2 Preliminaries

We first introduce NGCF [39], a representative and state-of-the-art GCN model for recommendation. We then perform ablation studies on NGCF to judge the usefulness of each operation. The novel contribution of this section is to show that the two common designs in GCNs, feature transformation and nonlinear activation, have no positive effect on collaborative filtering.

### 2.1 NGCF Brief
In the initial step, each user and item is associated with an ID embedding. Let $\mathbf{e}_u^{(0)}$ denote the ID embedding of user $u$ and $\mathbf{e}_i^{(0)}$ denote the ID embedding of item $i$. Then NGCF leverages the user-item interaction graph to propagate embeddings as:

$$\mathbf{e}_u^{(k+1)} = \sigma \left( \mathbf{W}_1 \mathbf{e}_u^{(k)} + \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} \left( \mathbf{W}_1 \mathbf{e}_i^{(k)} + \mathbf{W}_2 (\mathbf{e}_i^{(k)} \odot \mathbf{e}_u^{(k)}) \right) \right)$$

$$\mathbf{e}_i^{(k+1)} = \sigma \left( \mathbf{W}_1 \mathbf{e}_i^{(k)} + \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} \left( \mathbf{W}_1 \mathbf{e}_u^{(k)} + \mathbf{W}_2 (\mathbf{e}_u^{(k)} \odot \mathbf{e}_i^{(k)}) \right) \right)$$ (Equation 1)

where $\mathbf{e}_u^{(k)}$ and $\mathbf{e}_i^{(k)}$ respectively denote the refined embedding of user $u$ and item $i$ after $k$ layers propagation, $\sigma$ is the nonlinear activation function, $\mathcal{N}_u$ denotes the set of items that are interacted by user $u$, $\mathcal{N}_i$ denotes the set of users that interact with item $i$, and $\mathbf{W}_1$ and $\mathbf{W}_2$ are trainable weight matrices to perform feature transformation in each layer. By propagating $L$ layers, NGCF obtains $L+1$ embeddings to describe a user $(\mathbf{e}_u^{(0)}, \mathbf{e}_u^{(1)}, \dots, \mathbf{e}_u^{(L)})$ and an item $(\mathbf{e}_i^{(0)}, \mathbf{e}_i^{(1)}, \dots, \mathbf{e}_i^{(L)})$. It then concatenates these $L+1$ embeddings to obtain the final user embedding and item embedding, using inner product to generate the prediction score.

NGCF largely follows the standard GCN [23], including the use of nonlinear activation function $\sigma(\cdot)$ and feature transformation matrices $\mathbf{W}_1$ and $\mathbf{W}_2$. However, we argue that the two operations are not as useful for collaborative filtering. In semi-supervised node classification, each node has rich semantic features as input, such as the title and abstract words of a paper. Thus performing multiple layers of nonlinear transformation is beneficial to feature learning. Nevertheless, in collaborative filtering, each node of user-item interaction graph only has an ID as input which has no concrete semantics. In this case, performing multiple nonlinear transformations will not contribute to learn better features; even worse, it may add the difficulties to train well.

### 2.2 Empirical Explorations on NGCF
We conduct ablation studies on NGCF to explore the effect of nonlinear activation and feature transformation. We use the codes released by the authors of NGCF, running experiments on the same data splits and evaluation protocol to keep the comparison as fair as possible. Since the core of GCN is to refine embeddings by propagation, we are more interested in the embedding quality under the same embedding size. Thus, we change the way of obtaining final embedding from concatenation (i.e., $\mathbf{e}_u^* = \mathbf{e}_u^{(0)} \parallel \dots \parallel \mathbf{e}_u^{(L)}$) to sum (i.e., $\mathbf{e}_u^* = \mathbf{e}_u^{(0)} + \dots + \mathbf{e}_u^{(L)}$). Note that this change has little effect on NGCF's performance, but makes the following ablation studies more indicative of the embedding quality refined by GCN.

We implement three simplified variants of NGCF:
* **NGCF-f**, which removes the feature transformation matrices $\mathbf{W}_1$ and $\mathbf{W}_2$.
* **NGCF-n**, which removes the non-linear activation function $\sigma$.
* **NGCF-fn**, which removes both the feature transformation matrices and non-linear activation function.

For the three variants, we keep all hyper-parameters (e.g., learning rate, regularization coefficient, dropout ratio, etc.) same as the optimal settings of NGCF. We report the results of the 2-layer setting on the Gowalla and Amazon-Book datasets in Table 1. 

#### Table 1: Performance of NGCF and its three variants.
| Method | Gowalla (Recall) | Gowalla (NDCG) | Amazon-Book (Recall) | Amazon-Book (NDCG) |
| :--- | :---: | :---: | :---: | :---: |
| **NGCF** | 0.1547 | 0.1307 | 0.0330 | 0.0254 |
| **NGCF-f** | 0.1686 | 0.1439 | 0.0368 | 0.0283 |
| **NGCF-n** | 0.1536 | 0.1295 | 0.0336 | 0.0258 |
| **NGCF-fn** | **0.1742** | **0.1476** | **0.0399** | **0.0303** |

As can be seen, removing feature transformation (i.e., NGCF-f) leads to consistent improvements over NGCF on all three datasets. In contrast, removing nonlinear activation does not affect the accuracy that much. However, if we remove nonlinear activation on the basis of removing feature transformation (i.e., NGCF-fn), the performance is improved significantly. From these observations, we conclude the findings that:
1. Adding feature transformation imposes negative effect on NGCF, since removing it in both models of NGCF and NGCF-n improves the performance significantly;
2. Adding nonlinear activation affects slightly when feature transformation is included, but it imposes negative effect when feature transformation is disabled.
3. As a whole, feature transformation and nonlinear activation impose rather negative effect on NGCF, since by removing them simultaneously, NGCF-fn demonstrates large improvements over NGCF (9.57% relative improvement on recall).

![Figure 1: Training curves (training loss and testing recall) of NGCF and its three simplified variants.](images/lightgcn_figure1.png) *(Note: Please place the cropped image of Figure 1 from the PDF under the `images/` directory as `lightgcn_figure1.png`)*

From these evidences, we can draw the conclusion that the deterioration of NGCF stems from the training difficulty, rather than overfitting. Theoretically speaking, NGCF has higher representation power than NGCF-f, since setting the weight matrix $\mathbf{W}_1$ and $\mathbf{W}_2$ to identity matrix $\mathbf{I}$ can fully recover the NGCF-f model. However, in practice, NGCF demonstrates higher training loss and worse generalization performance than NGCF-f. And the incorporation of nonlinear activation further aggravates the discrepancy between representation power and generalization performance.

---

## 3 Method

Driven by these findings, we set the goal of developing a light yet effective model by including the most essential ingredients of GCN for recommendation.

![Figure 2: An illustration of LightGCN model architecture.](images/lightgcn_figure2.png) *(Note: Please place the cropped image of Figure 2 from the PDF under the `images/` directory as `lightgcn_figure2.png`)*

### 3.1 LightGCN
The basic idea of GCN is to learn representation for nodes by smoothing features over the graph [23, 40]. To achieve this, it performs graph convolution iteratively, i.e., aggregating the features of neighbors as the new representation of a target node. Such neighborhood aggregation can be abstracted as:

$$\mathbf{e}_u^{(k+1)} = \text{AGG}(\mathbf{e}_u^{(k)}, \{\mathbf{e}_i^{(k)} : i \in \mathcal{N}_u\})$$ (Equation 2)

#### 3.1.1 Light Graph Convolution (LGC)
In LightGCN, we adopt the simple weighted sum aggregator and abandon the use of feature transformation and nonlinear activation. The graph convolution operation (propagation rule) in LightGCN is defined as:

$$\mathbf{e}_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}} \mathbf{e}_i^{(k)}$$

$$\mathbf{e}_i^{(k+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i|}\sqrt{|\mathcal{N}_u|}} \mathbf{e}_u^{(k)}$$ (Equation 3)

The symmetric normalization term $1/\left(\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}\right)$ follows the design of standard GCN [23], which can avoid the scale of embeddings increasing with graph convolution operations. Note that in LGC, we aggregate only the connected neighbors and do not integrate the target node itself (i.e., self-connection). The layer combination operation, introduced next, captures the same effect as self-connections.

#### 3.1.2 Layer Combination and Model Prediction
In LightGCN, the only trainable model parameters are the embeddings at the 0-th layer, i.e., $\mathbf{e}_u^{(0)}$ for all users and $\mathbf{e}_i^{(0)}$ for all items. When they are given, the embeddings at higher layers can be computed via LGC defined in Equation (3). After $K$ layers LGC, we combine the embeddings obtained at each layer to form the final representation of a user (an item):

$$\mathbf{e}_u = \sum_{k=0}^K \alpha_k \mathbf{e}_u^{(k)}; \quad \mathbf{e}_i = \sum_{k=0}^K \alpha_k \mathbf{e}_i^{(k)}$$ (Equation 4)

where $\alpha_k \ge 0$ denotes the importance of the $k$-th layer embedding in constituting the final embedding. It can be treated as a hyper-parameter to be tuned manually, or as a model parameter to be optimized automatically. In our experiments, we find that setting $\alpha_k$ uniformly as $1/(K + 1)$ leads to good performance in general.

The reasons that we perform layer combination to get final representations are three-fold:
1. With the increasing of the number of layers, the embeddings will be over-smoothed [27]. Thus simply using the last layer is problematic.
2. The embeddings at different layers capture different semantics. E.g., the first layer enforces smoothness on users and items that have interactions, the second layer smooths users (items) that have overlap on interacted items (users), and higher-layers capture higher-order proximity [39].
3. Combining embeddings at different layers with weighted sum captures the effect of graph convolution with self-connections (proof in Section 3.2.1).

The model prediction is defined as the inner product of user and item final representations:

$$\hat{y}_{ui} = \mathbf{e}_u^\top \mathbf{e}_i$$ (Equation 5)

which is used as the ranking score for recommendation generation.

#### 3.1.3 Matrix Form
Let the user-item interaction matrix be $\mathbf{R} \in \mathbb{R}^{M \times N}$ where $M$ and $N$ denote the number of users and items, respectively, and each entry $R_{ui}$ is $1$ if $u$ has interacted with item $i$ otherwise $0$. We obtain the adjacency matrix of the user-item graph as:

$$\mathbf{A} = \begin{bmatrix} \mathbf{0} & \mathbf{R} \\ \mathbf{R}^\top & \mathbf{0} \end{bmatrix}$$ (Equation 6)

Let the 0-th layer embedding matrix be $\mathbf{E}^{(0)} \in \mathbb{R}^{(M+N) \times T}$, where $T$ is the embedding size. Then we can obtain the matrix equivalent form of LGC as:

$$\mathbf{E}^{(k+1)} = (\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}) \mathbf{E}^{(k)}$$ (Equation 7)

where $\mathbf{D}$ is degree matrix. Lastly, we get the final embedding matrix used for model prediction as:

$$\mathbf{E} = \alpha_0 \mathbf{E}^{(0)} + \alpha_1 \mathbf{E}^{(1)} + \dots + \alpha_K \mathbf{E}^{(K)}$$
$$= \alpha_0 \mathbf{E}^{(0)} + \alpha_1 \tilde{\mathbf{A}} \mathbf{E}^{(0)} + \alpha_2 \tilde{\mathbf{A}}^2 \mathbf{E}^{(0)} + \dots + \alpha_K \tilde{\mathbf{A}}^K \mathbf{E}^{(0)}$$ (Equation 8)

where $\tilde{\mathbf{A}} = \mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}$ is the symmetrically normalized adjacency matrix.

---

### 3.2 Model Analysis
#### 3.2.1 Relation with SGCN
In [40], the authors propose SGCN, which simplifies GCN by removing nonlinearities and collapsing the weight matrices. The graph convolution in SGCN is defined as:

$$\mathbf{E}^{(k+1)} = (\mathbf{D} + \mathbf{I})^{-\frac{1}{2}} (\mathbf{A} + \mathbf{I}) (\mathbf{D} + \mathbf{I})^{-\frac{1}{2}} \mathbf{E}^{(k)}$$ (Equation 9)

In SGCN, the embeddings obtained at the last layer are used for downstream prediction:

$$\mathbf{E}^{(K)} = (\mathbf{A} + \mathbf{I})^K \mathbf{E}^{(0)}$$
$$= \binom{K}{0} \mathbf{E}^{(0)} + \binom{K}{1} \mathbf{A} \mathbf{E}^{(0)} + \dots + \binom{K}{K} \mathbf{A}^K \mathbf{E}^{(0)}$$ (Equation 10)

This derivation shows that inserting self-connections into $\mathbf{A}$ and propagating embeddings on it is essentially equivalent to a weighted sum of the embeddings propagated at each LGC layer.

#### 3.2.2 Relation with APPNP
In [24], the authors connect GCN with Personalized PageRank and propose APPNP:

$$\mathbf{E}^{(k+1)} = \beta \mathbf{E}^{(0)} + (1 - \beta) \tilde{\mathbf{A}} \mathbf{E}^{(k)}$$ (Equation 11)

By expanding recursively, the last layer is:

$$\mathbf{E}^{(K)} = \beta \mathbf{E}^{(0)} + \beta(1 - \beta)\tilde{\mathbf{A}}\mathbf{E}^{(0)} + \dots + (1 - \beta)^K \tilde{\mathbf{A}}^K \mathbf{E}^{(0)}$$ (Equation 12)

By setting $\alpha_k$ accordingly, LightGCN can fully recover the prediction embedding used by APPNP. Thus, LightGCN shares the strength of APPNP in combating over-smoothing.

#### 3.2.3 Second-Order Embedding Smoothness
Taking the user side as an example, the second layer of LGC smooths users that have overlap on the interacted items:

$$\mathbf{e}_u^{(2)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}} \mathbf{e}_i^{(1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{|\mathcal{N}_i|} \sum_{v \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_v|}} \mathbf{e}_v^{(0)}$$ (Equation 13)

If another user $v$ has co-interacted with the target user $u$, the smoothness strength of $v$ on $u$ is measured by the coefficient:

$$c_{v \rightarrow u} = \frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_v|}} \sum_{i \in \mathcal{N}_u \cap \mathcal{N}_v} \frac{1}{|\mathcal{N}_i|}$$ (Equation 14)

This coefficient is highly interpretable, scaling with the number of co-interacted items, penalizing high popularity of the co-interacted items, and penalizing the activity of user $v$.

---

### 3.3 Model Training
We employ the Bayesian Personalized Ranking (BPR) loss [32] which is a pairwise loss that encourages the prediction of an observed entry to be higher than its unobserved counterparts:

$$L_{BPR} = -\sum_{u=1}^M \sum_{i \in \mathcal{N}_u} \sum_{j \notin \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|\mathbf{E}^{(0)}\|^2$$ (Equation 15)

We employ the Adam optimizer and use it in a mini-batch manner. We do not introduce dropout mechanisms as they are redundant when feature transformation matrices are removed.

---

## 4 Experiments

### 4.1 Experimental Settings
We utilize the same datasets as the NGCF work: **Gowalla**, **Yelp2018**, and **Amazon-Book**. The statistics are shown below:

#### Table 2: Statistics of the experimented data.
| Dataset | User # | Item # | Interaction # | Density |
| :--- | :---: | :---: | :---: | :---: |
| **Gowalla** | 29,858 | 40,981 | 1,027,370 | 0.00084 |
| **Yelp2018** | 31,668 | 38,048 | 1,561,406 | 0.00130 |
| **Amazon-Book** | 52,643 | 91,599 | 2,984,108 | 0.00062 |

---

### 4.2 Performance Comparison with NGCF
We record the performance of NGCF and LightGCN at different layer settings in Table 3:

#### Table 3: Performance comparison between NGCF and LightGCN at different layers.
| Layer # | Method | Gowalla (Recall) | Gowalla (NDCG) | Yelp2018 (Recall) | Yelp2018 (NDCG) | Amazon-Book (Recall) | Amazon-Book (NDCG) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Layer** | NGCF | 0.1556 | 0.1315 | 0.0543 | 0.0442 | 0.0313 | 0.0241 |
| | LightGCN | 0.1755 (+12.79%) | 0.1492 (+13.46%) | 0.0631 (+16.20%) | 0.0515 (+16.51%) | 0.0384 (+22.68%) | 0.0298 (+23.65%) |
| **2 Layers** | NGCF | 0.1547 | 0.1307 | 0.0566 | 0.0465 | 0.0330 | 0.0254 |
| | LightGCN | 0.1777 (+14.84%) | 0.1524 (+16.60%) | 0.0622 (+9.89%) | 0.0504 (+8.38%) | 0.0411 (+24.54%) | 0.0315 (+24.02%) |
| **3 Layers** | NGCF | 0.1569 | 0.1327 | 0.0579 | 0.0477 | 0.0337 | 0.0261 |
| | LightGCN | **0.1823 (+16.19%)** | **0.1555 (+17.18%)** | 0.0639 (+10.38%) | 0.0525 (+10.06%) | 0.0410 (+21.66%) | **0.0318 (+21.84%)** |
| **4 Layers** | NGCF | 0.1570 | 0.1327 | 0.0566 | 0.0461 | 0.0344 | 0.0263 |
| | LightGCN | 0.1830 (+16.56%) | 0.1550 (+16.80%) | **0.0649 (+14.58%)** | **0.0530 (+15.02%)** | **0.0406 (+17.92%)** | 0.0313 (+18.92%) |

![Figure 3: Training curves of LightGCN and NGCF.](images/lightgcn_figure3.png) *(Note: Please place the cropped image of Figure 3 from the PDF under the `images/` directory as `lightgcn_figure3.png`)*

In all cases, LightGCN outperforms NGCF by a large margin. On average, the recall improvement on the three datasets is 16.52% and the NDCG improvement is 16.87%.

---

### 4.3 Performance Comparison with State-of-the-Arts

#### Table 4: The comparison of overall performance among LightGCN and competing methods.
| Method | Gowalla (Recall) | Gowalla (NDCG) | Yelp2018 (Recall) | Yelp2018 (NDCG) | Amazon-Book (Recall) | Amazon-Book (NDCG) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NGCF** | 0.1570 | 0.1327 | 0.0579 | 0.0477 | 0.0344 | 0.0263 |
| **Mult-VAE** | 0.1641 | 0.1335 | 0.0584 | 0.0450 | 0.0407 | 0.0315 |
| **GRMF** | 0.1477 | 0.1205 | 0.0571 | 0.0462 | 0.0354 | 0.0270 |
| **GRMF-norm** | 0.1557 | 0.1261 | 0.0561 | 0.0454 | 0.0352 | 0.0269 |
| **LightGCN** | **0.1830** | **0.1554** | **0.0649** | **0.0530** | **0.0411** | **0.0315** |

LightGCN outperforms all other competitive recommendation baselines, including item-based VAEs and graph-regularized matrix factorization.

---

### 4.4 Ablation and Effectiveness Analyses

#### 4.4.1 Impact of Layer Normalization Schemes
Table 5 shows the performance of 3-layer LightGCN with different normalization configurations:

#### Table 5: Normalization scheme variations.
| Method | Gowalla (Recall) | Gowalla (NDCG) | Yelp2018 (Recall) | Yelp2018 (NDCG) | Amazon-Book (Recall) | Amazon-Book (NDCG) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGCN-L1-L** | 0.1724 | 0.1414 | 0.0630 | 0.0511 | **0.0419** | **0.0320** |
| **LightGCN-L1-R** | 0.1578 | 0.1348 | 0.0587 | 0.0477 | 0.0334 | 0.0259 |
| **LightGCN-L1** | 0.1590 | 0.1319 | 0.0573 | 0.0465 | 0.0361 | 0.0275 |
| **LightGCN-L** | 0.1589 | 0.1317 | 0.0619 | 0.0509 | 0.0383 | 0.0299 |
| **LightGCN-R** | 0.1420 | 0.1156 | 0.0521 | 0.0401 | 0.0252 | 0.0196 |
| **LightGCN** | **0.1830** | **0.1554** | **0.0649** | **0.0530** | 0.0411 | 0.0315 |

Using the symmetric square-root normalization on both sides (default LightGCN) yields the best overall performance.

![Figure 4: Results of LightGCN and LightGCN-single.](images/lightgcn_figure4.png) *(Note: Please place the cropped image of Figure 4 from the PDF under the `images/` directory as `lightgcn_figure4.png`)*

#### 4.4.2 Analysis of Embedding Smoothness
The smoothness loss $S_U$ evaluates the closeness of co-interacted users in the embedding space:

$$S_U = \sum_{u=1}^M \sum_{v=1}^M c_{v \rightarrow u} \left( \frac{\mathbf{e}_u}{\|\mathbf{e}_u\|^2} - \frac{\mathbf{e}_v}{\|\mathbf{e}_v\|^2} \right)^2$$ (Equation 17)

#### Table 6: Smoothness loss of the embeddings.
| Dataset | Gowalla (MF) | Gowalla (LightGCN) | Yelp2018 (MF) | Yelp2018 (LightGCN) | Amazon-Book (MF) | Amazon-Book (LightGCN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **User Smoothness** | 15449.3 | 12872.7 | 16258.2 | 10091.7 | 38034.2 | 32191.1 |
| **Item Smoothness** | 12106.7 | 5829.0 | 16632.1 | 6459.8 | 28307.9 | 16866.0 |

LightGCN exhibits a much lower smoothness loss compared to MF, justifying the value of neighborhood aggregation.

![Figure 5: Performance of 2-layer LightGCN w.r.t. different regularization coefficient.](images/lightgcn_figure5.png) *(Note: Please place the cropped image of Figure 5 from the PDF under the `images/` directory as `lightgcn_figure5.png`)*

---

## 5 Conclusion and Future Work
In this work, we argued the unnecessarily complicated design of GCNs for collaborative filtering. We proposed **LightGCN**, which discards feature transformation and nonlinear activation operations. Instead, it aggregates neighbor representations linearly and combines multiple layer embeddings with a weighted sum. Empirically, LightGCN is easier to train, less prone to overfitting, and achieves significant improvements over NGCF and other state-of-the-art methods.
