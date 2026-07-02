# Neural Graph Collaborative Filtering

**Authors:** Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, Tat-Seng Chua
* **National University of Singapore** (Xiang Wang, Fuli Feng, Tat-Seng Chua) - `{xiangwang@u.nus.edu, fulifeng93@gmail.com, dcscts@nus.edu.sg}`
* **University of Science and Technology of China** (Xiangnan He - Corresponding Author) - `xiangnanhe@gmail.com`
* **Hefei University of Technology** (Meng Wang) - `eric.mengwang@gmail.com`

**Venue:** Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '19), Paris, France.
**DOI:** [10.1145/3331184.3331267](https://doi.org/10.1145/3331184.3331267)

---

## Abstract
Learning vector representations (aka. embeddings) of users and items lies at the core of modern recommender systems. Ranging from early matrix factorization to recently emerged deep learning based methods, existing efforts typically obtain a user’s (or an item’s) embedding by mapping from pre-existing features that describe the user (or the item), such as ID and attributes. We argue that an inherent drawback of such methods is that, the collaborative signal, which is latent in user-item interactions, is not encoded in the embedding process. As such, the resultant embeddings may not be sufficient to capture the collaborative filtering effect.

In this work, we propose to integrate the user-item interactions — more specifically the bipartite graph structure — into the embedding process. We develop a new recommendation framework **Neural Graph Collaborative Filtering (NGCF)**, which exploits the user-item graph structure by propagating embeddings on it. This leads to the expressive modeling of high-order connectivity in user-item graph, effectively injecting the collaborative signal into the embedding process in an explicit manner. We conduct extensive experiments on three public benchmarks, demonstrating significant improvements over several state-of-the-art models like HOP-Rec [40] and Collaborative Memory Network [5]. Further analysis verifies the importance of embedding propagation for learning better user and item representations, justifying the rationality and effectiveness of NGCF. Codes are available at [https://github.com/xiangwang1223/neural_graph_collaborative_filtering](https://github.com/xiangwang1223/neural_graph_collaborative_filtering).

**CCS Concepts:** 
* **Information systems** $\rightarrow$ **Recommender systems**.

**Keywords:** Collaborative Filtering, Recommendation, High-order Connectivity, Embedding Propagation, Graph Neural Network

---

## 1 Introduction
Personalized recommendation is ubiquitous, having been applied to many online services such as E-commerce, advertising, and social media. At its core is estimating how likely a user will adopt an item based on the historical interactions like purchases and clicks. Collaborative filtering (CF) addresses it by assuming that behaviorally similar users would exhibit similar preference on items. To implement the assumption, a common paradigm is to parameterize users and items for reconstructing historical interactions, and predict user preference based on the parameters [1, 14].

Generally speaking, there are two key components in learnable CF models:
1. **Embedding**: which transforms users and items to vectorized representations.
2. **Interaction modeling**: which reconstructs historical interactions based on the embeddings.

For example, matrix factorization (MF) directly embeds user/item ID as a vector and models user-item interaction with inner product [20]; collaborative deep learning extends the MF embedding function by integrating the deep representations learned from rich side information of items [30]; neural collaborative filtering models replace the MF interaction function of inner product with nonlinear neural networks [14]; and translation-based CF models instead use Euclidean distance metric as the interaction function [28], among others.

Despite their effectiveness, we argue that these methods are not sufficient to yield satisfactory embeddings for CF. The key reason is that the embedding function lacks an explicit encoding of the crucial collaborative signal, which is latent in user-item interactions to reveal the behavioral similarity between users (or items). To be more specific, most existing methods build the embedding function with the descriptive features only (e.g., ID and attributes), without considering the user-item interactions — which are only used to define the objective function for model training [26, 28]. As a result, when the embeddings are insufficient in capturing CF, the methods have to rely on the interaction function to make up for the deficiency of suboptimal embeddings [14].

While intuitively useful to integrate user-item interactions into the embedding function, it is non-trivial to do it well. In particular, the scale of interactions can easily reach millions or even larger in real applications, making it difficult to distill the desired collaborative signal. In this work, we tackle the challenge by exploiting the **high-order connectivity** from user-item interactions, a natural way that encodes collaborative signal in the interaction graph structure.

### Running Example
![Figure 1: An illustration of the user-item interaction graph and the high-order connectivity.](images/ngcf_figure1.png) *(Note: Please place the cropped image of Figure 1 from the PDF under the `images/` directory as `ngcf_figure1.png`)*

Figure 1 illustrates the concept of high-order connectivity. The user of interest for recommendation is $u_1$, labeled with the double circle in the left subfigure of user-item interaction graph. The right subfigure shows the tree structure that is expanded from $u_1$. The high-order connectivity denotes the path that reaches $u_1$ from any node with the path length $l$ larger than $1$. Such high-order connectivity contains rich semantics that carry collaborative signal. 

For example:
* The path $u_1 \leftarrow i_2 \leftarrow u_2$ indicates the behavior similarity between $u_1$ and $u_2$, as both users have interacted with $i_2$.
* The longer path $u_1 \leftarrow i_2 \leftarrow u_2 \leftarrow i_4$ suggests that $u_1$ is likely to adopt $i_4$, since her similar user $u_2$ has consumed $i_4$ before.
* From the holistic view of $l = 3$, item $i_4$ is more likely to be of interest to $u_1$ than item $i_5$, since there are two paths connecting $\langle i_4, u_1 \rangle$, while only one path connects $\langle i_5, u_1 \rangle$.

### Present Work
We propose to model the high-order connectivity information in the embedding function. Instead of expanding the interaction graph as a tree which is complex to implement, we design a neural network method to propagate embeddings recursively on the graph. This is inspired by the recent developments of graph neural networks [8, 32, 38], which can be seen as constructing information flows in the embedding space.

Specifically, we devise an **embedding propagation layer**, which refines a user’s (or an item’s) embedding by aggregating the embeddings of the interacted items (or users). By stacking multiple embedding propagation layers, we can enforce the embeddings to capture the collaborative signal in high-order connectivities. 

To summarize, this work makes the following main contributions:
* We highlight the critical importance of explicitly exploiting the collaborative signal in the embedding function of model-based CF methods.
* We propose NGCF, a new recommendation framework based on graph neural network, which explicitly encodes the collaborative signal in the form of high-order connectivities by performing embedding propagation.
* We conduct empirical studies on three million-size datasets, demonstrating the state-of-the-art performance of NGCF and its effectiveness in improving the embedding quality with neural embedding propagation.

---

## 2 Methodology

The architecture of the proposed NGCF model consists of three components:
1. **An embedding layer** that offers initialization of user embeddings and item embeddings.
2. **Multiple embedding propagation layers** that refine the embeddings by injecting high-order connectivity relations.
3. **A prediction layer** that aggregates the refined embeddings from different propagation layers and outputs the affinity score of a user-item pair.

### 2.1 Embedding Layer
Following mainstream recommender models [1, 14, 26], we describe a user $u$ (an item $i$) with an embedding vector $\mathbf{e}_u \in \mathbb{R}^d$ ($\mathbf{e}_i \in \mathbb{R}^d$), where $d$ denotes the embedding size. This can be seen as building a parameter matrix as an embedding look-up table:

$$\mathbf{E} = [\mathbf{e}_{u_1}, \dots, \mathbf{e}_{u_N}, \mathbf{e}_{i_1}, \dots, \mathbf{e}_{i_M}]$$ (Equation 1)

This embedding table serves as an initial state for user embeddings and item embeddings, to be optimized in an end-to-end fashion. In traditional models like MF and Neural CF [14], these ID embeddings are directly fed into an interaction layer to achieve the prediction score. In contrast, in NGCF, we refine the embeddings by propagating them on the user-item interaction graph, step-by-step injecting collaborative signals.

### 2.2 Embedding Propagation Layers
We build upon the message-passing architecture of GNNs [8, 38] to capture CF signals along the graph structure.

#### 2.2.1 First-order Propagation
Interacted items provide direct evidence of a user’s preference. We perform embedding propagation between connected users and items via two major operations: **message construction** and **message aggregation**.

##### Message Construction
For a connected user-item pair $(u, i)$, the message from $i$ to $u$ is defined as:

$$\mathbf{m}_{u \leftarrow i} = f(\mathbf{e}_i, \mathbf{e}_u, p_{ui})$$ (Equation 2)

where $\mathbf{m}_{u \leftarrow i}$ is the message embedding, $f(\cdot)$ is the message encoding function, and $p_{ui}$ is the decay factor controlling the discount of propagation on edge $(u, i)$. We implement $f(\cdot)$ as:

$$\mathbf{m}_{u \leftarrow i} = \frac{1}{\sqrt{|\mathcal{N}_u| |\mathcal{N}_i|}} \left( \mathbf{W}_1 \mathbf{e}_i + \mathbf{W}_2 (\mathbf{e}_i \odot \mathbf{e}_u) \right)$$ (Equation 3)

where $\mathbf{W}_1, \mathbf{W}_2 \in \mathbb{R}^{d' \times d}$ are trainable weight matrices to distill useful information, and $d'$ is the transformation size. Unlike GCNs which only consider the neighbor's contribution ($\mathbf{e}_i$), we encode the interaction between $\mathbf{e}_i$ and $\mathbf{e}_u$ via element-wise product $\odot$, making the message dependent on the affinity between user and item. 

We set $p_{ui}$ as the graph Laplacian norm:

$$p_{ui} = \frac{1}{\sqrt{|\mathcal{N}_u| |\mathcal{N}_i|}}$$

where $\mathcal{N}_u$ and $\mathcal{N}_i$ denote the first-hop neighbors of user $u$ and item $i$, respectively.

##### Message Aggregation
We aggregate the messages propagated from $u$'s neighborhood to refine $u$'s representation:

$$\mathbf{e}_u^{(1)} = \text{LeakyReLU} \left( \mathbf{m}_{u \leftarrow u} + \sum_{i \in \mathcal{N}_u} \mathbf{m}_{u \leftarrow i} \right)$$ (Equation 4)

where $\mathbf{e}_u^{(1)}$ denotes the representation of user $u$ obtained after the first embedding propagation layer. The self-connection message is defined as $\mathbf{m}_{u \leftarrow u} = \mathbf{W}_1 \mathbf{e}_u$. Analogously, we obtain the representation $\mathbf{e}_i^{(1)}$ for item $i$ by propagating information from its connected users.

#### 2.2.2 High-order Propagation
By stacking $l$ embedding propagation layers, user and item representations can receive messages propagated from their $l$-hop neighbors. The recursive formulation of user $u$'s representation at the $l$-th step is:

$$\mathbf{e}_u^{(l)} = \text{LeakyReLU} \left( \mathbf{m}_{u \leftarrow u}^{(l)} + \sum_{i \in \mathcal{N}_u} \mathbf{m}_{u \leftarrow i}^{(l)} \right)$$ (Equation 5)

where the messages are defined as:

$$\mathbf{m}_{u \leftarrow i}^{(l)} = \frac{1}{\sqrt{|\mathcal{N}_u| |\mathcal{N}_i|}} \left( \mathbf{W}_1^{(l)} \mathbf{e}_i^{(l-1)} + \mathbf{W}_2^{(l)} (\mathbf{e}_i^{(l-1)} \odot \mathbf{e}_u^{(l-1)}) \right)$$

$$\mathbf{m}_{u \leftarrow u}^{(l)} = \mathbf{W}_1^{(l)} \mathbf{e}_u^{(l-1)}$$ (Equation 6)

#### Propagation Rule in Matrix Form
To facilitate batch implementation, the matrix-form of the layer-wise propagation rule is:

$$\mathbf{E}^{(l)} = \text{LeakyReLU} \left( (\mathbf{L} + \mathbf{I}) \mathbf{E}^{(l-1)} \mathbf{W}_1^{(l)} + \mathbf{L} \mathbf{E}^{(l-1)} \odot \mathbf{E}^{(l-1)} \mathbf{W}_2^{(l)} \right)$$ (Equation 7)

where $\mathbf{E}^{(l)} \in \mathbb{R}^{(N+M) \times d_l}$ represents user and item embeddings at layer $l$, $\mathbf{I}$ is the identity matrix, and $\mathbf{L}$ is the symmetric normalized Laplacian matrix for the user-item interaction graph:

$$\mathbf{L} = \mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}} \quad \text{and} \quad \mathbf{A} = \begin{bmatrix} \mathbf{0} & \mathbf{R} \\ \mathbf{R}^\top & \mathbf{0} \end{bmatrix}$$ (Equation 8)

where $\mathbf{R} \in \mathbb{R}^{N \times M}$ is the user-item interaction matrix, $\mathbf{0}$ is an all-zero matrix, and $\mathbf{D}$ is the diagonal degree matrix.

### 2.3 Model Prediction
After propagating with $L$ layers, we obtain multiple representation vectors for user $u$, namely $\{\mathbf{e}_u^{(1)}, \dots, \mathbf{e}_u^{(L)}\}$. We concatenate them to constitute the final embedding for a user (and do the same for items):

$$\mathbf{e}_u^* = \mathbf{e}_u^{(0)} \parallel \mathbf{e}_u^{(1)} \parallel \dots \parallel \mathbf{e}_u^{(L)}, \quad \mathbf{e}_i^* = \mathbf{e}_i^{(0)} \parallel \mathbf{e}_i^{(1)} \parallel \dots \parallel \mathbf{e}_i^{(L)}$$ (Equation 9)

where $\parallel$ is the concatenation operation. Finally, we conduct the inner product to estimate the user's preference towards the target item:

$$\hat{y}_{\text{NGCF}}(u,i) = {\mathbf{e}_u^*}^\top \mathbf{e}_i^*$$ (Equation 10)

---

### 2.4 Optimization
To learn model parameters, we optimize the pairwise BPR loss [26] which assumes that observed interactions should be assigned higher prediction values than unobserved ones:

$$\text{Loss} = \sum_{(u,i,j) \in \mathcal{O}} -\ln \sigma\left(\hat{y}_{ui} - \hat{y}_{uj}\right) + \lambda \|\Theta\|_2^2$$ (Equation 11)

where $\mathcal{O} = \{(u, i, j) \mid (u, i) \in \mathcal{R}^+, (u, j) \in \mathcal{R}^-\}$ denotes the pairwise training data, $\sigma(\cdot)$ is the sigmoid function, and $\Theta = \{\mathbf{E}, \{\mathbf{W}_1^{(l)}, \mathbf{W}_2^{(l)}\}_{l=1}^L\}$ denotes all trainable parameters. Mini-batch Adam [17] is used for optimization.

#### 2.4.1 Model Size
NGCF only introduces very few parameters relative to matrix factorization (MF) — specifically two small weight matrices $\mathbf{W}_1^{(l)}$ and $\mathbf{W}_2^{(l)}$ of size $d_l \times d_{l-1}$ per propagation layer. For Gowalla dataset, MF has 4.5 million parameters, while NGCF adds only 0.024 million parameters (using 3 layers of size 64).

#### 2.4.2 Message and Node Dropout
To prevent overfitting, we use two dropout techniques during training:
1. **Message Dropout**: randomly drops outgoing messages with a probability $p_1$ in Equation (6).
2. **Node Dropout**: randomly blocks particular nodes and discards all their outgoing messages with a probability $p_2$ by removing rows/columns from the Laplacian matrix $\mathbf{L}$.

---

### 2.5 Discussions
#### 2.5.1 NGCF Generalizes SVD++
SVD++ [19] can be viewed as a special case of NGCF with no high-order propagation ($L=1$). Disabling the transformation matrices and nonlinear activation functions, the prediction is formulated as:

$$\hat{y}_{\text{NGCF-SVD}} = \left(\mathbf{e}_u + \sum_{i' \in \mathcal{N}_u} p_{ui'} \mathbf{e}_{i'} \right)^\top \left(\mathbf{e}_i + \sum_{u' \in \mathcal{N}_i} p_{iu'} \mathbf{e}_{u'} \right)$$ (Equation 12)

By setting $p_{ui'}$ as $1/\sqrt{|\mathcal{N}_u|}$ and $p_{iu'} = 0$ respectively, we recover the exact SVD++ model.

#### 2.5.2 Time Complexity Analysis
For the $l$-th propagation layer, matrix multiplication has a complexity of $O(|\mathcal{R}^+| d_l d_{l-1})$ where $|\mathcal{R}^+|$ is the number of interactions. The overall complexity for evaluating NGCF is $O(\sum_{l=1}^L |\mathcal{R}^+| d_l d_{l-1} + \sum_{l=1}^L |\mathcal{R}^+| d_l)$. Empirically, training NGCF on Gowalla takes ~80s per epoch compared to ~20s for MF.

---

## 3 Related Work
### 3.1 Model-based CF Methods
Modern recommendation methods [5, 14, 33] parameterize users and items by vectorized representations. While traditional MF uses linear inner products, recent works use deep neural networks to capture nonlinear interactions (e.g. NeuMF [14], LRML [28]). However, they lack explicit encoding of the transitive collaborative filtering signals in the embedding space.

### 3.2 Graph-Based CF Methods
Early methods like ItemRank [7] and BiRank [12] use label propagation on the user-item graph. HOP-Rec [40] combines graph random walks to enrich training data for MF, but does not modify the embedding function itself.

### 3.3 Graph Convolutional Networks
GC-MC [29] applies GCNs to the user-item graph, but only employs a single convolution layer. PinSage [42] uses multiple graph convolution layers on an item-item graph for Pinterest. SpectralCF [43] proposes spectral convolutions but suffers from high computational complexity due to eigen-decomposition.

---

## 4 Experiments

### 4.1 Dataset Description
We evaluate NGCF on three public datasets: **Gowalla**, **Yelp2018**, and **Amazon-Book**. The statistics are shown below:

| Dataset | #Users | #Items | #Interactions | Density |
| :--- | :---: | :---: | :---: | :---: |
| **Gowalla** | 29,858 | 40,981 | 1,027,370 | 0.00084 |
| **Yelp2018** | 31,668 | 38,048 | 1,561,406 | 0.00130 |
| **Amazon-Book** | 52,643 | 91,599 | 2,984,108 | 0.00062 |

*   **Training/Test Split:** 80% of historical interactions are randomly selected for training, and the remaining 20% for testing. 10% of the training set is used as validation.
*   **Evaluation Metrics:** Recall@K and NDCG@K (default $K=20$).

---

### 4.2 Baselines
We compare NGCF against:
1. **MF [26]**: Matrix Factorization optimized by BPR loss.
2. **NeuMF [14]**: Neural Collaborative Filtering combining GMF and MLP.
3. **CMN [5]**: Collaborative Memory Network exploiting neighborhood users.
4. **HOP-Rec [40]**: Graph-based model augmenting MF with random walks.
5. **GC-MC [29]**: GCN-based model utilizing first-order neighbors.
6. **PinSage [42]**: GraphSAGE applied to recommendation.

---

### 4.3 Performance Comparison (RQ1)

#### 4.3.1 Overall Comparison
The overall performance comparison is presented in Table 2:

| Model | Gowalla (Recall@20) | Gowalla (NDCG@20) | Yelp2018 (Recall@20) | Yelp2018 (NDCG@20) | Amazon-Book (Recall@20) | Amazon-Book (NDCG@20) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MF** | 0.1291 | 0.1109 | 0.0433 | 0.0354 | 0.0250 | 0.0196 |
| **NeuMF** | 0.1399 | 0.1212 | 0.0451 | 0.0363 | 0.0258 | 0.0200 |
| **CMN** | 0.1405 | 0.1221 | 0.0457 | 0.0369 | 0.0267 | 0.0218 |
| **HOP-Rec** | 0.1399 | 0.1214 | 0.0517 | 0.0428 | **0.0309** | **0.0232** |
| **GC-MC** | 0.1395 | 0.1204 | 0.0462 | 0.0379 | 0.0288 | 0.0224 |
| **PinSage** | 0.1380 | 0.1196 | 0.0471 | 0.0393 | 0.0282 | 0.0219 |
| **NGCF-3** | **0.1569*** | **0.1327*** | **0.0579*** | **0.0477*** | **0.0337*** | **0.0261*** |
| *% Improv.* | *11.68%* | *8.64%* | *11.97%* | *11.29%* | *9.61%* | *12.50%* |

*(Asterisk * denotes statistically significant improvement).*

Key findings:
* **NGCF-3** consistently performs best across all datasets, improving over the strongest baselines by up to 12.5% in NDCG.
* Models utilizing graph connections (GC-MC, HOP-Rec, PinSage, NGCF) generally outperform standard MF and NeuMF, validating the value of exploiting structural connectivity.

#### 4.3.2 Sparsity Performance
By partitioning users based on interaction sparsity levels, NGCF yields more significant improvements for inactive users (e.g. 8.49% improvement for users with $<24$ interactions in Gowalla), demonstrating that embedding propagation effectively mitigates the cold-start/sparsity problem.

---

### 4.4 Study of NGCF (RQ2)

#### 4.4.1 Effect of Layer Numbers
Table 3 shows the effect of the number of embedding propagation layers $L$:

| Model | Gowalla (Recall) | Gowalla (NDCG) | Yelp2018 (Recall) | Yelp2018 (NDCG) | Amazon-Book (Recall) | Amazon-Book (NDCG) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NGCF-1** | 0.1556 | 0.1315 | 0.0543 | 0.0442 | 0.0313 | 0.0241 |
| **NGCF-2** | 0.1547 | 0.1307 | 0.0566 | 0.0465 | 0.0330 | 0.0254 |
| **NGCF-3** | 0.1569 | 0.1327 | **0.0579** | **0.0477** | 0.0337 | 0.0261 |
| **NGCF-4** | **0.1570** | **0.1327** | 0.0566 | 0.0461 | **0.0344** | **0.0263** |

* Stacking up to 3 layers consistently improves performance. Adding a 4th layer can lead to overfitting (e.g. on Yelp2018).

#### 4.4.2 Effect of Embedding Propagation Type
Table 4 compares different convolution layers:

| Model | Gowalla (Recall) | Gowalla (NDCG) | Yelp2018 (Recall) | Yelp2018 (NDCG) | Amazon-Book (Recall) | Amazon-Book (NDCG) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NGCF-1** | **0.1556** | **0.1315** | **0.0543** | **0.0442** | **0.0313** | **0.0241** |
| **NGCF-1_SVD++** | 0.1517 | 0.1265 | 0.0504 | 0.0414 | 0.0297 | 0.0232 |
| **NGCF-1_GC-MC** | 0.1523 | 0.1307 | 0.0518 | 0.0420 | 0.0305 | 0.0234 |
| **NGCF-1_PinSage** | 0.1534 | 0.1308 | 0.0516 | 0.0420 | 0.0293 | 0.0231 |

* Incorporating the interactive feature representation ($\mathbf{e}_u \odot \mathbf{e}_i$) makes NGCF-1 superior to all other variants.

---

### 4.5 Effect of High-order Connectivity (RQ3)
Using t-SNE, we visualize user and item representations. In **NGCF-3**, embeddings show noticeable clustering where users and their relevant items form tightly clustered groups compared to MF (NGCF-0), proving that high-order propagation explicitly brings related nodes closer in the embedding space.

---

## 5 Conclusion and Future Work
We proposed **Neural Graph Collaborative Filtering (NGCF)**, integrating user-item interactions directly into the embedding function. We designed an embedding propagation layer to harvest collaborative signals from high-order connectivities. 

In future work, we plan to improve NGCF by incorporating attention mechanisms (e.g. to learn variable weights for neighbors) and exploring adversarial training to enhance robustness.

---

## References
1. Yixin Cao, et al. 2019. Unifying Knowledge Graph Learning and Recommendation. In *WWW*.
2. Jingyuan Chen, et al. 2017. Attentive Collaborative Filtering. In *SIGIR*.
3. Zhiyong Cheng, et al. 2018. Aspect-Aware Latent Factor Model. In *WWW*.
4. Michaël Defferrard, et al. 2016. Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. In *NeurIPS*.
5. Travis Ebesu, et al. 2018. Collaborative Memory Network for Recommendation Systems. In *SIGIR*.
6. Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of training deep feedforward neural networks. In *AISTATS*.
7. Marco Gori and Augusto Pucci. 2007. ItemRank: A Random-Walk Based Scoring Algorithm for Recommender Engines. In *IJCAI*.
8. William L. Hamilton, et al. 2017. Inductive Representation Learning on Large Graphs. In *NeurIPS*.
9. Ruining He and Julian McAuley. 2016. Ups and Downs. In *WWW*.
10. Ruining He and Julian McAuley. 2016. VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. In *AAAI*.
11. Xiangnan He and Tat-Seng Chua. 2017. Neural Factorization Machines for Sparse Predictive Analytics. In *SIGIR*.
12. Xiangnan He, et al. 2017. BiRank: Towards Ranking on Bipartite Graphs. *TKDE* 29, 1.
13. Xiangnan He, et al. 2018. Adversarial Personalized Ranking for Recommendation. In *SIGIR*.
14. Xiangnan He, et al. 2017. Neural Collaborative Filtering. In *WWW*.
15. Cheng-Kang Hsieh, et al. 2017. Collaborative Metric Learning. In *WWW*.
16. Santosh Kabbur, et al. 2013. FISM: factored item similarity models for top-N recommender systems. In *KDD*.
17. Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In *ICLR*.
18. Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In *ICLR*.
19. Yehuda Koren. 2008. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In *KDD*.
20. Yehuda Koren, et al. 2009. Matrix Factorization Techniques for Recommender Systems. *IEEE Computer* 42, 8.
21. Dawen Liang, et al. 2016. Modeling User Exposure in Recommendation. In *WWW*.
22. Zhenguang Liu, et al. 2017. FastShrinkage. In *MM*.
23. Andrew L Maas, et al. 2013. Rectifier nonlinearities improve neural network acoustic models. In *ICML*.
24. Athanasios N Nikolakopoulos and George Karypis. 2019. RecWalk. (2019).
25. Jiezhong Qiu, et al. 2018. DeepInf: Social Influence Prediction with Deep Learning. In *KDD*.
26. Steffen Rendle, et al. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In *UAI*.
27. Xuemeng Song, et al. 2018. Neural Compatibility Modeling with Attentive Knowledge Distillation. In *SIGIR*.
28. Yi Tay, et al. 2018. Latent relational metric learning via memory-based attention. In *WWW*.
29. Rianne van den Berg, et al. 2017. Graph Convolutional Matrix Completion. In *KDD*.
30. Hao Wang, et al. 2015. Collaborative Deep Learning for Recommender Systems. In *KDD*.
31. Jun Wang, et al. 2017. IRGAN. In *SIGIR*.
32. Xiang Wang, et al. 2019. KGAT: Knowledge Graph Attention Network for Recommendation. In *KDD*.
33. Xiang Wang, et al. 2018. TEM. In *WWW*.
34. Xiang Wang, et al. 2017. Item Silk Road. In *SIGIR*.
35. Xiang Wang, et al. 2019. Explainable Reasoning over Knowledge Graphs. In *AAAI*.
36. Yao Wu, et al. 2016. Collaborative Denoising Auto-Encoders. In *WSDM*.
37. Xin Xin, et al. 2019. Relational Collaborative Filtering. In *SIGIR*.
38. Keyulu Xu, et al. 2018. Representation Learning on Graphs with Jumping Knowledge Networks. In *ICML*.
39. Feng Xue, et al. 2019. Deep Item-based Collaborative Filtering. *TOIS* 37, 3 (2019).
40. Jheng-Hong Yang, et al. 2018. HOP-rec: high-order proximity for implicit recommendation. In *RecSys*.
41. Xun Yang, et al. 2019. Interpretable Fashion Matching with Rich Attributes. In *SIGIR*.
42. Rex Ying, et al. 2018. Graph Convolutional Neural Networks for Web-Scale Recommender Systems. In *KDD*.
43. Lei Zheng, et al. 2018. Spectral collaborative filtering. In *RecSys*.
