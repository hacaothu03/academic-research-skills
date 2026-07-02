# Language Representations Can Be What Recommenders Need: Findings and Potentials

**Authors:** Leheng Sheng, An Zhang, Yi Zhang, Yuxin Chen, Xiang Wang, Tat-Seng Chua
* **National University of Singapore** (Leheng Sheng, An Zhang, Yuxin Chen, Tat-Seng Chua) - `{leheng.sheng@u.nus.edu, anzhang@u.nus.edu, e1143404@u.nus.edu, dcscts@nus.edu.sg}`
* **University of Science and Technology of China** (Yi Zhang, Xiang Wang) - `{zy1230@mail.ustc.edu.cn, xiangwang1223@gmail.com}`
* *An Zhang is the corresponding author.*

**Venue:** Published as a conference paper at ICLR 2025.
**DOI/arXiv:** [arXiv:2407.05441](https://arxiv.org/abs/2407.05441)
**Code:** [https://github.com/LehengTHU/AlphaRec](https://github.com/LehengTHU/AlphaRec)

---

## Abstract
Recent studies empirically indicate that language models (LMs) encode rich world knowledge beyond mere semantics, attracting significant attention across various fields. However, in the recommendation domain, it remains uncertain whether LMs implicitly encode user preference information. Contrary to prevailing understanding that LMs and traditional recommenders learn two distinct representation spaces due to the huge gap in language and behavior modeling objectives, this work re-examines such understanding and explores extracting a recommendation space directly from the language representation space. 

Surprisingly, our findings demonstrate that item representations, when linearly mapped from advanced LM representations, yield superior recommendation performance. This outcome suggests the possible homomorphism between the advanced language representation space and an effective item representation space for recommendation, implying that collaborative signals may be implicitly encoded within LMs. Motivated by the finding of homomorphism, we explore the possibility of designing advanced collaborative filtering (CF) models purely based on language representations without ID-based embeddings. To be specific, we incorporate several crucial components (i.e., a multilayer perceptron (MLP), graph convolution, and contrastive learning (CL) loss function) to build a simple yet effective model, with the language representations of item textual metadata (i.e., title) as the input. 

Empirical results show that such a simple model can outperform leading ID-based CF models on multiple datasets, which sheds light on using language representations for better recommendation. Moreover, we systematically analyze this simple model and find several key features for using advanced language representations: a good initialization for item representations, superior zero-shot recommendation abilities in new datasets, and being aware of user intention. Our findings highlight the connection between language modeling and behavior modeling, which can inspire both natural language processing and recommender system communities.

---

## 1 Introduction
Language models (LMs) have achieved great success across various domains [Vaswani et al., 2017; Devlin et al., 2019; Dubey et al., 2024; OpenAI, 2023], raising a critical question about the knowledge encoded within the language space. Recent studies empirically find that LMs extend beyond semantic understanding to encode comprehensive world knowledge about various domains, such as game states [Li et al., 2023a], lexical attributes [Vulic et al., 2020], and even concepts of space and time [Gurnee & Tegmark, 2023] through language modeling. However, in the domain of recommendation where the integration of LMs is attracting widespread interest [Fan et al., 2023; Li et al., 2023b; Wu et al., 2023a], it remains unclear whether LMs inherently encode relevant information on user preferences and behaviors in the language space.

Currently, one prevailing understanding holds that general LMs and traditional recommenders (e.g., collaborative filtering models [Koren et al., 2009; He et al., 2021]) encode distinct representation spaces — one for language space and the other for behavior space — but they offer the potential to enhance each other in downstream recommendation tasks [Liao et al., 2024]. Specifically:
* **LMs as recommenders**: aligning the language space with the behavior space can significantly improve recommendation performance [Lin et al., 2023a; Vats et al., 2024; Xu et al., 2024]. Various alignment strategies include fine-tuning LMs with user behavior data [Zhang et al., 2023e; Bao et al., 2023; Geng et al., 2022], incorporating embeddings from traditional recommenders as a new modality of LMs [Liao et al., 2024; Zhang et al., 2023f; Yang et al., 2023], and extending the vocabulary of LMs with item tokens [Zhu et al., 2023; Zheng et al., 2023; Zhai et al., 2024].
* **LMs as enhancers**: traditional recommenders benefit from text embeddings [Yuan et al., 2022; 2023; Li et al., 2023c; Hou et al., 2024a; Liu et al., 2024b], semantic and reasoning information [Wei et al., 2024; Ren et al., 2024b], and generated user behaviors [Zhang et al., 2023b;d].

Despite these efforts, explicit studies of the relationship between language and behavior spaces remain largely unexplored.

In this work, we re-examine this prevailing understanding by exploring whether LM-generated language space has inherently encoded user preferences and behaviors. Specifically, we test the possibility of directly deriving a behavior space from the language space — that is, we assess whether the language representations of item text metadata (e.g., titles) generated by LMs can independently predict user behaviors and achieve competitive recommendation performance.

![Figure 1: Linearly mapping item titles in language representation space into behavior space yields superior recommendation performance on Movies & TV.](images/alpharec_figure1.png) *(Note: Please place the cropped image of Figure 1 from the PDF under the `images/` directory as `alpharec_figure1.png`)*

Our empirical observations and findings include:
* **Before linear mapping**, language representation similarities (i.e., semantic textual similarities (STS) [Muennighoff et al., 2023]) may reflect user preference similarities for item contents. As shown in Figure 1c, movies with themes of superheroes and monsters cluster together in both language and behavior spaces.
* **After linear mapping**, language representations are transformed into high-quality behavior representations, which achieve exceptional recommendation performance, as Figure 1b and experimental results in Section 3.2 show. The performance improves as the language model size increases and remains robust to prompt disturbances.
* **Post-mapping language representations** encode user behavioral similarities beyond STS. For instance, movies of distinct textual genres but with high user preference similarities project and cluster together in the behavior space, whereas they are dispersed in the raw language space.

---

## 2 Preliminary

### 2.1 Task Formulation
Collaborative filtering (CF) aims to select item $i \in \mathcal{I}$ that best matches user $u \in \mathcal{U}$'s preferences based on binary interaction behaviors $\mathbf{Y} = [y_{ui}]$, where $y_{ui} = 1$ indicates user $u$ has interacted with item $i$, and $y_{ui} = 0$ otherwise. 

We summarize a common paradigm $\hat{y}_{ui} = s \circ \phi_\theta(\mathbf{x}_u, \mathbf{x}_i)$ involving three components:
1. **Pre-existing features**: $\mathbf{x}_u$ and $\mathbf{x}_i$, which are set as ID information or one-hot encodings.
2. **Representation generator**: $\phi_\theta(\cdot)$ transfers inputs into behavior representations $\mathbf{e}_u$ and $\mathbf{e}_i$, encoding the user behavior patterns.
3. **Scoring function**: $s(\mathbf{e}_u, \mathbf{e}_i)$ quantifies relevance. A widely-used function is cosine similarity:
   
   $$s(\mathbf{e}_u, \mathbf{e}_i) = \frac{\mathbf{e}_u^\top \mathbf{e}_i}{\|\mathbf{e}_u\| \cdot \|\mathbf{e}_i\|}$$

### 2.2 Item Representation Generation
We categorize generators into two types:
* **ID-based generator**: prevailing CF models [Koren et al., 2009; Rendle, 2022; He et al., 2021] typically convert the ID of each item $i$ into one-hot encodings, which are passed through trainable embedding matrices. However, they suffer from poor domain transferability and lack user intention-aware abilities since one-hot IDs carry no semantic meaning.
* **LM-based generator**: uses text metadata of item $i$ (e.g., titles, descriptions) as pre-existing features $\mathbf{x}_i$. A frozen LM is first used to extract $i$'s language representation $\mathbf{z}_i$, and then a trainable projector maps $\mathbf{z}_i$ into the final representation $\mathbf{e}_i$.

---

## 3 Uncovering Collaborative Signals in LMs via Linear Mapping

We address two key research questions:
* **RQ1:** Do LMs inherently encode collaborative signals (user preferences on items) within their representation spaces?
* **RQ2:** Does the presence of such signals scale with model size, and are they robust across different settings?

### 3.1 Linear Mapping
We train a linear mapping matrix $\mathbf{W}$ to project representations from the language space into a behavior space. We use frozen LMs to extract item title representations $\mathbf{z}_i$. To derive user representations, we compute the average of the language representations of the user's historical items:

$$\mathbf{z}_u = \frac{1}{|\mathcal{N}_u|} \sum_{i \in \mathcal{N}_u} \mathbf{z}_i$$

The linear mapping matrix sets behavior representations of user $u$ and item $i$ as:

$$\mathbf{e}_u = \mathbf{W}\mathbf{z}_u \quad \text{and} \quad \mathbf{e}_i = \mathbf{W}\mathbf{z}_i$$

To optimize the matrix $\mathbf{W}$, we employ the InfoNCE loss as the objective function.

### 3.2 Empirical Findings
We test the recommendation performance of this linear mapping method across three Amazon datasets [Ni et al., 2019] compared against classic ID-based CF baselines.

#### Table 1: Linear mapping compared with classical ID-based CF baselines.
| Dataset | Movies & TV (Recall) | Movies & TV (NDCG) | Movies & TV (HR) | Video Games (Recall) | Video Games (NDCG) | Video Games (HR) | Books (Recall) | Books (NDCG) | Books (HR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MF** | 0.0568 | 0.0519 | 0.3377 | 0.0323 | 0.0195 | 0.0864 | 0.0437 | 0.0391 | 0.2476 |
| **MultVAE** | 0.0853 | 0.0776 | 0.4434 | 0.0908 | 0.0531 | 0.2211 | 0.0722 | 0.0597 | 0.3418 |
| **LightGCN** | 0.0849 | 0.0747 | 0.4397 | 0.1007 | 0.0590 | 0.2281 | 0.0723 | 0.0608 | **0.3489** |
| **Linear (BERT)** | 0.0415 | 0.0399 | 0.2362 | 0.0524 | 0.0309 | 0.1245 | 0.0226 | 0.0194 | 0.1240 |
| **Linear (RoBERTa)** | 0.0406 | 0.0387 | 0.2277 | 0.0578 | 0.0338 | 0.1339 | 0.0247 | 0.0209 | 0.1262 |
| **Linear (Llama2-7B)** | 0.1027 | 0.0955 | 0.4952 | 0.1249 | 0.0729 | 0.2746 | 0.0662 | 0.0559 | 0.3176 |
| **Linear (Mistral-7B)** | 0.1039 | 0.0963 | 0.4994 | 0.1270 | 0.0687 | 0.2428 | 0.0650 | 0.0544 | 0.3124 |
| **Linear (ada-v2)** | 0.0926 | 0.0874 | 0.4563 | 0.1176 | 0.0683 | 0.2579 | 0.0515 | 0.0436 | 0.2570 |
| **Linear (3-large)** | 0.1109 | 0.1023 | 0.5200 | 0.1367 | 0.0793 | 0.2928 | 0.0735 | 0.0608 | 0.3355 |
| **Linear (SFR-Mistral)** | **0.1152** | **0.1065** | **0.5327** | **0.1370** | **0.0787** | **0.2927** | **0.0738** | **0.0610** | 0.3371 |

Key Observations:
* **Homomorphism between spaces**: Projecting representations of advanced LMs (e.g. Llama2-7B, OpenAI text-embeddings-3-large, SFR-Embedding-Mistral) consistently performs better than leading ID-based collaborative filtering models (like LightGCN) on most metrics.
* **Model scaling**: Figure 2 shows that as the parameter size of the language model scales from 7B to 70B, the recommendation performance improves consistently, indicating that larger LMs encode user behavioral similarities in a more refined manner.
* **Robustness to prompt noise**: Adding 5-10 random letters to item titles (prompt noise) has minimal impact on recommendation accuracy (Table 2), indicating that the projected collaborative signals are robust.

#### Table 2: Robustness of language representations under prompt disturbances.
| Prompt Style | Movies & TV (Recall) | Movies & TV (NDCG) | Movies & TV (HR) | Video Games (Recall) | Video Games (NDCG) | Video Games (HR) | Books (Recall) | Books (NDCG) | Books (HR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Title + Random Noise** | 0.0952 | 0.0887 | 0.4731 | 0.1213 | 0.0706 | 0.2722 | 0.0632 | 0.0525 | 0.3099 |
| **Title Only** | 0.1027 | 0.0955 | 0.4952 | 0.1249 | 0.0729 | 0.2746 | 0.0662 | 0.0559 | 0.3176 |

---

## 4 Leveraging Language Representations for Better Recommendation

We present **AlphaRec**, a simple yet effective GCN-based recommender model built purely on language representations without ID embeddings.

### 4.1 AlphaRec Model Design
AlphaRec comprises three main components:
1. **Nonlinear projection**: Instead of a linear map, we employ a 2-layer MLP with LeakyReLU to project the raw language representations $\mathbf{z}_i$ and user averaged representations $\mathbf{z}_u$ into initial behavior representations:
   
   $$\mathbf{e}_i^{(0)} = \mathbf{W}_2 \text{LeakyReLU} (\mathbf{W}_1 \mathbf{z}_i + \mathbf{b}_1) + \mathbf{b}_2$$
   
   $$\mathbf{e}_u^{(0)} = \mathbf{W}_2 \text{LeakyReLU} (\mathbf{W}_1 \mathbf{z}_u + \mathbf{b}_1) + \mathbf{b}_2$$ (Equation 1)

2. **Graph convolution**: We run a normalized neighborhood aggregation to capture collaborative patterns from high-order connectivity:
   
   $$\mathbf{e}_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|} \sqrt{|\mathcal{N}_i|}} \mathbf{e}_i^{(k)}, \quad \mathbf{e}_i^{(k+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i|} \sqrt{|\mathcal{N}_u|}} \mathbf{e}_u^{(k)}$$ (Equation 2)

   The final behavior representations are obtained by averaging across layers:
   
   $$\mathbf{e}_u = \frac{1}{K+1} \sum_{k=0}^K \mathbf{e}_u^{(k)}, \quad \mathbf{e}_i = \frac{1}{K+1} \sum_{k=0}^K \mathbf{e}_i^{(k)}$$ (Equation 3)

3. **Contrastive learning objective**: We optimize the network using the InfoNCE loss on cosine similarities without graph data augmentation:
   
   $$\mathcal{L}_{\text{InfoNCE}} = -\sum_{(u,i) \in \mathcal{O}^+} \log \frac{\exp(s(\mathbf{e}_u, \mathbf{e}_i) / \tau)}{\exp(s(\mathbf{e}_u, \mathbf{e}_i) / \tau) + \sum_{j \in \mathcal{S}_u} \exp(s(\mathbf{e}_u, \mathbf{e}_j) / \tau)}$$ (Equation 4)

   where $\tau$ is the temperature hyperparameter and $\mathcal{S}_u$ is a randomly sampled negative item set.

---

### 4.2 Empirical Findings

#### Table 3: Performance comparison of AlphaRec against leading ID-based CF baselines.
| Model | Movies & TV (Recall) | Movies & TV (NDCG) | Movies & TV (HR) | Video Games (Recall) | Video Games (NDCG) | Video Games (HR) | Books (Recall) | Books (NDCG) | Books (HR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MF** | 0.0568 | 0.0519 | 0.3377 | 0.0323 | 0.0195 | 0.0864 | 0.0437 | 0.0391 | 0.2476 |
| **MultVAE** | 0.0853 | 0.0776 | 0.4434 | 0.0908 | 0.0531 | 0.2211 | 0.0722 | 0.0597 | 0.3418 |
| **LightGCN** | 0.0849 | 0.0747 | 0.4397 | 0.1007 | 0.0590 | 0.2281 | 0.0723 | 0.0608 | 0.3489 |
| **SGL** | 0.0916 | 0.0838 | 0.4680 | 0.1089 | 0.0634 | 0.2449 | 0.0789 | 0.0657 | 0.3734 |
| **BC Loss** | 0.1039 | 0.0943 | 0.5037 | 0.1145 | 0.0668 | 0.2561 | 0.0915 | 0.0779 | 0.4045 |
| **XSimGCL** | 0.1057 | 0.0984 | 0.5128 | 0.1138 | 0.0662 | 0.2550 | 0.0879 | 0.0745 | 0.3918 |
| **XSimGCL_t** | 0.1015 | 0.0951 | 0.5016 | 0.1199 | 0.0679 | 0.2674 | 0.0900 | 0.0736 | 0.4036 |
| **KAR** | 0.1084 | 0.1001 | 0.5134 | 0.1181 | 0.0693 | 0.2571 | 0.0852 | 0.0734 | 0.3834 |
| **RLMRec** | 0.1119 | 0.1013 | 0.5301 | 0.1384 | 0.0809 | 0.2997 | 0.0928 | 0.0774 | 0.4092 |
| **EMB-KNN** | 0.0548 | 0.0380 | 0.2916 | 0.0879 | 0.0389 | 0.1970 | 0.0434 | 0.0248 | 0.1851 |
| **AlphaRec** | **0.1221*** | **0.1144*** | **0.5587*** | **0.1519*** | **0.0894*** | **0.3207*** | **0.0991*** | **0.0828*** | **0.4185*** |
| *Imp. %* | *6.79%* | *5.34%* | *2.27%* | *9.12%* | *10.75%* | *5.40%* | *9.75%* | *10.51%* | *7.01%* |

*(Asterisk * denotes statistically significant improvement).*

AlphaRec consistently outperforms the best baseline models across all datasets, achieving a 6.79% to 9.75% recall improvement. 

![Figure 3: Ablation studies showing the impact of MLP, GCN and CL on Books dataset, and training convergence speed comparison.](images/alpharec_figure3.png) *(Note: Please place the cropped image of Figure 3 from the PDF under the `images/` directory as `alpharec_figure3.png`)*

---

## 5 Exploring Potentials of Language Representations for Recommendation

We analyze three key potentials of advanced language representations:

### 5.1 Good Initialization (Potential 1)
Using language representations to initialize item embeddings allows the model to converge at a breakneck speed (Figure 3b), matching or exceeding the speed of the fastest ID-based models.

### 5.2 Zero-Shot Transferability (Potential 2)
Because language features are domain-agnostic, AlphaRec can generalize to entirely unseen datasets with zero ID overlap. We train AlphaRec on a mixed dataset (Books, Movies, Games) and evaluate it zero-shot on three target datasets:

#### Table 4: Zero-shot recommendation performance on entirely new datasets.
| Dataset | Industrial & Scientific (Recall) | Industrial & Scientific (NDCG) | Industrial & Scientific (HR) | MovieLens-1M (Recall) | MovieLens-1M (NDCG) | MovieLens-1M (HR) | Book Crossing (Recall) | Book Crossing (NDCG) | Book Crossing (HR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MF** | 0.0344 | 0.0225 | 0.0521 | 0.1855 | 0.3765 | 0.9634 | 0.0316 | 0.0317 | 0.2382 |
| **MultVAE** | 0.0751 | 0.0459 | 0.1125 | 0.2039 | 0.3741 | 0.9710 | 0.0736 | 0.0634 | 0.3716 |
| **LightGCN** | 0.0785 | 0.0533 | 0.1078 | 0.2019 | **0.4017** | 0.9715 | 0.0630 | 0.0588 | 0.3475 |
| **Random** | 0.0148 | 0.0061 | 0.0248 | 0.0068 | 0.0185 | 0.2611 | 0.0039 | 0.0036 | 0.0443 |
| **Pop** | 0.0216 | 0.0087 | 0.0396 | 0.0253 | 0.0679 | 0.5439 | 0.0119 | 0.0101 | 0.1157 |
| **ZESRec** | 0.0326 | 0.0272 | 0.0628 | 0.0274 | 0.0787 | 0.5786 | 0.0155 | 0.0143 | 0.1347 |
| **UniSRec** | 0.0453 | 0.0350 | 0.0863 | 0.0578 | 0.1412 | 0.7135 | 0.0396 | 0.0332 | 0.2454 |
| **VQ-Rec** | 0.0645 | 0.0410 | 0.0963 | 0.0804 | 0.1921 | 0.8167 | 0.0485 | 0.0492 | 0.2825 |
| **AlphaRec** | **0.0913*** | **0.0573** | **0.1277*** | **0.1486*** | **0.3215*** | **0.9296*** | **0.0660*** | **0.0545*** | **0.3381*** |
| *Imp. %* | *157.09%* | *127.69%* | *30.29%* | *66.67%* | *64.16%* | *37.78%* | *101.55%* | *63.71%* | *47.97%* |

AlphaRec outperforms all other zero-shot baselines by large margins (ranging from 30.29% to 157.09%).

---

### 5.3 Intention-Aware Recommendations (Potential 3)
AlphaRec can perceive text-based user intentions and refine recommendation results. We combine the original user embedding with an intention vector:

$$\tilde{\mathbf{e}}_u^{(0)} = (1 - \alpha) \mathbf{e}_u^{(0)} + \alpha \mathbf{e}_u^{\text{Intention}}$$

where $\alpha$ controls the strength of user intention query relative to historical interests.

#### Table 5: Performance comparison in user intention capture.
| Model | MovieLens-1M (HR@5) | MovieLens-1M (NDCG@5) | Video Games (HR@5) | Video Games (NDCG@5) |
| :--- | :---: | :---: | :---: | :---: |
| **TEM** [Bi et al., 2020] | 0.2738 | 0.1973 | 0.2212 | 0.1425 |
| **AlphaRec (w/o Intention)** | 0.0793 | 0.0498 | 0.0663 | 0.0438 |
| **AlphaRec (w/ Intention)** | **0.4704*** | **0.3738*** | **0.2569*** | **0.1862*** |

![Figure 4: Case study of user intention capture on MovieLens-1M and the effect of intention strength \alpha.](images/alpharec_figure4.png) *(Note: Please place the cropped image of Figure 4 from the PDF under the `images/` directory as `alpharec_figure4.png`)*

---

## 6 Limitations
1. This research lacks theoretical guarantees and does not propose new CF architectures.
2. AlphaRec lacks personalized modeling for the MLP component across users.
3. The user intention-aware experiments use simple fixed weight parameters $\alpha$.

---

## 7 Conclusion
We explored the relationship between language space and behavior spaces in recommendation. Our findings suggest the presence of a homomorphism between advanced LM representations and an effective behavior space. We proposed **AlphaRec**, showing that a simple model utilizing language representations can outperform leading ID-based collaborative filtering models, demonstrating excellent initialization, zero-shot transfer, and user-intention capture capabilities.

---

## Appendix (Key Statistics & Settings)

#### Table 6: Dataset statistics.
| Metric | Books | Movies & TV | Video Games | Industrial & Scientific | MovieLens-1M | Book Crossing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **# Users** | 71,306 | 26,073 | 40,834 | 15,141 | 6,040 | 6,273 |
| **# Items** | 26,073 | 12,464 | 14,344 | 5,163 | 3,043 | 5,335 |
| **# Interactions**| 2,209,030 | 876,027 | 390,013 | 82,578 | 995,492 | 253,057 |
| **Density** | 0.0008 | 0.0026 | 0.0007 | 0.0010 | 0.0542 | 0.0076 |

#### Table 7: Effect of varying language representations in AlphaRec.
| LM Source | Movies & TV (Recall) | Movies & TV (NDCG) | Movies & TV (HR) | Video Games (Recall) | Video Games (NDCG) | Video Games (HR) | Books (Recall) | Books (NDCG) | Books (HR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BERT** | 0.0994 | 0.0923 | 0.4873 | 0.0960 | 0.0550 | 0.2179 | 0.0719 | 0.0607 | 0.3434 |
| **RoBERTa** | 0.0967 | 0.0895 | 0.4793 | 0.0947 | 0.0545 | 0.2167 | 0.0710 | 0.0596 | 0.3386 |
| **Llama2-7B** | 0.1160 | 0.1092 | 0.5388 | 0.1395 | 0.0817 | 0.3003 | 0.0940 | 0.0793 | 0.4081 |
| **Mistral-7B** | 0.1161 | 0.1097 | 0.5421 | 0.1413 | 0.0828 | 0.3020 | 0.0945 | 0.0799 | 0.4090 |
| **ada-v2** | 0.1152 | 0.1083 | 0.5382 | 0.1437 | 0.0844 | 0.3062 | 0.0933 | 0.0784 | 0.4061 |
| **3-large** | **0.1221** | **0.1144** | **0.5587** | **0.1519** | **0.0894** | **0.3207** | **0.0991** | **0.0828** | **0.4185** |
| **SFR-Mistral**| 0.1225 | 0.1139 | 0.5571 | 0.1521 | 0.0887 | 0.3209 | 0.0982 | 0.0820 | 0.4161 |

#### Table 10: Training cost of AlphaRec (seconds per epoch / in total).
| Model | Books | Movies & TV | Video Games | Amazon-Mix |
| :--- | :---: | :---: | :---: | :---: |
| **AlphaRec** | 40.1 / 1363.4 | 12.3 / 479.7 | 7.4 / 214.6 | 107.2 / 5788.8 |

#### Table 15: Hyperparameters ($\tau$) of AlphaRec.
| Dataset | Books | Movies & TV | Video Games | Amazon-Mix |
| :--- | :---: | :---: | :---: | :---: |
| **$\tau$** | 0.15 | 0.15 | 0.2 | 0.15 |

#### Table 16: Cost for extracting language representations on Amazon Movie & TV (per 10,000 items).
| Model | Time Cost | Money Cost (API) |
| :--- | :---: | :---: |
| **Llama2-7B** | 5 mins | $0.00 |
| **text-embedding-3-large** | 40 s | $0.17 |
