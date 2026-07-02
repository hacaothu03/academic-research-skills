# Tổng quan tài liệu: Sự tiến hóa của Collaborative Filtering dựa trên đồ thị — từ NGCF đến AlphaRec

> Phạm vi: bài tổng quan so sánh bốn công trình cột mốc — **NGCF** (SIGIR '19), **LightGCN** (SIGIR '20), **LightGCN++** (RecSys '24) và **AlphaRec** (ICLR 2025) — nhằm truy vết một mạch phát triển liên tục của Graph-based Collaborative Filtering (CF) và một bước "đổi hệ quy chiếu" ở cuối. Mọi nhận định đều dựa trên nội dung bốn tài liệu được cung cấp.

---

## 1. Bối cảnh và câu hỏi xuyên suốt

Collaborative Filtering hiện đại quy về hai thành phần (theo cách phát biểu của Wang et al. trong NGCF): (i) **embedding** — biến user/item thành vector, và (ii) **interaction modeling** — tái dựng tương tác lịch sử từ embedding. Matrix Factorization (MF) là dạng sơ khai: ánh xạ thẳng ID → embedding rồi dùng inner product.

Bốn công trình dưới đây có thể đọc như bốn câu trả lời nối tiếp cho cùng một câu hỏi: **embedding của user/item nên được sinh ra như thế nào để nắm bắt "tín hiệu cộng tác" (collaborative signal) tốt nhất?**

- NGCF: *hãy nhồi tín hiệu cộng tác vào embedding bằng lan truyền trên đồ thị.*
- LightGCN: *phần lớn cỗ máy GNN là thừa — hãy đơn giản hóa đến mức tối thiểu.*
- LightGCN++: *bản "tối giản" đó lại quá cứng nhắc và thiếu nhất quán — hãy thêm lại đúng vài bậc tự do.*
- AlphaRec: *có thể ta không cần ID-embedding chút nào — biểu diễn ngôn ngữ từ LM đã ngầm chứa tín hiệu cộng tác.*

---

## 2. NGCF (2019) — Đưa high-order connectivity vào hàm embedding

**Luận điểm cốt lõi.** Các phương pháp trước (MF, NeuMF, translation-based CF) xây embedding chỉ từ đặc trưng mô tả (ID/thuộc tính); tương tác user-item chỉ được dùng để định nghĩa hàm mục tiêu, **không** được mã hóa *tường minh* vào embedding. NGCF lập luận rằng chính điều này làm embedding không đủ sức nắm bắt CF, buộc hàm tương tác phải "gánh" phần thiếu hụt.

**Cơ chế.** NGCF khai thác **high-order connectivity** trên đồ thị hai phía (bipartite) user-item bằng cách lan truyền embedding theo kiến trúc message-passing của GNN. Mỗi *embedding propagation layer* gồm:

- **Message construction** có thêm số hạng tương tác phần tử `W₁·eᵢ + W₂·(eᵢ ⊙ eᵤ)`, chuẩn hóa bằng hệ số Laplacian `1/√(|Nᵤ||Nᵢ|)`. Số hạng `eᵢ ⊙ eᵤ` khiến message phụ thuộc *độ tương hợp* giữa user và item (khác GCN thuần chỉ dùng `eᵢ`).
- **Message aggregation** phi tuyến (LeakyReLU) cộng thêm self-connection.
- **Prediction**: nối (concatenate) embedding của mọi tầng `eᵤ* = eᵤ⁽⁰⁾ ‖ … ‖ eᵤ⁽ᴸ⁾` rồi inner product.

**Kết quả và hạn chế nội tại.** NGCF vượt các baseline mạnh (HOP-Rec, GC-MC, PinSage) tới ~12.5% NDCG trên Gowalla/Yelp2018/Amazon-Book, và cải thiện rõ cho user thưa (cold-start). Tuy nhiên NGCF **thừa kế nguyên xi** feature transformation (`W₁, W₂`) và activation phi tuyến từ GCN — vốn thiết kế cho node classification trên đồ thị có thuộc tính phong phú. Đây chính là điểm mà công trình kế tiếp tấn công.

---

## 3. LightGCN (2020) — Đơn giản hóa bằng phương pháp loại trừ (ablation)

**Đóng góp phương pháp luận.** He et al. thực hiện ablation có kiểm soát trên chính NGCF, tạo ba biến thể: NGCF-f (bỏ feature transformation), NGCF-n (bỏ phi tuyến), NGCF-fn (bỏ cả hai). Phát hiện phản trực giác nhưng vững chắc: **bỏ feature transformation cải thiện đáng kể; bỏ cả hai (NGCF-fn) cho mức cải thiện lớn nhất** (~9.57% recall). Nguyên nhân được chẩn đoán là *khó huấn luyện*, không phải overfitting — vì mỗi node chỉ là một ID one-hot vô nghĩa ngữ nghĩa, các phép biến đổi phi tuyến nhiều tầng không mang lại lợi ích mà chỉ làm tối ưu hóa khó hơn.

**Mô hình.** LightGCN giữ lại **duy nhất** thành phần cốt lõi của GCN — *neighborhood aggregation* tuyến tính:

- **Light Graph Convolution (LGC)**: `eᵤ⁽ᵏ⁺¹⁾ = Σ_{i∈Nᵤ} 1/(√|Nᵤ|·√|Nᵢ|) · eᵢ⁽ᵏ⁾`. Không `W`, không activation, **không** cả self-connection.
- **Layer combination**: `eᵤ = Σ_{k=0}^K αₖ·eᵤ⁽ᵏ⁾`, mặc định `αₖ = 1/(K+1)` (mean pooling). Cơ chế này thay thế self-connection, chống over-smoothing, và trộn ngữ nghĩa của các bậc lân cận khác nhau.
- Tham số huấn luyện **chỉ còn** embedding tầng 0. Prediction vẫn là inner product; loss vẫn là BPR.

**Định vị lý thuyết.** LightGCN được liên hệ với SGCN (layer combination ≈ đưa self-connection vào ma trận kề) và APPNP (chống over-smoothing kiểu Personalized PageRank). Thực nghiệm: cải thiện trung bình ~16% recall / ~16.9% NDCG so với NGCF trên cùng data split, dễ huấn luyện hơn, ít overfit hơn.

**Ý nghĩa lịch sử.** LightGCN trở thành *backbone* mặc định của cả một dòng nghiên cứu (social, bundle, multimedia, contrastive-learning CF), và chính sự phổ biến này làm nảy sinh nhu cầu "soi lại" nó — dẫn tới LightGCN++.

---

## 4. LightGCN++ (2024) — Xét lại: sự cứng nhắc và thiếu nhất quán ẩn giấu

**Góc tiếp cận mới.** Lee, Kim & Shin (KAIST) không đề xuất kiến trúc mới mà **phân tích sâu cơ chế** của LightGCN trên dữ liệu thực, và phát hiện ba vấn đề bị che khuất bởi vẻ ngoài "gọn đẹp":

1. **Tác dụng kép của chuẩn hóa.** Số hạng `1/√(|Nᵢ||Nᵤ|)` tách thành hai vai trò: `1/√|Nᵢ|` (trái) *co giãn norm* của embedding tổng hợp; `1/√|Nᵤ|` (phải) *quyết định ảnh hưởng* của từng láng giềng. Điểm mấu chốt: ảnh hưởng thực (effective weight) của láng giềng `u` là `‖eᵤ⁽ᵏ⁾‖ / √|Nᵤ|` — bao gồm cả ảnh hưởng *ngầm* qua norm embedding, chứ không chỉ số hạng tường minh.

2. **Quan sát thực nghiệm & hệ quả.** Trên dữ liệu thực, norm của embedding tổng hợp gần như *tỉ lệ tuyến tính với số láng giềng* → suy ra `‖eᵢ⁽ᵏ⁾‖ ∝̃ √|Nᵢ|` với `k ≥ 1`. Điều này gây ra:
   - **Inflexibility (cứng nhắc)**: effective weight `∝̃ √|Nᵤ|/√|Nᵤ| = 1` → mọi láng giềng bị gộp với trọng số gần **đồng đều** khi `k ≥ 1`, bất chấp bậc của chúng — trái với kỳ vọng rằng láng giềng bậc cao nên ảnh hưởng ít hơn.
   - **Inconsistency (thiếu nhất quán)**: quy luật norm `∝̃ √|Nᵢ|` chỉ đúng với `k ≥ 1`; embedding tầng `k=0` không chịu chuẩn hóa/tổng hợp → lệch chuẩn giữa tầng 0 và các tầng sau, khiến mean pooling với tỉ lệ cố định `1:K` trở nên dưới tối ưu.

**Phương thuốc — thêm lại đúng ba bậc tự do (nhưng không thêm tham số huấn luyện):**

- **Aggregation**: `eᵢ⁽ᵏ⁺¹⁾ = 1/|Nᵢ|^α · Σ_{u∈Nᵢ} 1/|Nᵤ|^β · eᵤ⁽ᵏ⁾/‖eᵤ⁽ᵏ⁾‖^γ`. `α` điều khiển co giãn norm (`‖eᵢ⁽ᵏ⁾‖ ∝̃ |Nᵢ|^{1−α}`); `β` nghiêng trọng số về láng giềng bậc thấp (`β>0`) hay bậc cao (`β<0`); chuẩn hóa từng embedding trước khi gộp giúp effective weight nhất quán giữa mọi tầng.
- **Pooling**: `eᵤ = γ·eᵤ⁽⁰⁾ + (1−γ)·(1/K)·Σ_{k=1}^K eᵤ⁽ᵏ⁾` — tách riêng trọng số của đặc trưng ban đầu (`k=0`) khỏi đặc trưng cấu trúc (`k≥1`), giải quyết trực tiếp vấn đề thiếu nhất quán.

**Kết quả.** `α, β, γ` là các siêu tham số, **không** thêm tham số huấn luyện nào → độ phức tạp không gian `O(|V|d)` giữ nguyên, chi phí thời gian tăng không đáng kể (0.08%–5.29%/epoch). Cải thiện tới **17.81% NDCG@20** so với LightGCN, và — quan trọng về tính tổng quát — khi thay LightGCN bằng LightGCN++ làm backbone cho NCL, SimGCL, XSimGCL, CrossCBR (bundle), LATTICE (multimedia), KGCL (knowledge graph) thì các mô hình SOTA này đều được nâng cấp.

---

## 5. AlphaRec (2025) — Đổi hệ quy chiếu: từ ID-embedding sang biểu diễn ngôn ngữ

**Đây là điểm gãy trong mạch tiến hóa.** Ba công trình trước cùng nằm trong khung "ID-based CF trên đồ thị". AlphaRec (Sheng et al., ICLR 2025) đặt lại câu hỏi nền tảng: *liệu ta có cần ID-embedding không?* Giả thuyết phổ biến cho rằng không gian ngôn ngữ (LM) và không gian hành vi (behavior) là hai không gian tách biệt. AlphaRec bác bỏ điều này bằng phát hiện về **tính đồng cấu (homomorphism)**: khi ánh xạ *tuyến tính* biểu diễn ngôn ngữ của **tiêu đề item** (từ LM tiên tiến) sang không gian hành vi, ta thu được biểu diễn recommendation **vượt cả LightGCN** — nghĩa là tín hiệu cộng tác đã được *ngầm mã hóa* trong LM.

**Mô hình.** AlphaRec là mô hình GCN dựng **thuần trên biểu diễn ngôn ngữ, không có ID-embedding**:

1. **Nonlinear projection**: MLP 2 tầng + LeakyReLU chiếu `zᵢ` (biểu diễn LM của tiêu đề) và `zᵤ` (trung bình biểu diễn các item lịch sử) → embedding hành vi khởi tạo `e⁽⁰⁾`.
2. **Graph convolution**: dùng **đúng công thức LGC của LightGCN** `Σ 1/(√|Nᵤ|√|Nᵢ|)` + trung bình theo tầng `1/(K+1)·Σ eᵏ`. → LightGCN được kế thừa nguyên vẹn làm *lõi lan truyền*.
3. **Contrastive learning**: loss InfoNCE trên cosine similarity (thay vì BPR + inner product), không cần graph augmentation.

**Kết quả và ba "tiềm năng".** AlphaRec vượt các baseline ID-based mạnh nhất (bao gồm cả RLMRec, XSimGCL) 6.79%–9.75% recall trên Amazon Movies/Games/Books. Ngoài độ chính xác, ba năng lực mới mà dòng ID-based không có:

- **Khởi tạo tốt** (Potential 1): hội tụ cực nhanh nhờ đặc trưng ngôn ngữ.
- **Zero-shot transfer** (Potential 2): vì đặc trưng ngôn ngữ độc lập miền, AlphaRec suy luận trên tập dữ liệu hoàn toàn mới (không trùng ID) và vượt các baseline zero-shot 30%–157%.
- **Intention-aware** (Potential 3): trộn vector ý định dạng văn bản `(1−α)eᵤ⁽⁰⁾ + α·eᵤ^Intention` để tinh chỉnh gợi ý theo truy vấn ngôn ngữ tự nhiên.

**Giới hạn tự nhận.** Không có bảo đảm lý thuyết; không đề xuất kiến trúc CF mới (kiến trúc lõi vẫn là MLP + LightGCN-conv + CL); MLP chưa cá nhân hóa theo user; thí nghiệm ý định dùng trọng số `α` cố định.

---

## 6. Tổng hợp so sánh

### 6.1 Bảng đối chiếu bốn mô hình

| Khía cạnh | NGCF (2019) | LightGCN (2020) | LightGCN++ (2024) | AlphaRec (2025) |
|---|---|---|---|---|
| **Triết lý** | Nhồi tín hiệu cộng tác vào embedding qua lan truyền đồ thị | Đơn giản hóa tối đa: chỉ giữ neighborhood aggregation | Xét lại & thêm lại đúng vài bậc tự do | Bỏ ID-embedding, khai thác biểu diễn ngôn ngữ từ LM |
| **Nguồn embedding** | ID-embedding học được | ID-embedding học được | ID-embedding học được | Biểu diễn LM của tiêu đề item (đóng băng) + MLP |
| **Feature transform / phi tuyến** | Có (`W₁,W₂`, LeakyReLU) | **Bỏ cả hai** | Bỏ (kế thừa LightGCN) | Có MLP ở đầu vào, lan truyền vẫn tuyến tính |
| **Aggregation** | Có số hạng `eᵤ⊙eᵢ`, chuẩn hóa Laplacian | LGC tuyến tính, chuẩn hóa đối xứng | LGC tổng quát hóa với `α,β,γ` + chuẩn hóa norm | Dùng lại LGC của LightGCN |
| **Layer pooling** | Concatenation | Mean `1/(K+1)` | Tách tầng 0 khỏi tầng ≥1 qua `γ` | Mean `1/(K+1)` |
| **Prediction / loss** | Inner product / BPR | Inner product / BPR | Inner product / BPR | Cosine / **InfoNCE (CL)** |
| **Tham số huấn luyện thêm** | `W` mỗi tầng | Chỉ embedding tầng 0 | **Không thêm** (chỉ siêu tham số) | MLP + `W` chiếu; không có ID-embedding |
| **Đóng góp chính** | High-order connectivity tường minh | Bằng chứng ablation: bớt là hơn | Chẩn đoán inflexibility/inconsistency | Homomorphism ngôn ngữ↔hành vi; zero-shot & intention-aware |

> **Lưu ý khi đọc số liệu:** các con số Recall/NDCG **không so sánh chéo trực tiếp** giữa bốn bài, vì tập dữ liệu và giao thức đánh giá khác nhau (NGCF/LightGCN dùng Gowalla/Yelp2018/Amazon-Book; LightGCN++ dùng 5 tập gồm LastFM/MovieLens/Gowalla/Yelp/Amazon với `K=2`; AlphaRec dùng Amazon Movies/Games/Books). Chỉ nên so sánh mức *cải thiện tương đối trong cùng một bài*. Ví dụ minh họa: trong bảng riêng của AlphaRec, LightGCN đạt Recall 0.0849 (Movies&TV) còn AlphaRec đạt 0.1221; trong bảng riêng của LightGCN++, LightGCN đạt NDCG 0.1426 (Gowalla) còn LightGCN++ đạt 0.1469 — các cặp số này thuộc hai thiết lập thực nghiệm độc lập.

### 6.2 Ba mạch tiến hóa xuyên suốt

1. **Đơn giản hóa rồi tinh chỉnh (NGCF → LightGCN → LightGCN++).** Một chu trình khoa học điển hình: NGCF mượn nguyên cỗ máy GCN → LightGCN chứng minh bằng ablation rằng phần lớn là thừa và cắt bỏ → LightGCN++ phát hiện việc cắt bỏ đã vô tình tạo ra sự cứng nhắc/thiếu nhất quán, rồi thêm lại đúng ba bậc tự do (`α,β,γ`) **mà không thêm tham số huấn luyện**. Đáng chú ý: LightGCN++ phản biện chính "bài học đơn giản hóa" của LightGCN, cho thấy "tối giản" và "tối ưu" không đồng nghĩa.

2. **Neighborhood aggregation tuyến tính là bất biến bền vững.** Công thức LGC `Σ 1/(√|Nᵤ|√|Nᵢ|)` sống sót qua cả ba thế hệ sau NGCF, và được AlphaRec **tái sử dụng nguyên vẹn**. Đây là "hạt nhân" ổn định nhất của cả dòng nghiên cứu — điều tiến hóa là *đầu vào của embedding* và *hàm mục tiêu*, không phải bản thân phép lan truyền.

3. **Dịch chuyển nguồn tri thức: từ ID sang ngôn ngữ (→ AlphaRec).** Ba bài đầu vắt kiệt tín hiệu từ *cấu trúc đồ thị tương tác*; AlphaRec bổ sung nguồn tri thức *ngoài đồ thị* — world knowledge trong LM. Điều này mở khóa những năng lực mà ID-based CF về bản chất không thể có: transfer zero-shot xuyên miền và điều khiển bằng ý định ngôn ngữ tự nhiên. Song cần lưu ý AlphaRec **không thay thế** LightGCN mà *đứng trên vai* nó (dùng LGC làm lõi) — đây là sự hội tụ, không phải phủ định.

### 6.3 Đường dây học thuật

Bốn công trình gắn kết chặt về nhân sự: **Xiang Wang** và **Xiangnan He** là tác giả NGCF và LightGCN; Xiang Wang tiếp tục đồng tác giả AlphaRec (cùng Tat-Seng Chua, NUS/USTC). LightGCN++ đến từ nhóm độc lập (KAIST) — vị trí "người ngoài xét lại" này phù hợp với vai trò phản biện của bài. Mạch trích dẫn trực tiếp: LightGCN ablation trên NGCF → LightGCN++ phân tích LightGCN → AlphaRec dùng LightGCN làm baseline chính và mượn LGC làm lõi.

---

## 7. Kết luận và hướng mở

Từ NGCF đến AlphaRec là hành trình sáu năm mà mỗi bước đều là một *phản biện có kiểm chứng* với bước trước: NGCF khẳng định giá trị của lan truyền đồ thị; LightGCN chỉ ra phần lớn kiến trúc GNN là thừa cho CF; LightGCN++ cảnh báo rằng đơn giản hóa quá tay tạo ra điểm mù về độ linh hoạt; AlphaRec đặt lại câu hỏi liệu ID-embedding có còn cần thiết khi LM đã ngầm mã hóa tín hiệu cộng tác.

Các hướng mở lộ ra từ chính giới hạn của bốn bài: (i) nền tảng lý thuyết cho tính đồng cấu ngôn ngữ↔hành vi mà AlphaRec quan sát được; (ii) cá nhân hóa thành phần projection theo user; (iii) kết hợp bậc tự do kiểu LightGCN++ *lên trên* lõi biểu diễn ngôn ngữ của AlphaRec — một hợp lưu chưa được khai thác giữa hai nhánh; và (iv) đánh giá chuẩn hóa xuyên các thiết lập thực nghiệm vốn đang phân mảnh giữa các bài.

---

*Ghi chú AI: Bản tổng quan này được tổng hợp hoàn toàn từ bốn tài liệu do người dùng cung cấp; không truy cập nguồn ngoài. Các số liệu định lượng được trích trực tiếp từ bảng trong từng bài và chỉ nên so sánh trong phạm vi thiết lập thực nghiệm gốc của bài đó.*
