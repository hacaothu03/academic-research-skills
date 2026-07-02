# Literature Search Report — NGCF → AlphaRec (lit-review mode)

## Paper Configuration Record (đã xác nhận)

| Tham số | Giá trị |
|---|---|
| **Chủ đề** | Sự tiến hóa của Graph-based Collaborative Filtering: NGCF → LightGCN → LightGCN++ → AlphaRec |
| **Loại bài viết** | Literature Review — focused comparative review (4 công trình cột mốc, không phải khảo sát diện rộng) |
| **Ngành** | Computer Science — Recommender Systems / Graph Neural Networks |
| **Domain Evidence Profile** | `cs_ml` |
| **Định dạng trích dẫn** | IEEE |
| **Ngôn ngữ nội dung** | Tiếng Việt |
| **Tóm tắt** | Song ngữ (VI + EN) |
| **Phạm vi nguồn** | Cố định đúng 4 nguồn do người dùng cung cấp — không tìm kiếm mở rộng |
| **Citation Verification** | advisory (mark only) — người dùng chọn không tra cứu API; các mục bên dưới được đánh dấu UNVERIFIED khi chưa được xác minh qua DOI/API |

---

## Search Strategy

**Chế độ corpus cố định (user-curated, `user_corpus_only`).** Không chạy tìm kiếm 4-Layer Progressive Strategy vì phạm vi đã được người dùng giới hạn có chủ đích ở 4 công trình cột mốc. `uncovered_topics` không được xử lý bằng tìm kiếm ngoài — theo lựa chọn "giữ nguyên 4 nguồn".

- **Nguồn**: bản nháp `literature_review_ngcf_to_alpharec.md` do người dùng cung cấp.
- **Tiêu chí sàng lọc áp dụng** (Iron Rule 1 — cùng tiêu chí cho corpus và nguồn ngoài, dù ở đây không có nguồn ngoài): liên quan trực tiếp đến RQ, không có lỗi phương pháp luận nghiêm trọng, không phải nguồn ngụy khoa học/predatory. Cả 4 nguồn đều là công trình đã qua bình duyệt tại các venue hàng đầu ngành (SIGIR, RecSys, ICLR) → đạt ngưỡng universal gates.

## Coverage Distribution Advisory

DISTRIBUTIONAL_SKEW_ADVISORY:
- **Dimension**: đường dây tác giả (authorship lineage — không nằm trong 4 trục chuẩn nhưng đáng nêu vì bản nháp §6.3 đã tự nhận diện)
- **Concentration**: Xiang Wang xuất hiện là đồng tác giả ở 3/4 nguồn (NGCF, LightGCN, AlphaRec) = 75%
- **Advisory**: đây là tín hiệu về mạch học thuật liên tục, không phải khiếm khuyết — nhưng người đọc nên biết LightGCN++ (KAIST) là góc nhìn "người ngoài" duy nhất trong bộ 4 nguồn, không có tiếng nói phản biện độc lập thứ hai.
- **Search response**: không mở rộng — nằm ngoài phạm vi đã chốt (giữ nguyên 4 nguồn).

Các trục chuẩn (thời gian, địa lý, phương pháp, venue tier): với N=4 không đủ để đánh giá có ý nghĩa thống kê — không phát advisory trên các trục này. Về thời gian, 4 nguồn trải đều 2019–2025 (6 năm), không tập trung.

## Screening Results

- Corpus ban đầu (do người dùng cung cấp): 4
- Sau sàng lọc Phase A/B: 4/4 Included
- Nguồn ngoài: 0 (theo lựa chọn user_corpus_only)

---

## Annotated Bibliography

> **Lưu ý xác minh (bắt buộc đọc trước khi dùng chính thức):** các trích dẫn dưới đây được dựng lại từ nội dung bản nháp gốc + kiến thức nền, **KHÔNG được xác minh qua DOI/Semantic Scholar/arXiv API** — đúng theo lựa chọn "giữ nguyên 4 nguồn, không tra cứu" của bạn. Tên tác giả đầy đủ, số trang chính xác, và một phần tên hội nghị có thể không chính xác 100%. Trước khi đưa vào một bài nộp chính thức, khuyến nghị chạy `academic-paper citation-check` hoặc xác minh thủ công qua DOI/arXiv ID. Không có DOI nào được bịa ra ở đây (tuân thủ IRON RULE chống fabricated citation).

### [1] X. Wang, X. He, M. Wang, F. Feng, and T.-S. Chua, "Neural Graph Collaborative Filtering," *SIGIR '19*, 2019. **[UNVERIFIED — xác minh tên đầy đủ tác giả/số trang/DOI]**
- **Loại**: Conference paper (bình duyệt, top-tier)
- **Phương pháp**: Thiết kế kiến trúc GNN (message-passing) cho CF; thực nghiệm so sánh trên Gowalla/Yelp2018/Amazon-Book
- **Phát hiện chính**: Mã hóa tường minh high-order connectivity vào hàm embedding thông qua lan truyền đồ thị hai phía; vượt baseline ~12.5% NDCG, đặc biệt cải thiện cold-start
- **Liên quan RQ**: Câu trả lời đầu tiên cho câu hỏi "embedding nên sinh ra thế nào để nắm tín hiệu cộng tác" — điểm khởi đầu của toàn bộ mạch tiến hóa
- **Chất lượng**: Cao — venue hàng đầu, được trích dẫn rộng rãi, là nền cho 3 công trình còn lại. Hạn chế tự thân: thừa kế nguyên xi feature transformation + phi tuyến từ GCN thuần, thiết kế cho node classification chứ không phải CF
- **Sử dụng dự kiến**: Phần mở đầu / nền tảng lý thuyết

### [2] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation," *SIGIR '20*, 2020. **[UNVERIFIED — xác minh tên đầy đủ tác giả/số trang/DOI]**
- **Loại**: Conference paper (bình duyệt, top-tier)
- **Phương pháp**: Ablation có kiểm soát trên NGCF (3 biến thể: NGCF-f/-n/-fn)
- **Phát hiện chính**: Bỏ feature transformation + phi tuyến (NGCF-fn) cho cải thiện lớn nhất (~9.57% recall); nguyên nhân là khó huấn luyện, không phải overfitting
- **Liên quan RQ**: Phản biện trực tiếp NGCF — chứng minh phần lớn cỗ máy GNN là dư thừa cho CF
- **Chất lượng**: Cao — trở thành backbone mặc định của cả dòng nghiên cứu sau này (social/bundle/multimedia/contrastive CF)
- **Sử dụng dự kiến**: Phần lõi so sánh phương pháp luận

### [3] G. Lee, W. Kim, and K. Shin, "LightGCN++: [Revisiting LightGCN — Unexpected Inflexibility, Inconsistency, and A Remedy]," *RecSys '24*, 2024. **[UNVERIFIED — tên bài đầy đủ, tên tác giả, số trang, DOI đều cần xác minh]**
- **Loại**: Conference paper (bình duyệt) — nhóm KAIST, độc lập với dòng tác giả NGCF/LightGCN
- **Phương pháp**: Phân tích cơ chế thực nghiệm sâu (không đề xuất kiến trúc mới); chẩn đoán effective weight qua norm embedding
- **Phát hiện chính**: LightGCN có 2 vấn đề ẩn — inflexibility (mọi láng giềng gộp gần như đồng đều khi k≥1) và inconsistency (lệch chuẩn giữa tầng 0 và tầng ≥1); đề xuất thêm 3 siêu tham số `α, β, γ` không tăng tham số huấn luyện
- **Liên quan RQ**: Phản biện chính "bài học đơn giản hóa" của LightGCN — cho thấy tối giản ≠ tối ưu
- **Chất lượng**: Cao — cải thiện tới 17.81% NDCG@20, tổng quát hóa tốt khi thay làm backbone cho 6 mô hình SOTA khác (NCL, SimGCL, XSimGCL, CrossCBR, LATTICE, KGCL)
- **Sử dụng dự kiến**: Phần phản biện/xét lại phương pháp luận

### [4] L. Sheng *et al.* (bao gồm T.-S. Chua), "[AlphaRec — tên bài đầy đủ cần xác minh, khả năng: 'Language Representations Can be What Recommenders Need: Findings and Potentials']," *ICLR*, 2025. **[UNVERIFIED — danh sách tác giả đầy đủ, tên bài chính xác, DOI/arXiv ID đều cần xác minh]**
- **Loại**: Conference paper (bình duyệt, top-tier)
- **Phương pháp**: Phát hiện tính đồng cấu (homomorphism) giữa không gian ngôn ngữ (LM) và không gian hành vi; kiến trúc MLP + LGC (tái dùng nguyên vẹn LightGCN) + contrastive learning (InfoNCE)
- **Phát hiện chính**: Ánh xạ tuyến tính biểu diễn LM của tiêu đề item sang không gian hành vi vượt cả LightGCN; mở khóa 3 năng lực mới: khởi tạo tốt, zero-shot transfer, intention-aware
- **Liên quan RQ**: Điểm gãy trong mạch tiến hóa — đặt lại câu hỏi liệu ID-embedding có còn cần thiết
- **Chất lượng**: Cao — vượt baseline ID-based mạnh nhất (RLMRec, XSimGCL) 6.79–9.75% recall; tự nhận hạn chế rõ ràng (không có bảo đảm lý thuyết, MLP chưa cá nhân hóa)
- **Sử dụng dự kiến**: Phần kết luận / hướng mở

---

## Literature Matrix

| Nguồn | Th1: Kiến trúc lan truyền đồ thị | Th2: Đơn giản hóa/Ablation | Th3: Chẩn đoán & tinh chỉnh | Th4: Dịch chuyển nguồn tri thức (ID→ngôn ngữ) | Phương pháp | Chất lượng |
|---|---|---|---|---|---|---|
| [1] NGCF (2019) | **main** | | | | Kiến trúc + thực nghiệm so sánh | High |
| [2] LightGCN (2020) | x | **main** | | | Ablation có kiểm soát | High |
| [3] LightGCN++ (2024) | x | x | **main** | | Phân tích cơ chế thực nghiệm | High |
| [4] AlphaRec (2025) | x | | | **main** | Kiến trúc + contrastive learning | High |

Mỗi Theme ≥ 1 nguồn "main" — với N=4 và phạm vi cố định, ngưỡng "≥3 sources/theme" của Quality Gate mặc định không áp dụng nguyên vẹn (được nới lỏng có ý thức theo lựa chọn phạm vi hẹp của người dùng, xem Failure Handling Note bên dưới).

## Identified Gaps

1. **Khoảng trống lý thuyết** — chưa có nền tảng lý thuyết giải thích *tại sao* tồn tại tính đồng cấu ngôn ngữ↔hành vi mà AlphaRec quan sát được bằng thực nghiệm.
2. **Khoảng trống cá nhân hóa** — thành phần MLP projection của AlphaRec chưa được cá nhân hóa theo user.
3. **Khoảng trống hợp lưu phương pháp** — chưa có công trình kết hợp các bậc tự do kiểu LightGCN++ (`α, β, γ`) *lên trên* lõi biểu diễn ngôn ngữ của AlphaRec; đây là hợp lưu chưa khai thác giữa hai nhánh tiến hóa.
4. **Khoảng trống đánh giá chuẩn hóa** — 4 bài dùng 3 bộ dữ liệu/giao thức đánh giá khác nhau (Gowalla/Yelp2018/Amazon-Book; LastFM/MovieLens/Gowalla/Yelp/Amazon; Amazon Movies/Games/Books), khiến so sánh chéo trực tiếp không khả thi — cần một benchmark thống nhất xuyên các thế hệ mô hình.
5. **Khoảng trống phản biện độc lập** — 3/4 nguồn chia sẻ đường dây tác giả (Xiang Wang/Xiangnan He); LightGCN++ là tiếng nói "người ngoài" duy nhất — thiếu một góc nhìn phản biện độc lập thứ hai đối với hướng AlphaRec.

## Failure Handling Note (minh bạch, theo Quality Gate protocol)

Với phạm vi cố định N=4 (chủ đích của người dùng, không phải thiếu sót tìm kiếm), 3 Quality Gate mặc định của `literature_strategist_agent` không đạt ngưỡng nguyên bản và được ghi nhận công khai thay vì âm thầm bỏ qua:
- Source count (mặc định ≥30 cho Literature Review) → **không đạt, có chủ đích** — đây là *focused comparative review* 4 công trình cột mốc, không phải survey diện rộng.
- Peer-reviewed ratio ≥70% → **đạt** (4/4 = 100%).
- Currency ≥50% trong 5 năm gần nhất → **đạt** (3/4 nguồn trong 2020–2025).

## Recommended Sources by Section

| Phần | Nguồn chính |
|---|---|
| Mở đầu / Bối cảnh | [1] |
| Thân bài — dòng tiến hóa | [1], [2], [3] |
| Điểm gãy / Đổi hệ quy chiếu | [4] |
| Tổng hợp so sánh | [1], [2], [3], [4] |
| Kết luận / Hướng mở | [3], [4] |

---

## Tóm tắt song ngữ (Bilingual Abstract)

### Tiếng Việt

Bài tổng quan này truy vết một mạch tiến hóa liên tục của Collaborative Filtering dựa trên đồ thị qua bốn công trình cột mốc: NGCF (2019), LightGCN (2020), LightGCN++ (2024) và AlphaRec (2025). NGCF mã hóa tường minh tín hiệu cộng tác vào embedding bằng lan truyền trên đồ thị hai phía, nhưng thừa kế cỗ máy GNN nặng nề từ node classification. LightGCN dùng ablation có kiểm soát chứng minh phần lớn cỗ máy đó là dư thừa, giữ lại duy nhất neighborhood aggregation tuyến tính. LightGCN++ xét lại phát hiện rằng việc đơn giản hóa này vô tình tạo ra sự cứng nhắc và thiếu nhất quán trong trọng số láng giềng, và khắc phục bằng ba siêu tham số không làm tăng chi phí huấn luyện. AlphaRec đánh dấu một điểm gãy: thay vì tối ưu tiếp trên ID-embedding, nó khai thác tính đồng cấu giữa biểu diễn ngôn ngữ và không gian hành vi, mở khóa khả năng zero-shot transfer và điều khiển bằng ý định ngôn ngữ tự nhiên — trong khi vẫn tái sử dụng nguyên vẹn lõi lan truyền của LightGCN. Bài viết chỉ ra ba mạch xuyên suốt (đơn giản hóa-rồi-tinh chỉnh, tính bất biến của neighborhood aggregation tuyến tính, và dịch chuyển nguồn tri thức từ ID sang ngôn ngữ), đồng thời nêu năm khoảng trống nghiên cứu, trong đó đáng chú ý nhất là khả năng hợp lưu giữa cơ chế tinh chỉnh của LightGCN++ và lõi biểu diễn ngôn ngữ của AlphaRec.

**Từ khóa**: Collaborative Filtering, Graph Neural Network, LightGCN, biểu diễn ngôn ngữ, hệ thống gợi ý

### English

This review traces a continuous line of evolution in graph-based Collaborative Filtering across four milestone works: NGCF (2019), LightGCN (2020), LightGCN++ (2024), and AlphaRec (2025). NGCF explicitly encodes collaborative signal into embeddings via propagation over the user-item bipartite graph, but inherits the full GNN machinery designed for node classification. LightGCN uses controlled ablation to show that most of this machinery is redundant, retaining only linear neighborhood aggregation. LightGCN++ revisits this simplification and finds it inadvertently introduces inflexibility and inconsistency in neighbor weighting, remedied by three hyperparameters that add no training cost. AlphaRec marks a genuine break: rather than further optimizing ID-based embeddings, it exploits a homomorphism between language representations and the behavior space, unlocking zero-shot transfer and natural-language intention control — while still reusing LightGCN's propagation core unchanged. The review identifies three cross-cutting threads (simplify-then-refine, the invariance of linear neighborhood aggregation, and the knowledge-source shift from IDs to language) and five research gaps, most notably the unexplored confluence between LightGCN++'s refinement mechanism and AlphaRec's language-representation core.

**Keywords**: Collaborative Filtering, Graph Neural Network, LightGCN, language representations, recommender systems

---

*Ghi chú AI (tuân thủ Quality Standard #17 — AI disclosure): Báo cáo này được tạo bởi `academic-paper` skill (mode `lit-review`), dựa trên bản nháp do người dùng cung cấp làm nguồn duy nhất (không tìm kiếm nguồn ngoài, theo lựa chọn của người dùng). Các trích dẫn IEEE trong Annotated Bibliography CHƯA được xác minh qua DOI/API — xem cảnh báo UNVERIFIED ở từng mục. Trước khi dùng cho mục đích nộp bài chính thức, khuyến nghị xác minh thủ công hoặc chạy `/ars-citation-check`.*
