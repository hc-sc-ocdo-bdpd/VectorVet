# 🧬 VectorVet: Quick Embedding Diagnostics for Semantic Search & RAG Pipelines  
  
VectorVet provides simple, practical metrics and tooling for evaluating the quality of embedding vectors generated by local language models. It's especially useful for quickly assessing whether embeddings from smaller or quantized models (such as those that fit on a laptop GPU) are suitable for semantic search and Retrieval-Augmented Generation (RAG) applications.  
  
The library calculates intrinsic embedding metrics—such as isotropy, hubness, clustering quality, and cosine similarity statistics—to help you quickly determine whether your embeddings are well-structured for semantic search.  
  
---  
  
## 🚩 Why VectorVet?  
  
When building internal RAG chatbots or semantic search systems, embedding quality is crucial. Smaller or quantized models are attractive for privacy and portability, but their embeddings can often suffer from poor distribution or semantic structure.  
  
VectorVet helps you rapidly measure embedding quality, giving you early insights into whether your embeddings are likely to support effective semantic search.  
  
**Quickly answer questions like:**  
  
- **Are my embeddings well-distributed (isotropic)?**  
- **Do my embeddings suffer from "hubness" problems?**  
- **Do the embeddings naturally cluster by topic?**  
- **Will semantic search likely yield meaningful results with this embedding model?**  
  
---  
  
## 🔍 Metrics Explained  
  
VectorVet computes the following intrinsic metrics to assess your embeddings:  
  
### Isotropy  
Measures how evenly embeddings utilize vector space dimensions. An isotropic embedding set uses all directions in vector space, which is desirable for effective semantic search.  
  
- **IsoScore** ∈ [0, 1]; higher is better.  
  
### Hubness  
Identifies whether certain embeddings ("hubs") appear disproportionately often as nearest neighbors, reducing search quality.  
  
- **Skewness**: Lower is better.  
- **Robin Hood Index**: Lower is better.  
- **Antihub Rate**: Indicates embeddings rarely appearing as nearest neighbors; lower is generally preferable.  
  
### Clustering Quality  
Evaluates how well embeddings naturally form distinct clusters, indicative of good semantic separation.  
  
- **Silhouette Score**: Higher is better.  
- **Davies-Bouldin Index**: Lower is better.  
  
### Pairwise Cosine Similarity  
Provides general statistics (mean and standard deviation) of cosine similarity between embeddings, indicating overall vector similarity distribution.  
  
---  
  
## 📖 Example Usage  
  
For a complete example workflow—including dataset preparation, embedding generation, metric calculations, and results summarization—please see the provided Jupyter Notebook demo:  
  
- [`notebooks/vectorvet_demo.ipynb`](notebooks/vectorvet_demo.ipynb)  
  
This notebook demonstrates end-to-end usage of VectorVet, from loading embeddings to interpreting the resulting metrics.  
  
---  

## 💡 Interpretation Guidelines  
  
Intrinsic metrics help you quickly diagnose embedding quality. While exact values depend somewhat on your dataset and embedding dimensionality, good embeddings typically exhibit:  
  
- **Reasonable isotropy:** Embeddings utilize multiple directions in vector space rather than collapsing into a few dimensions.  
- **Low hubness:** No embeddings disproportionately dominate nearest-neighbor queries.  
- **Clear clustering structure:** Embeddings naturally form meaningful semantic clusters.  
- **Balanced cosine similarity:** Embeddings are neither too similar (collapsing) nor too dissimilar (sparse).  
  
Use the following table as a practical starting point to interpret your results:  
  
| Metric                           | ✅ Good                         | ⚠️ Concerning                    | 🚫 Poor / Likely Problematic    |  
|----------------------------------|---------------------------------|----------------------------------|---------------------------------|  
| **Isotropy (IsoScore)**          | ≥ 0.01                          | 0.001–0.01                       | < 0.001                         |  
| **Hubness: Skewness**            | ≤ 1.5                           | 1.5–2.5                          | > 2.5                           |  
| **Hubness: Robin Hood Index**    | ≤ 0.25                          | 0.25–0.30                        | > 0.30                          |  
| **Hubness: Antihub Rate**        | ~0.0 (ideal)                    | 0.0–0.05                         | > 0.05                          |  
| **Clustering: Silhouette**       | ≥ 0.10                          | 0.05–0.10                        | < 0.05                          |  
| **Clustering: Davies-Bouldin**   | ≤ 2.5                           | 2.5–3.5                          | > 3.5                           |  
| **Cosine Similarity (Mean)**     | Moderate (0.3–0.7)              | Slightly Low (0.1–0.3) or Slightly High (0.7–0.8) | Very Low (<0.1) or Very High (>0.8) |  
| **Cosine Similarity (Std Dev)**  | ≥ 0.20                          | 0.10–0.20                        | < 0.10                          |  
  
### 📌 **Practical Interpretation Tips:**  
  
- **Isotropy (IsoScore)**:   
  Most embedding models—even strong ones—often yield low isotropy scores. Scores ≥ 0.01 typically suggest relatively balanced embeddings. Very low isotropy scores (<0.001) indicate embeddings are concentrated in very few directions.  
  
- **Hubness Metrics**:   
  Lower skewness and Robin Hood indices indicate healthier nearest-neighbor distributions. Scores significantly above recommended thresholds (e.g., skewness > 2.5, Robin Hood > 0.3) indicate problematic embeddings that dominate search results.  
  
- **Clustering Quality**:  
  Intrinsic clustering metrics typically yield modest numeric values—even good embeddings often score around 0.05–0.10 for silhouette. Scores above these ranges are strong indicators of semantic coherence. Consistently low scores (<0.05 silhouette, >3.5 Davies-Bouldin) suggest embeddings with poor semantic structure.  
  
- **Cosine Similarity**:  
  Mean cosine similarities between ~0.3 and ~0.7 usually reflect balanced embeddings. Values near extremes (<0.1 or >0.8) often indicate embedding collapse or excessive sparsity. A higher cosine similarity standard deviation (≥ 0.20) is desirable, reflecting a healthy embedding variation.  
  
### ⚠️ **Important Notes:**  
  
These thresholds are practical, heuristic guidelines derived from empirical observations. Actual optimal metric ranges can vary depending on your specific dataset, embedding dimensionality, and semantic domain. Always validate embedding quality through practical semantic search and retrieval tests with representative data.  

---  
  
## ⚠️ Caveats  
  
Intrinsic metrics provide a quick and powerful diagnostic tool but are not a guarantee of actual semantic search performance. Always perform at least a minimal end-to-end semantic retrieval test on your specific data and embedding model before committing significant resources.  
  
Additionally, intrinsic metric thresholds are approximate guidelines. Optimal thresholds can vary significantly depending on your dataset domain, use-case, and embedding dimensionality. If possible, benchmark your results against known high-quality embedding models or datasets to calibrate your expectations more accurately.  
  
Intrinsic metrics also depend heavily on the dataset used. A model performing well intrinsically on one dataset might not perform equally well on another. Always run these metrics against embeddings from your actual data or a representative subset of your data.  
  
If embeddings consistently score poorly on several metrics, consider trying other embedding models or applying post-processing techniques (mean centering, PCA whitening, or hubness reduction) to improve quality.  