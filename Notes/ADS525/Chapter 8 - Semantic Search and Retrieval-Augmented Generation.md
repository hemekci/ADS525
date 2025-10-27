One of the things language models enabled is the ability to search using semantic meaning instead of just having to use keyword matching. This is called ***semantic search***.

While the development of generation models allowed for fluent and readable responses, these responses are not always completely factual. To address this, one of the ways developed to provide knowledge to the language model is by using **RAG (Retrieval-Augmented Generation)**.

### Overview of Semantic Search and RAG

The top **three** broad categories of language models used for search are:
1. **Dense Retrieval**
2. **Reranking**
3. **RAG**

Below, we will go through each of these categories one by one and in-depth.

#### Dense Retrieval

**Dense Retrieval** systems rely on the concept of *embeddings* (discussed in depth in the early chapters). Using embeddings, we can turn searching into retrieving the nearest neighbors. 


> [!tip] Recall
> Remember that the float values of an embedding represent their properties and their coordinates in an embedding space. 
> 
> This embedding space could have a hundred or more dimensions. The closer the embeddings of texts are, **the closer their meanings are.**
> 

![[Pasted image 20251027223341.png]]

