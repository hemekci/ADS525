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

#### Reranking

Search systems often contain multiple steps. A **reranking** language model is one of these steps.

***Reranking language models*** are tasked with *scoring the relevance of a subset of results* against the query. See the diagram below.

![[Pasted image 20251027224405.png]]

#### RAG

The growing LLM capabilities have led to a new type of search systems that include a model that generates an answer in response to a query. 
 
Generative search is a subset of a broader type of category of systems called **RAG systems.** 

**RAG Systems** incorporate search capabilities to reduce hallucinations, increase factuality and ground the generation model on a specific dataset.

![[Pasted image 20251027225003.png]]


### Semantic Search with Language Models

#### Dense Retrievals

As briefly mentioned earlier in the text, we can think of the values in an embedding as points in a space. This is illustrated below.

![[Pasted image 20251027230009.png]]

**Example, consider the illustration below:**  
- Should text 3 even be returned as a result? The answer depends on the nature of the system and the choice of you - the designer. Limits and thresholds are used to filter out *irrelevant* results. 
- Are query and its best result always semantically similar (and are close to each other in the embedding space)? Not necessarily. This is why language models need to be *trained* on question-answer pairs to become better at retrieval.

![[Pasted image 20251027230606.png]]

#### Dense Retrieval Example

Here, we'll test and use embeddings to search for text.

Particularly, we will:
1. Get the text we want to make searchable and apply some processing to chunk it into sentences.
2. Embed the sentences.
3. Build the search index.
4. Search and see the results.


```python
import cohere 
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paste your API key here. Remember not to share it publicly.
api_key = ''

# Create and retrieve a Cohere API key fromos.cohere.ai
co = cohere.Client(api_key)
```

Let's use the first section of the article on the film Interstellar.
##### Getting the text archive and chunking it
```python
text = """
Interstellar is a 2014 epic science fiction film co-written,
directed, and produced by Christopher Nolan.
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain,
Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.
Set in a dystopian future where humanity is struggling to
survive, the film follows a group of astronauts who travel
through a wormhole near Saturn in search of a new home for
mankind.
Brothers Christopher and Jonathan Nolan wrote the screenplay,
which had its origins in a script Jonathan developed in 2007.
Caltech theoretical physicist and 2017 Nobel laureate in
Physics[4] Kip Thorne was an executive producer, acted as a
scientific consultant, and wrote a tie-in book, The Science of
Interstellar.
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in
the Panavision anamorphic format and IMAX 70 mm.
Principal photography began in late 2013 and took place in
Alberta, Iceland, and Los Angeles.
Interstellar uses extensive practical and miniature effects and
the company Double Negative created additional digital effects.
Interstellar premiered on October 26, 2014, in Los Angeles.
In the United States, it was first released on film stock,
expanding to venues using digital projectors.
The film had a worldwide gross over $677 million (and $773
million with subsequent re-releases), making it the tenth-highest
grossing film of 2014.
It received acclaim for its performances, direction, screenplay,
musical score, visual effects, ambition, themes, and emotional
weight.
It has also received praise from many astronomers for its
scientific accuracy and portrayal of theoretical astrophysics.
Since its premiere, Interstellar gained a cult following,[5] andnow is regarded by many sci-fi experts as one of the best
science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy
Awards, winning Best Visual Effects, and received numerous other
accolades"""


# Split into a list of sentences
texts = text.split('.')

# Clean up to remove empty spaces and new lines
texts = [t.strip(' \n') for t in texts]
```

Let's now embed the texts. We will use Cohere to get a vector embedding for each text.
##### Embedding the text chunks
```python
# Get the embeddings
response = co.embed(
	texts=texts,
	input_type="search_document",
).embeddings

embeds = np.array(response)
print(embeds.shape)
```

This outputs `(15, 4096)`, which indicates that we have 15 vectors, each one of size 4,096.

##### Building the search index

Before we can search, we need to build a **search index**. An index stores the embeddings and is *optimized to quickly retrieve the **nearest neighbors***

```python
import faiss
dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
print(index.is_trained)
index.add(np.float32(embeds))
```

##### Search the index

After building the search index, we can now use it to search the dataset (Wikipedia article for Interstellar)

We embed the query and present its embedding to the index, which will retrieve the most similar sentence. 

Below is the search function:
```python
def search(query, number_of_results=3):

	# 1. Get the query's embedding
	query_embed = co.embed(texts=[query], input_type="search_query",).embeddings[0]
	
	# 2. Retrieve the nearest neighbors
	distances , similar_item_ids = index.search(np.float32([query_embed]), number_of_results)
	
	# 3. Format the results
	texts_np = np.array(texts) # Convert texts list to numpy for
	easier indexing
	results = pd.DataFrame(data={'texts': texts_np[similar_item_ids[0]], 'distance': distances[0]})
	
	# 4. Print and return the results
	print(f"Query:'{query}'\nNearest neighbors:")
	return results
```

We are now ready to write a query and search the texts:

```python 
query = "how precise was the science"
results = search(query)
results
```

This outputs:
```markdown
Query: 'how precise was the science'
Nearest neighbors:
```
![[Pasted image 20251028002530.png]]

The **first** result has the **least distance**, and so is the **most similar** to the query.

Realize that because we used the word 'science', if we used *keyword matching*, there would not be any result. 

However, since we're using *semantic search*, it returned the results we see (containing '**scientific**..', '**theoretical** **astrophysics**', etc.). We can see how useful semantic search can be. 

Let's see how the **BM25** (one of the
leading lexical search methods) algorithm performs. 

```python
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string

def bm25_tokenizer(text):
	tokenized_doc = []
	for token in text.lower().split():
		token = token.strip(string.punctuation)
		if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
			tokenized_doc.append(token)
	return tokenized_doc
	
tokenized_corpus = []
for passage in tqdm(texts):
	tokenized_corpus.append(bm25_tokenizer(passage))
	
bm25 = BM25Okapi(tokenized_corpus)

def keyword_search(query, top_k=3, num_candidates=15):
	print("Input question:", query)
	
	##### BM25 search (lexical search) #####
	bm25_scores = bm25.get_scores(bm25_tokenizer(query))
	top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
	
	bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
	
	bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
	
	print(f"Top-3 lexical search (BM25) hits")
	for hit in bm25_hits[0:top_k]:
		print("\t{:.3f}\t{}".format(hit['score'], texts[hit['corpus_id']].replace("\n", " ")))

```

Now when we search for the same query, we get a different set of results from the dense retrieval search:

```python
keyword_search(query = "how precise was the science")
```

Results:

```markdown
Input question: how precise was the science
Top-3 lexical search (BM25) hits
	1.789 Interstellar is a 2014 epic science fiction film
co-written, directed, and produced by Christopher Nolan
	1.373 Caltech theoretical physicist and 2017 Nobel
laureate in Physics[4] Kip Thorne was an executive producer,
acted as a scientific consultant, and wrote a tie-in book, The
Science of Interstellar
	0.000 It stars Matthew McConaughey, Anne Hathaway,
Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and
Michael Caine
```

Note that the first result does not really answer the question despite it sharing the word “science” with the query.

##### Caveats of dense retrieval
Dense retrieval *always* returns the results, even when the documents don't contain the answer. It doesn't have a built-in no answer. An **example** is provided below:
```markdown
Query: 'What is the mass of the moon?'
Nearest neighbors:
```
![[Pasted image 20251030184518.png]]

In cases like this, one possible heuristic is to set a **threshold level**. 

A lot of search systems (e.g., search engines) present the user with the best info they can get and let the user decide if it's relevant or not. They also track information such as whether the user clicked on a result, or was satisfied by it, to improve future versions of the search system.

Another caveat of dense retrieval is when a user wants to find an exact match for a specific phrase. **That's a case that's perfect for keyword matching.** That's also why a hybrid search system (that combines keyword matching and dense retrieval) is advised instead of relying solely on dense retrieval. 

Dense retrieval systems also find it challenging to work properly other than the ones they are trained on

Summary:
- **No built-in relevance detection**
  - Always returns results even when documents don't contain the
  answer
  - All results get similarity scores regardless of actual
  relevance
  - Solution: Set distance thresholds, track user clicks to
  improve system

- **Poor at exact phrase matching**
  - Struggles when users need specific keyword/phrase matches
  - Solution: Use hybrid search (semantic + keyword search like
  BM25)

- **Domain mismatch problems**
  - Model trained on Wikipedia/internet data fails on specialized
  domains (legal, medical, etc.)
  - Solution: Include domain-specific data in training, or
  fine-tune for target domain

- **Text chunking challenges**
  - Hard to handle answers spanning multiple sentences
  - Design decision: How to chunk long texts optimally?
  - Trade-off between chunk size and information completeness

  ***Key takeaway:*** Dense retrieval works best combined with other
  techniques (hybrid search, thresholding, domain adaptation)
  rather than used alone.


##### Chunking Long Texts
As we've seen in early chapters, every model has a context size limit. This is a limitation of **Transformer LMs**. This means we cannot input very long texts.

One way to solve this is by using *chunking*.

An illustration of chunking is provided below.
![[Pasted image 20251030195417.png]]

##### One vector per document
When doing this, there are some possibilities:
1. Embedding only a **representative part of the text** (e.g., title, beginning of text).  This means embedding only the **title, or beginning of the document**. This method may cause in a lot of loss of information.
2. Embedding the documents in **chunks**, **embedding those chunks**, and then **aggregating (averaging) those chunks into a single vector**. A downside is that it results in a highly compressed vector that loses a lot of the information of the document. 

This approach can satisfy some information needs, but not others. This depends on whether the search is for a specific piece of information or a semantic concept.

##### Multiple vectors per document
This is basically **chunking the document into smaller pieces**, and **embedding those chunks, without aggregating them**.    

Because it has full coverage of the text, it leads to a more expressive search index. 

![[Pasted image 20251030205029.png]]

The best way of chunking a long text will depend on the types of texts and queries your system anticipates. 
![[Pasted image 20251030205221.png]]

Was of chunking include:
1. **Each sentence is a chunk.** 
	The issue is that it may not capture the overall context of a document. 
2. **Each **