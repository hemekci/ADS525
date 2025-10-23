#### Important Topics
1. **Model I/O**: Loading and working with LLMs
2. **Memory**: Helping LLMs to remember
3. **Agents**: Combining complex behavior with external tools
4. **Chains**: Connecting methods and modules

These methods are all integrated with the LangChain framework.
![[Pasted image 20251021113933.png]]

### Model I/O: Loading Quantized Models with LangChain

As in previous chapters, we will be using Phi-3 GGUF (GPT-Generated Unified Format) model variant.

A **GGUF** model represents a compressed version of its original counterpart through a method called **quantization**, which *reduces the number of bits needed to represent the parameters of an LLM.*
![[Pasted image 20251021141524.png]]


> [!tip] Quantization
> **Quantization** reduces the number of bits required to represent the parameters the parameters of an LLM while attempting most of the original information. This could cause some reduction in precision. But this loss is made up for in speed, and lower resource needs (requires less VRAM). 

Know that for the rest of the chapter, we will be using an 8-bit variant of Phi-3 compared to the original 16-bit variant, cutting memory requirements almost in half.

First, we need to download the model. `FP16`, the chosen model, represents the **16-bit** variant.

```markdown
!wget https://huggingface.co.microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf
```

We use `llama-cpp-python` together with LangChain to load the GGUF file:
```python
from langchain import LLamaCpp

# Make sure the model path is correct for your system
llm = LlamaCpp(
	model_path = "Phi-3-mini-4k-instruct-fp16.gguf",
	n_gpu_layers = -1 # -1 means you want all the layers to use the GPU.
	max_tokens = 500,
	n_ctx = 2048, # this defines the context length, discussed in previous chapters
	seed = 42,
	verbose = False
	)
```

In LangChain, we use the `invoke` function to generate output:
```python
llm.invoke("Hi! My name is Maarten. What is 1 + 1?")
```

This outputs:
```markdown
''
```

As you can see, there is no output. Looking back to the previous chapters, we know `Phi-3` requires a specific prompt template. But instead of having to use this template every time, we could instead use **LangChain**'s core functionality, called '*chains*'. Click [here](https://python.langchain.com/api_reference/langchain/chains.html) for the documentation.


> [!tip] TIP
> Although we have been using `Phi-3`, you can choose any LLM. To see the best current models, visit [this page](https://www.vellum.ai/open-llm-leaderboard?utm_source=duckduckgo&utm_medium=organic).

#### Chains: Extending the Capabilities of LLMs

Although LLMs can be run in isolation, chains allow for better integration with tools, agent-like behavior, and combination with other tools.

The most basic form of a chain in LangChain is a single chain. Although a chain can vary in complexity and form, it generally connects an LLM with some additional tool, prompt or feature.

![[Pasted image 20251023130128.png]]


#### A Single Link in the Chain: Prompt Template

We start with creating our first chain, namely the prompt template that `Phi-3` expects. 
![[Pasted image 20251023130435.png]]

The template for `Phi-3` is comprised of **four main components:**
- `<s>` to indicate when the prompt starts
- `<|user|>` to indicate the start of the user's prompt
- `<|assistant|>` to indicate the start of the model's output
- `<|end|>` to indicate the end of either the prompt or the model's output

The diagram below further illustrates this with an example:
![[Pasted image 20251023131145.png]]

**Let's try this with a real example:**

To generate our simple chain, we first need to create a prompt template that adheres to `Phi-3`'s expected template. Using this template, the model takes in a `system_prompt`, which generally describes what we expect from the LLM. 

Then we can use the `input_prompt` to ask the LLM specific questions:
```python
from langchain import PromptTemplate 

# Create a prompt template with the "input_prompt variable"
```

