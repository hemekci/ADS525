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

First,


