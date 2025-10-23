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

# Create a prompt template with the "input_prompt" variable
template = """<s><|user|>
	{input_prompt}<|end|>
	<|assistant|>"""
prompt = PromptTemplate(
	template = template,
	input_variables = ["input_prompt"]
)
```

To create our first chain, we can use both the prompt that we created and the LLM and chain them together:
```python
basic_chain = prompt | llm # llm is the Phi-3 GGUF we defined early in the chapter
```

To use the chain, we need to use the `invoke` function

```python
basic_chain.invoke(
	{
		"input_prompt": "Hi! My name is Maarten. What is 1 + 1?",
	}
)
```

This outputs:
```markdown
The answer to 1 + 1 is 2. It's a basic arithmetic operation where you add one unit to another, resulting in two units altogether.
```

This output gives us the response without any unnecessary tokens. Now that we have created this chain, we do not have to create the prompt template from scratch every time we use the LLM.

![[Pasted image 20251023133420.png]]


> [!info] NOTE
> While `Phi-3` and other models expect a specific template, other models like ChatGPT handles the underlying template.

#### A Chain with Multiple Prompts

In our previous example, we created a single chain consisting of a prompt template an an LLM. Because the previous prompt was pretty straightforward and contains one task (e.g., adding 1 + 1), the LLM was able to handle it with no issues. 

This is not always the case. Some prompts could be more complex and contain multiple tasks and dimensions. Giving a this as an input to an LLM could produce poor results. 

**Prompt chaining**, as seen in Chapter 6, addresses this. 

Instead of providing an LLM with a long, complex prompt, we break it down to multiple chained prompts. If you remember, the example in chapter 6 was asking the LLM to generate a *name*, a *slogan*, and a *business plan*. 

Below, this method is illustrated:
![[Pasted image 20251023135209.png]]

Let's try a new example. Assume we want to generate a story that has three components:
- A title
- A description
- A summary of the story

To generate that story, we use LangChain to describe the first component, namely the title. The first link is the only component that requires user input, we define the "summary".

```python
from langchain import LLMChain

# Create a chain for the title of our story
template = """<s><|user|>
Create a title for a story about {summary}. Only return the title.
<|end|>
<|assistant|>"""

title_prompt = PromptTemplate(template=template, input_variables=["summary"])
title = LLMChain(llm=llm, prompt = title_pompt, output_key = title)
```

Let’s run an example:
```python
title.invoke({"summary": "a girl that lost her mother"})
```

This outputs:
```markdown
{'summary': 'a girl that lost her mother',
'title': ' "Whispers of Loss: A Journey Through Grief"'}
```

Using the generated output, let's generate the next component, the "description".

```python
# Create a chain for the character description using the summary and title

template = """<s><|user|>
Describe the main character of a story about {summary} with the
title {title}. Use only two sentences.<|end|>
<|assistant|>"""

character_prompt = PromptTemplate(
	template=template, input_variables=["summary", "title"]
)
character = LLMChain(llm=llm, prompt=character_prompt, output_key="character")
```

Now let's create a short description of the story using all the components we have. 
```python
# Create a chain for the story using the summary, title, and character description
template = """<s><|user|>
Create a story about {summary} with the title {title}. The main
character is: {character}. Only return the story and it cannot be
longer than one paragraph. <|end|>
<|assistant|>"""

story_prompt = PromptTemplate(
	template=template, input_variables=["summary", "title",
	"character"]
)
story = LLMChain(llm=llm, prompt=story_prompt, output_key="story")
```

Now that we have created all three components, we can link them together to create our full chain:

```python
# Combine all three components to generate a story
llm_chain = title | character | story
```

This outputs:
```markdown
{'summary': 'a girl that lost her mother',
'title': ' "In Loving Memory: A Journey Through Grief"',
'character': ' The protagonist, Emily, is a resilient young girl who struggles to cope with her overwhelming grief after losing her beloved and caring mother at an early age. As she embarks on a journey of self-discovery and healing, she learnsvaluable life lessons from the memories and wisdom shared by those around her.',
'story': " In Loving Memory: A Journey Through Grief revolves around Emily, a resilient young girl who loses her beloved mother at an early age. Struggling to cope with overwhelming grief, she embarks on a journey of self-discovery and healing, drawing strength from the cherished memories and wisdom shared by those around her. Through this transformative process, Emily learns valuable life lessons about resilience, love, and the power of human connection, ultimately finding solace in honoring her mother's legacy while embracing a newfound sense of inner peace amidst the painful loss."}
```

#### Memory: Helping LLMs to Remember Conversations

When we use LLMs out of the box (basic inference), they will not remember what was being said in a conversation. Any information shared in one prompt will not be remembered in the next.

Let's test this with code:
```python
# Let's give the LLM our name
basic_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"})
```

```markdown
Hello Maarten! The answer to 1 + 1 is 2.
```

Now let's see if it can remember the name we have provided.

```python
# Next, we ask the LLM to reproduce the name  
basic_chain.invoke({"input_prompt": "What is my name?"})
```

```markdown
I'm sorry, but as a language model, I don't have the ability to know personal information about individuals. You can provide the name you'd like to know more about, and I can help you with information or general inquiries related to it.
```

The reason for this behavior is that these models are stateless. It does not store any memory of past interactions. **Every transaction is handled as if it were the first time.**

To make these models **stateful**, we can add specific types of memory to the chain that we created earlier. 

In this section, we will go through **two common methods** for helping LLMs to remember conversations:
- **Conversation buffer**
- **Conversation summary**
![[Pasted image 20251023144903.png]]


#### Conversation Buffer

This method is basically reminding LLMs of what has happened in the past. 

As can be seen in the diagram below, this can be done by copying the full conversation history and pasting it to the prompt.

![[Pasted image 20251023145637.png]]


In LangChain, this form of memory is called a **`ConversationBufferMemory`**. 

```python 
# Create an updated prompt template to include a chat history

template = """<s><|user|>Current conversation:{chat_history}
{input_prompt}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
	template=template,
	input_variables=["input_prompt", "chat_history"] # additional "chat_history" variable
)
```

Notice that we added an additional input variable `chat_history`. This variable holds the conversation history.

Next, we can create LangChain's `ConversationBufferMemory` and assign it to the `chat_history` variable.

```python
from langchain.memory import ConversationBufferMemory

# Define the type of memory we will use
memory = ConversationBufferMemory(memory_key="chat_history")

# Chain the LLM, prompt, and memory together
llm_chain = LLMChain(
	prompt=prompt,
	llm=llm,
	memory=memory
)
```

Let's check 

