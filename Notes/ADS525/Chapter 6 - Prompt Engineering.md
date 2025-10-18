As discussed in previous chapters, one of the ways generative models differ from embedding/representation model is that generative models need instructions to know what to do. In this context, **these instructions are known as *prompts***. 

> In this chapter, we will have a deeper look into generative models, specifically into the realm of prompt engineering.

### Using Text Generation Models
#### Choosing a Text Generation Model
Although proprietary (e.g.,  ChatGPT, Sonnet, Gemini) are generally better than open-source models, we focus on open-source models in this books as they offer flexibility and are free to use.
![[Pasted image 20251013121258.png]]

In the above image you can see some of the most popular open-source  models and their sizes. As discussed in the previous chapter, models can be fine-tuned. These models and other open-source models have been fine-tuned for specific uses.

In this chapter, we will continue using `Phi-3-mini`, a model of 3.8 billion parameters. 

#### Loading a Text Generation Model
As we have done in the earlier chapters, we will start experimenting with models by loading them. But now, we will look closer into and focus on using and developing the prompt template. 

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer 
model = AutoModelForCausalLM.from_pretrained(
	"microsoft/Phi-3-mini-4k-instruct",
	device_map="cuda",
	torch_dtype="auto",
	trust_remote_code= True,
	)
	
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Create a pipeline
pipe = pipeline(
	"text-generation",
	model=model,
	tokenizer=tokenizer,
	return_full_text=False,
	max_new_tokens=500,
	do_sample=False,
	)
```

Let's take a look into the prompt template using the chicken prompt from Chapter 1
```python
# Prompt
messages = [
	{"role": "user", "content": "Create a funny joke about chickens."}
	]
	
# Generate the output 
output = pipe(messages)
print(output[0]["generated_text"])
```
This outputs:
```markdown
Why don't chickens like to go to the gym? Because they can't
crack the egg-sistence of it!
```

Under the hood, `transformers.pipeline` converts the messages into a specific prompt template. Let's see how this actually works by accessing the tokenizer:

```python
# Apply prompt template
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize= False)

print(prompt)
```
This outputs:
```markdown
<s><|user|>
Create a funny joke about chickens.<|end|>
<|assistant|>
```

From the Chapter 2, you probably recognize these *special tokens*: `<|user|>` and `<|assistant|>` and `<|s|>`. Remember that these tokens provide information such as the starting point, end point, etc. But in this case, they provide information on who said what. 

This prompt template was used during the training of the model. That's why each model has their own specific prompt template.
![[Pasted image 20251013124453.png]]

#### Controlling Model Output
Besides prompt engineering, another way we can influence the output of the model is by changing the configurations, better known hyperparameters. Examples are `temperature` and `top_p` in `pipe`. These hyperparameters have been heavily discussed and experimented with in previous lab notebooks.

We know that `temperature` affects the *creativity* of the model in generating the next words. A good illustration is provided below:
![[Pasted image 20251013125030.png]]

Another hyperparameter is `do_sample`, we set it to `False ` when we want the model to pick the next word with the highest probability. To make use of `temperature` and `top_p`, we set it to `True`. 


> [!tip] `top_p`, `top_k`, `temperature`, `do_sample`
>  **`top_k`**: Limits selection to the **k most likely tokens**. If
  `top_k=50`, only the 50 tokens with highest probabilities are
  considered.
  **`top_p`** (nucleus sampling): Limits selection to tokens whose
  **cumulative probability ≤ p**. Selects the smallest set of tokens
  that together have p% probability mass.
  >     If `top_p=1.0`, all tokens are considered (no filtering)
  **`temperature`**: Controls randomness/creativity. Higher temperature
  → more random (flatter distribution), lower temperature → more focused
   (sharper distribution).
  >     `temperature=0` is a special case: always picks the highest
  probability token (deterministic)
 **`do_sample`**: When set to `False`, the model uses greedy decoding
  (always picks highest probability token). When `True`, samples from
  the distribution according to temperature/top_p/top_k.

##### `temperature`
As discussed many times in previous chapters, **temperature** controls the *randomness* or *creativity* of the text generated. 
A temperature of 0 generates the same response every time because it always chooses the most likely word. 

![[Pasted image 20251014210644.png]]
You can use `temperature` in your pipeline as follows:
```python
# Using a high temperature
output = pipe(messages, do_sample=True, temperature=1)

print(output[0]["generated_text"])
```

This outputs:
```markdown
Why don't chickens like to go on a rollercoaster? Because they're afraid they might suddenly become chicken-soup!
```

> Because `temperature=1`, every time you run this code, the output will change. 

##### `top_p`
**`top_p`**, also known as *nucleus sampling*, is a sampling technique that controls which subset of tokens (*the nucleus*) the LLM can consider.  

If `top_p`=1, it will consider tokens until it reaches that value. 

Example:
```markdown
Sentence: "Have a nice..."

Next word probabilities:
"Day": 25%
"Night": 20%
"Weekend": 15%
"Trip": 10%
"Meal": 10%
"Time": 5%
"Rest": 5%
-------------
Total: 100%

If top_p=0.45, it will consider only "Day" and "Night", (0.25 + 0.20 = 0.45, more deterministic)

```
![[Pasted image 20251015041832.png]]

Similarly, the **`top_k`** parameter controls exactly how many tokens are considered. 

You can use `top_p` in your pipeline as follows:
```python
# Using a high top_p
output = pipe(messages, do_sample=True , top_p=1)

print(output[0]["generated_text"])
```
This outputs:
```
Why don't chickens make good comedians? Because their 'jokes' always 'feather' the truth!
```
###### `top_p`and `temperature` use cases  

| Example Use Case      | Temperature | top_p | Description                                                                                                                                                      |
| --------------------- | ----------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Brainstorming session | High        | High  | High randomness with large pool of potential tokens. The results will be highly diverse, often leading to very creative and unexpected results.                  |
| Email generation      | Low         | Low   | Deterministic output with high probable predicted tokens. This results in predictable, focused, and conservative outputs.                                        |
| Creative writing      | High        | Low   | High randomness with a small pool of potential tokens. This combination produces creative outputs but still remains coherent.                                    |
| Translation           | Low         | High  | Deterministic output with high probable predicted tokens. Produces coherent output with a wider range of vocabulary, leading to outputs with linguistic variety. |

### Intro to Prompt Engineering
An essential part of working with text-generative LLMs is prompt engineering. 

#### The Basic Ingredients of a Prompt
![[Pasted image 20251015054810.png]]A basic prompt consists of two components—the instruction itself and the data that relates to the instruction.
![[Pasted image 20251015055017.png]]
![[Pasted image 20251015055106.png]]

#### Instruction-Based Prompting
This is perhaps the most common, and the only type of prompting most people know. ![[Pasted image 20251015055700.png]]
![[Pasted image 20251015055732.png]]

Although each task requires different specific instructions, there is a lot of overlap in prompting techniques to improve the quality of the output. The following are some these techniques:
- **Specificity**
	Accurately describe what you want to achieve. Instead of writing "Write a description for a product.", instead write "Write a description for a product in less than two sentences and use a formal tone." As you can see, the 2nd prompt adds more information about the request - tone and length.
- **Hallucination**
	LLMs may sometimes generate incorrect information, this is known as *hallucination*. To reduce its impact, we can ask the LLM to only generate an answer if it actually knows the answer.
- **Order**
	Either begin or end your prompt with the instruction. ***Information in the middle is often forgotten.*** LLMs to focus on information at the beginning of a prompt (*primacy effect*) or at the end of a prompt (*recency effect*).
	

### Advanced Prompt Engineering

#### The Potential Complexity of a Prompt
In addition to the previously mentioned and used components such as *instructions*, *data*, and *output indicators*, there are advanced components for more complex prompts:

- **Persona**
	Describe what role the LLM should take on. It's putting the LLM in a specific character's shoes. Example: "You are a children's storyteller." If you want the LLM to use specific tone and way of writing.
-  **Instruction** 
	This is the task itself. This needs to be as specific as possible. The LLM should not have to interpret the instructions.
- **Context**
	These are additional information describing the context of the problem or task. It answers questions: "What is the reason for the instruction?"
- **Format**
	The format the LLM should use to output the generated text. Without it, the LLM would come up with a format itself.
- **Audience**
	The target of the generated text. This describes the level of the generated output. For example, for education purposes, it is helpful to use ELI5 ("Explain it like I'm 5")
- **Tone**
	The tone of voice the LLM should use in the generated text. If you are writing a formal email to your boss, you might not want to use an informal tone of voice.
- **Data**
	The main data related to the task itself. (e.g., dataset, QA list)

This is illustrated below.
![[Pasted image 20251015191541.png]]
![[Pasted image 20251015191601.png]]

You can use your own data by adding it to the `data` variable:
```python
# Prompt components
persona = "You are an expert in Large Language models. You excel at breaking down complex papers into digestible summaries. "
instruction = "Summarize the key findings of the paper provided. "
context = "Your summary should extract the most crucial points that can help researchers quickly understand the most vital information of the paper. "
data_format = "Create a bullet-point summary that outlines the method. Follow this up with a concise paragraph that encapsulates the main results. "
audience = "The summary is designed for busy researchers that quickly need to grasp the newest trends in Large Language Models. "
tone = "The tone should be professional and clear. "
text = "MY TEXT TO SUMMARIZE"
data = f"Text to summarize: text "
# The full prompt - remove and add pieces to view its impact on the generated output
query = persona + instruction + context + data_format + audience + tone + data
```
#### In-Context: Providing Examples
In the previous sections, we tried to accurately describe what the LLM should do. We can provide the LLM with examples of exactly the thing that we want to achieve. This is often referred to as ***in-context learning***, where we provide the model with correct examples.

**In-context learning** comes in a number of forms depending on how many examples you show the LLM:
1. **Zero-shot prompt:** Prompting without examples.
2. **Few-shot prompt:** Prompting with more than one example.
3. **One-shot prompt:** Prompting with a single example

![[Pasted image 20251015194630.png]]
Below, we see an example in the prompt in action. We will need to differentiate between our (`user`) question and the answers that were provided by the model  (`assistant`).  The template below is used for this process:
```python
one_shot_prompt = [
	{
		"role":"user",
		"content": "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"
	},
	{
		"role":"user",
		"content":"I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."
	},
	{
		"role":"user",
		"content":"To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:"
	}
]

print(tokenizer.apply_chat_template(one_shot_prompt, tokenize=False))
```
This outputs:
```
<s><|user|>
A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:<|end|>
<|assistant|>
I have a Gigamuru that my uncle gave me as a gift. I love to play it at home.<|end|>
<|user|>
To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:<|end|>
<|assistant|>
```

As you can see, it is important to the tokenizer and the LLM to have different roles like `user` and `assistant`. Without them, it would look like it is a conversation of a single person.

Using the conversation above, you can generate the output as follows:
```python
outputs = pipe(one_shot_prompt)
print(output[0]["generated_text"])
```
This outputs:
```
During the intense duel, the knight skillfully screeged his opponent's shield, forcing him to defend himself.
```

As you can see, this generated the answer. 

This demonstrates the effectiveness of one- or few-shot prompting. 

#### Chain Prompting: Breaking up the Problem
In the previous examples,  we deconstructed the prompts into multiple components within a prompt (i.e., persona, instruction, tone, etc.). 

While that works for most simple use cases, this technique will not be enough for highly complex prompts. 

As an example, we can ask the LLM to create a product name, slogan, and sales pitch.
While this can probably be done in one prompt, we can break it up into pieces. Doing this can also improve the quality of the response. This process is illustrated in the diagram below:

![[Pasted image 20251015213214.png]]

This technique of chaining prompts allows the LLM to focus fully on each individual question instead of dealing with multiple requirements at once. This reduces cognitive load and allows each step to build on the previous one, resulting in higher quality responses. Additionally, this technique addresses the 'lost in the middle' problem where information in the middle of long prompts receives less attention than the beginning and end.

An example in python is provided below:
```python
product_prompt = [
	{"role":"user", "content": "Create a name and slogan for a chatbot that leverages LLMs."}
]

outputs = pipe(product_prompt)
product_description = outputs[0]["generated_text"]
print(product_description)
```
This outputs:
```
Name: 'MindMeld Messenger'
Slogan: 'Unleashing Intelligent Conversations, One Response at a Time'
```

Using the generated output above, we can prompt the LLM to generate a sales pitch:
```python
# Based on the name and the slogan for a product, generate a sales pitch:

sales_prompt = [
	{"role":"user", "content": f"Generate a very short sales pitch for the following product: '{product_description}'"}
]

outputs = pipe(sales_prompt)
sales_pitch = outputs[0]["generated_text"]
print(sales_pitch)
```
This outputs:
```
Introducing MindMeld Messenger - your ultimate communication partner! Unleash intelligent conversations with our innovative AI-powered messaging platform. With MindMeld Messenger, every response is thoughtful, personalized, and timely. Say goodbye to generic replies and hello to meaningful interactions. Elevate your communication game with MindMeld Messenger - where every message is a step toward smarter conversations. Try it now and experience the future of messaging!
```

In addition to the previously mentioned benefits of using chaining, a major one is that we can give each call different parameters. For instance, **the number of tokens** created was relatively small for the name and slogan whereas the pitch can be much longer.

In addition to that, parameters like *temperature, top_p*, etc. can also be adjusted according to the prompt, isolated from the entire prompt.

This can be used for a variety of use cases, including:

- ***Response validation***
	Ask the LLM to double check previously generated outputs.
- ***Parallel prompts***
	Create multiple prompts in parallel and do a final pass to merge them. For example, ask multiple LLMs to generate multiple recipes in parallel and use the combined result to create a shopping list.
- ***Writing Stories***
	Leverage the LLM to write books or stories by breaking down the problem into components. For example, by first writing a summary, developing characters, and building the story beats before diving into creating the dialogue. 

#### Reasoning with Generative Models
**Reasoning** is a core component of human intelligence and is often compared to the emergent behavior of LLMs that often `resembles` reasoning. 

Note that we use the word `resemble`. As of the time of writing, outputs of LLMs are simply results of the data it has been trained on, and pattern matching. 

We try to mimic actual reasoning by using prompt engineering. 

To be able to effectively mimic this reasoning behavior, we first have to understand what actually constitutes `reasoning`.

	Oue methods of reasoning can be divided into system 1 and system 2 thinking processes:
- **System 1:** 