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

- 