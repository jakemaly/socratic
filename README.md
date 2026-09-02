# Socratic: AI Alignment & Fine-tuning

A low-rank adaptation fine-tuning example of Gemma 4 31B. View the model card on Hugging Face.

![View the model card on Hugging Face!](https://miro.medium.com/v2/resize:fit:1400/1*7y6-k8o3pwCBw-0ency01g.jpeg)(https://huggingface.co/jakemaly/Gemma4-31B-Socratic-LoRA)

*Written by me, not AI, which can explain my not-100%-precise ML language.*

---
## Two problems are frequently brought up in discussions about GenAI.


1. GenAI impedes learning, lets students take shortcuts, and is correlated with cognitive decline.

2. GenAI has a well-noted "people-pleasing" behaviour, bringing the consequence of unintended model leakages, hallucinations, etc.

## A solution: The socratic method
The socratic method, in today's context, is a form of dialogue where a teacher asks probing questions that seek to help the learner critically think and stumble upon the answer themselves.

Given the accessibility of artificial intelligence, a socratic tutor agent is a great tool to help students learn, instead of receiving and immediately forgetting the answers.

![A comparison of a basic chatbot and a Socratic tutor.](base-vs-adapter-chat.png)

### How can we implement a Socratic tutor?

The "people-pleasing" behaviour is mostly trained from **reinforcement learning from human feedback,** or **RLHF**. This makes the model more likely to leak the answer when a student asks for it, and it also **makes system prompts or basic instructions less effective.** 

This is where we look towards **Low-Rank Adaptation,** or **LoRA,** to help. LoRA is a highly-efficient training method to influence model behaviour, and works by freezing the original weights and only training a small set of adapter matrices. When combined with quantization, this allows us to make large changes to models while only using a small amount of compute.

## Training & Evaluation

![A diagram of the training and evaluation process.](training-evaluation-pipeline.png)

We implemented **supervised fine-tuning (SFT)** an open-weights model, Gemma 4 31B, on 400 synthetic socratic tutor dialogues teaching Python. I used the AdamW 8-bit algorithm and 3 epochs. 

Pre-quantized model and training notebook inspiration from [the GOAT of local AI, Unsloth.](https://unsloth.ai/docs/models/gemma-4/train)

The adapter model was benchmarked against 48 cases (4 categories, 12 per), where deepseek-v4-pro semantically judged **leakage and conceptual diagnosis**. 

## Results

We achieved 0% leakage on the testing set with training attempt 3, epoch 2. Full training results can be found at [results/summary.md](https://github.com/jakemaly/socratic/blob/main/results/summary.md).



