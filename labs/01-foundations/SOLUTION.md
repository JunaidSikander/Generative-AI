# Lab 1 — Solutions & Discussion

> **Attempt the lab first.** A wrong answer you reasoned through teaches more than a right answer you read. If you've done that, read on.

Three of the Part A rows have **legitimately arguable answers**, and they're flagged. If you disagree with a flagged row and your reasoning holds, you're not wrong — you're doing the exercise properly.

---

## Part A — Answer key with reasoning

### 1. Thermostat *(worked example)*

**Rule-based · Narrow · None**

A human wrote the threshold. Nothing was learned.

---

### 2. Gmail's spam filter

**Classical ML or Deep Learning · Narrow · Supervised** ⚠️ *technique is arguable*

**Reasoning:** The paradigm is unambiguous — **supervised**, and you supply the labels. Every time someone clicks "Report spam" or "Not spam," they hand Google a labelled training example. Billions of users produce an enormous labelled dataset for free.

**Why the technique is arguable:** early spam filters were famously **classical ML** — a Naive Bayes classifier over word frequencies, simple enough to implement in an afternoon. Modern Gmail uses **deep learning** on top of far richer signals (sender reputation, embedded links, image content, network-level patterns). Both answers are defensible; "classical ML historically, deep learning now" is the fullest answer.

**Discriminative**, either way: input an email, output a label. It creates nothing.

---

### 3. Spotify's "Discover Weekly"

**Classical ML + Deep Learning · Narrow · Unsupervised / self-supervised** ⚠️ *most arguable row*

**Reasoning:** This is a hybrid system and that's the lesson. Real products are rarely one clean technique.

- **Collaborative filtering** — "people whose taste overlaps yours also played this" — is closest to **unsupervised** learning over a matrix of listening behaviour. Nobody labelled anything as "good recommendation."
- It also analyses **raw audio with neural networks** (deep learning) to recommend tracks too new or obscure to have listening data — the *cold-start problem*.
- And it mines **playlist titles and text** with NLP.

**If you answered "supervised," here's the argument for you:** skips and completions can be treated as implicit labels — a skip is a negative example. Many production recommenders are trained exactly that way. This is why "implicit feedback" gets its own literature.

**Discriminative.** It selects and ranks existing tracks. It doesn't compose music.

---

### 4. Face ID

**Deep Learning · Narrow · Supervised**

**Reasoning:** A convolutional neural network over an infrared depth map — squarely **deep learning**. Apple trained the general face-recognition network on a large labelled dataset (**supervised**), and your enrolment scans adapt it to you specifically.

**Narrow** in the extreme: it answers exactly one question — is this the enrolled face, yes or no? It cannot tell you whose face it *is*, or that it's looking at a face at all in any meaningful sense.

**Discriminative.**

---

### 5. AlphaGo

**Deep Learning · Narrow · Reinforcement** *(with a supervised first stage)*

**Reasoning:** The headline paradigm is **reinforcement learning** — it improved by playing millions of games against itself, rewarded for wins.

The precise history is worth knowing:

- **AlphaGo (2016)** bootstrapped with **supervised** learning on a database of expert human games, *then* switched to reinforcement learning through self-play. So: both.
- **AlphaGo Zero (2017)** removed the human games entirely. Pure reinforcement learning from the rules alone — and it was stronger.

That second result is one of the most striking in modern AI: removing the human examples made it *better*.

**Narrow** is worth dwelling on. AlphaGo was superhuman at Go and could not play chess, explain a move, or recognise that Go is a game. **Superhuman ≠ general.** This is exactly why the technique axis and the breadth axis are separate.

---

### 6. ChatGPT

**Generative AI · Narrow · Self-supervised** *(then RLHF)*

**Reasoning:** Two-stage training:

1. **Pretraining** — self-supervised next-token prediction over a vast text corpus. The labels come free from the data (Module 1 §1.3).
2. **Alignment** — supervised fine-tuning on example conversations, then **RLHF**: humans ranked candidate responses and the model was rewarded for producing preferred ones.

So the honest full answer touches self-supervised, supervised, *and* reinforcement learning. If you wrote all three, that's the strongest answer on the sheet.

**Narrow.** Broad-seeming but narrow: one training objective, no self-set goals, no learning after training. See §1.6.

**Generative.** It produces new text.

---

### 7. A chess engine's hard-coded opening book

**Rule-based · Narrow · None**

**Reasoning:** A deliberate trap, and the point of the row. An opening book is a **stored lookup table** — "in this position, play this move" — compiled by humans from known theory. Nothing was learned. It's a dictionary with chess notation in it.

This matters because the *engine around it* may be highly sophisticated (tree search, and in modern engines a neural evaluation function). But the row asked specifically about the opening book. **Systems are made of parts, and the parts have different answers.** When someone says a product "uses AI," always ask *which part*.

---

### 8. Midjourney

**Generative AI (Deep Learning) · Narrow · Self-supervised**

**Reasoning:** A **diffusion model**, and its training is a lovely example of self-supervision: take a real image, add a known amount of random noise, and train the network to predict the noise you added. The label — the noise — is something you generated yourself, so no human annotation is needed. Repeat across billions of image–caption pairs and you get a system that can start from pure noise and denoise its way to a coherent image matching a text prompt.

**Generative**, obviously. **Narrow** — it makes images and does nothing else.

> If you're curious how text and images get connected at all, that's **contrastive learning** (CLIP), and it's covered properly in Module 10.

---

### 9. A bank's credit-scoring model

**Classical ML · Narrow · Supervised**

**Reasoning:** **Supervised** with a very clean label: historical applicants, and whether they actually defaulted. Ground truth arrives with time.

**Classical ML** — and this is the interesting part. Banks *could* use deep learning and often deliberately don't. Logistic regression and gradient-boosted trees remain standard in regulated credit decisions because they're **explainable**. In many jurisdictions a declined applicant is entitled to know why, and "the neural network said no" is not a lawful answer. So the industry frequently accepts slightly lower accuracy in exchange for auditability.

**This is a real engineering lesson, not a historical footnote.** The best model on a metric is not always the right model to deploy. Module 14 returns to this.

**Discriminative.**

---

### 10. Tesla Autopilot

**Deep Learning · Narrow · Supervised** ⚠️ *paradigm is arguable*

**Reasoning:** Firmly **deep learning** — convolutional networks over multiple camera feeds.

**Why the paradigm is arguable:** the dominant signal is **supervised** — enormous volumes of labelled driving footage, some hand-annotated, much of it auto-labelled from fleet data, plus **imitation learning** from recordings of human drivers making the right call. **Reinforcement learning** appears mainly in *simulation*, for the reason given in §1.3: real reinforcement learning requires thousands of failures, and you cannot crash real cars thousands of times to learn.

If you answered "reinforcement," the reasoning to check is whether the system can safely afford to fail. That's the discriminating question, and it's the one that matters.

**Narrow.** It drives. It cannot read a road sign to you or explain its decisions.

---

### Part A summary

| # | System | Technique | Breadth | Paradigm |
|---|--------|-----------|---------|----------|
| 1 | Thermostat | Rule-based | Narrow | None |
| 2 | Gmail spam filter | Classical ML → Deep Learning ⚠️ | Narrow | Supervised |
| 3 | Discover Weekly | Classical ML + Deep Learning ⚠️ | Narrow | Unsupervised / self-supervised ⚠️ |
| 4 | Face ID | Deep Learning | Narrow | Supervised |
| 5 | AlphaGo | Deep Learning | Narrow | Reinforcement (+ supervised) |
| 6 | ChatGPT | Generative AI | Narrow | Self-supervised (+ RLHF) |
| 7 | Chess opening book | Rule-based | Narrow | None |
| 8 | Midjourney | Generative AI | Narrow | Self-supervised |
| 9 | Credit scoring | Classical ML | Narrow | Supervised |
| 10 | Tesla Autopilot | Deep Learning | Narrow | Supervised ⚠️ |

**Every single row is Narrow.** That's not a quirk of the question set — it's the state of the field. Nothing in existence is anything else (§1.6).

---

## Part B — What you should have observed

### Experiment 1 — Sampling

**Expected:** two different sentences, both about a lighthouse, both grammatical. Perhaps very different in tone or plot.

**Why:** at each step the model produces a probability distribution over its whole vocabulary and **samples** from it (§1.7, Step 3). Different random draws → different text. The setting that controls how adventurous the sampling is, is called **temperature**, and you'll set it yourself in Module 3.

**If your two answers were nearly identical:** entirely possible and not a failure. Either the product runs at a low temperature for consistency, or a strong pattern dominates — "one-sentence story about a lighthouse" has some very high-probability shapes. Try a more open prompt (`Write one sentence about the colour blue.`) and the spread usually widens.

**The practical consequence:** an LLM is **not a function**. Same input does not guarantee same output. That single fact is why testing LLM applications is genuinely hard and why Module 11 exists.

---

### Experiment 2 — The knowledge cutoff

**The date question is a trap, and the interesting part of the experiment.**

Most chatbots **will** tell you today's date correctly. That does not mean the model knows it. The date is almost always **injected into the hidden system prompt** by the product wrapper before your message is sent. The *model* has no clock. The *product* pastes the date in as text.

This is your first sight of a distinction that runs through the whole course:

> **The model** is the text predictor. **The product** is the model plus scaffolding — system prompts, tools, memory, retrieval, guardrails. Almost everything impressive about ChatGPT-the-product is scaffolding, and Modules 5–11 teach you to build every layer of it.

**The stock price question separates the two:**

- **Refused and explained it can't know** → you're seeing the raw model. Correct behaviour.
- **Gave a specific price with a source or link** → a search tool is attached. You're seeing the product. (This is RAG and tool use — Modules 8 and 9.)
- **Gave a specific price with no source** → ⚠️ **you just caught a hallucination.** Plausible-shaped output, no grounding. Exactly §1.7.

Any of the three is a good result. The third is the most instructive.

---

### Experiment 3 — Step-by-step reasoning

**Correct answer to the cookie problem: 64.** (6 × 24 = 144; 144 − 80 = 64.)

**Correct answer to the warehouse problem: 12,952.**

```
17 shelves × 23 boxes           =    391 boxes
391 boxes × 41 items            = 16,031 items total
3 shelves emptied:
    3 × 23 × 41                 =  2,829 items removed
16,031 − 2,829                  = 13,202 items
13,202 − 250 individual items   = 12,952 items remaining
```

**Expected:** both prompts get the cookie problem right. The warehouse problem is where they diverge — and one or both may well be wrong.

**Why step-by-step helps:** the model predicts each token from the text visible to it (§1.7). Asked for "just the number," it must produce the answer in one leap, with no room for intermediate work. Asked to show its steps, `391` appears in the text, and the next stage of the calculation can be predicted *from that written number* rather than from nothing. **Writing the steps out creates the working memory the model doesn't otherwise have.**

**And the more important benefit:** with the steps visible you can find the exact line where it went wrong. With a bare number you cannot. In production that auditability usually matters more than the accuracy gain.

**The honest caveat:** models can also produce confident, plausible, *wrong* reasoning that leads to a wrong answer. Visible reasoning is not proof of correct reasoning — it's a way to check. For anything arithmetic that actually matters, give the model a calculator tool (Module 9) instead of trusting either prompt.

---

## Part C — A worked example

Here's the depth to aim for. **Feature: Google Photos search for "dog".**

**1. What it does.** Lets me type "dog" and returns every photo in my library containing a dog, without my ever having tagged one.

**2. Technique — deep learning.** No rule could describe "dog" across breeds, angles, lighting and occlusion. And the labels aren't in my library — nothing in my photos says "dog." A convolutional network trained on millions of labelled images learned visual features that generalise (§1.4).

**3. Training data.** A very large labelled image dataset — millions of images tagged with the objects they contain — assembled by Google, not from my photos. My library is only ever *inference*.

**4. Paradigm — supervised.** The image-recognition model needs images paired with correct labels, which is the definition of supervised learning. Evidence: it recognises categories it was never shown *in my library*, so the category knowledge must have come from an external labelled set.

**5. Discriminative.** It labels and retrieves existing photos. It creates nothing. *(Google Photos also has generative editing features — "remove this object" — which are a separate, generative model in the same product. Same lesson as the chess opening book: one product, multiple systems, different answers.)*

**What makes this a good answer:** it reasons from *observable evidence* — "it recognises things I never tagged, therefore the labels came from elsewhere" — rather than asserting. That inference-from-evidence is the transferable skill.

---

## 🚀 Stretch — Discussion

If you did the Teachable Machine exercise, you very likely observed:

**~200 images per class → high confidence, near-perfect accuracy.** Compelling, and slightly misleading.

**~10 images per class → noticeably worse and jumpier.** This is the data-appetite point from §1.4 made concrete. Deep learning is hungry.

**Changing background or lighting → accuracy collapses.** The important one. You believed you were training a pen detector. You were actually training *"scenes that look like the ones I recorded."* The model has no concept of a pen; it found whatever pixel patterns separated your two sets of recordings — and if you recorded one class by a window and the other under a lamp, **it may have learned to detect the lighting.**

**The shortcut test → the model follows position, not the object.** If left/right position perfectly predicted the class in your training data, the model will happily learn *position*, because that was the easiest available signal. Swap the positions at test time and it confidently gets everything backwards.

This is **spurious correlation**, and it is one of the most expensive failure modes in real machine learning. Documented real-world cases include:

- A skin-cancer classifier that learned to detect **surgical rulers**, because clinicians place a ruler beside lesions they already suspect are malignant.
- A pneumonia detector that learned to identify **which hospital** an X-ray came from, because the sicker-patient hospital used a different scanner with a distinctive image border.

Both models scored superbly in testing. Both were useless — worse, dangerous — in deployment, because the shortcut they'd learned didn't exist in the new setting.

> **The takeaway:** a model learns whatever most easily predicts the label in *your* data, not the thing you meant. It cannot know what you meant. Guarding against this is a large part of the working discipline of machine learning, and it's why Module 11 treats evaluation as a first-class engineering concern rather than a final checkbox.

You produced this failure deliberately in about two minutes. That intuition is worth more than the accuracy number.

---

## Ready for Module 2?

Check yourself. You should be able to answer these without looking:

- [ ] Name the four nested circles, outermost to innermost
- [ ] State the difference between traditional programming and machine learning in one sentence
- [ ] Explain why self-supervised learning scales better than supervised learning
- [ ] Describe how an LLM generates a sentence, in four steps
- [ ] Give two reasons an LLM hallucinates
- [ ] Explain why superhuman performance at Go is still *narrow* AI

Any gaps, reread that section of [Module 1](../../modules/01-foundations.md). No rush — the sequence only works if each module is solid before the next.

**Next up: Module 2 — Python & Your Environment.** You'll install Python, create a virtual environment, install packages, and make your first API call safely.

---

<div align="center">

**[⬅ Back to Lab 1](README.md)** · **[📖 Module 1](../../modules/01-foundations.md)** · **[🏠 README](../../README.md)**

</div>
