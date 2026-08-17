# Lab 1 Worksheet — The AI Autopsy

**Your name:**
**Date:**

> **How to use this file:** save a copy as `worksheet-mine.md` and fill that in, so the blank original stays available. Keep [Module 1](../../modules/01-foundations.md) open in another tab — this is open-book.
>
> Don't open `SOLUTION.md` until you've attempted everything.

---

## Part A — Classify ten systems

Fill in the three classification columns plus one sentence of reasoning. **The reasoning matters more than the label.**

**Choose from:**

- **Technique:** `Rule-based` · `Classical ML` · `Deep Learning` · `Generative AI`
- **Breadth:** `Narrow` · `General` · `Super`
- **Paradigm:** `Supervised` · `Unsupervised` · `Reinforcement` · `Self-supervised` · `None`

---

### 1. A thermostat that turns the heating on below 18°C ✅ *(worked example)*

| | |
|---|---|
| **Technique** | Rule-based |
| **Breadth** | Narrow |
| **Paradigm** | None |

**Reasoning:** A human wrote the 18°C threshold by hand, so the system never learned anything from data. It performs exactly one task, so its competence is narrow.

---

### 2. Gmail's spam filter

| | |
|---|---|
| **Technique** | |
| **Breadth** | |
| **Paradigm** | |

**Reasoning:**

---

### 3. Spotify's "Discover Weekly" playlist

| | |
|---|---|
| **Technique** | |
| **Breadth** | |
| **Paradigm** | |

**Reasoning:**

---

### 4. Face ID unlocking an iPhone

| | |
|---|---|
| **Technique** | |
| **Breadth** | |
| **Paradigm** | |

**Reasoning:**

---

### 5. AlphaGo (the system that beat the world Go champion)

| | |
|---|---|
| **Technique** | |
| **Breadth** | |
| **Paradigm** | |

**Reasoning:**

---

### 6. ChatGPT

| | |
|---|---|
| **Technique** | |
| **Breadth** | |
| **Paradigm** | |

**Reasoning:**

---

### 7. A chess engine's hard-coded opening book (a stored list of known good first moves)

| | |
|---|---|
| **Technique** | |
| **Breadth** | |
| **Paradigm** | |

**Reasoning:**

---

### 8. Midjourney (text-to-image generation)

| | |
|---|---|
| **Technique** | |
| **Breadth** | |
| **Paradigm** | |

**Reasoning:**

---

### 9. A bank's credit-scoring model that approves or declines loan applications

| | |
|---|---|
| **Technique** | |
| **Breadth** | |
| **Paradigm** | |

**Reasoning:**

---

### 10. Tesla Autopilot

| | |
|---|---|
| **Technique** | |
| **Breadth** | |
| **Paradigm** | |

**Reasoning:**

---

## Part B — Three experiments

**Chatbot used:**
**Web search / browsing / extended thinking disabled?** `Yes` / `No` / `Couldn't disable`

---

### Experiment 1 — Does it sample, or always pick the top token?

**Prompt sent (both times):** `Write a one-sentence story about a lighthouse.`

**Response in chat 1:**

```
(paste verbatim)
```

**Response in chat 2:**

```
(paste verbatim)
```

**Were they identical, similar, or completely different?**

**What does that tell you about how the next token is chosen?**

---

### Experiment 2 — Where does its knowledge stop?

**Q1:** `What is today's date, and what is the most recent event you have reliable knowledge of?`

```
(paste response)
```

**Q2:** `What was the closing price of Apple stock yesterday?`

```
(paste response)
```

**Did it know today's date?**

**For the stock price, did it (a) give you a number, (b) refuse and explain it can't know, or (c) look it up?**

**What does your answer suggest about whether a tool is attached to this chatbot?**

---

### Experiment 3 — Does making reasoning visible change the answer?

**Version A** — `...Answer with just the number.`

```
(paste response)
```

**Version B** — `...Think step by step, showing each calculation, then give the final number.`

```
(paste response)
```

**Was version A correct?** *(The right answer is 64.)*
**Was version B correct?**
**Was the reasoning visible in B?**

---

#### Escalation — only if both A and B were already correct

Modern models handle the cookie problem either way. Push until the effect appears. Try this in two fresh chats, once with `Answer with just the number.` and once with `Think step by step.`:

```
A warehouse has 17 shelves. Each shelf holds 23 boxes.
Each box contains 41 items. 3 shelves are emptied completely,
and then 250 individual items are removed from the remaining stock.
How many items are left?
```

**Version A (just the number):**

```
(paste response)
```

**Version B (step by step):**

```
(paste response)
```

**Did the two answers agree with each other?**
**Which one would you trust, and why?**

> The point is not which is right — it's that with version B you can *check the working* and locate the exact step that went wrong. With version A you get a number and no way to audit it. That auditability is the real reason step-by-step prompting matters in production, and it's Module 5's central theme.

---

## Part C — Autopsy an AI feature you use

**Feature chosen:**

**1. What does it do?**

**2. Technique — rule-based, classical ML, deep learning, or generative? Why?**

**3. What data must it have been trained on?**

**4. Which learning paradigm, and what's your evidence?**

**5. Generative or discriminative — does it create new content, or judge existing content?**

---

## 🚀 Stretch — Teachable Machine

**Did you train a model?** `Yes` / `No`

**What two classes did you use?**

**With ~200 images per class, roughly how accurate was it?**

**With only ~10 images per class, what changed?**

**When you changed the background or lighting, what happened?**

**The shortcut test** — you trained class 1 always on the left of frame and class 2 always on the right, then swapped the positions. Did the model follow the *object* or the *position*?

**In one sentence: what did the model actually learn to detect, as opposed to what you intended it to learn?**

---

## Reflection

**One thing that surprised you:**

**One thing you're still unsure about:** *(bring this into Module 2 — and if it's still unclear by Module 4, open an issue. An unclear explanation is a bug in the material.)*
