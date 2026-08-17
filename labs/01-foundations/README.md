# 🧪 Lab 1: The AI Autopsy

**Module:** [1 — Foundations](../../modules/01-foundations.md)

---

## Objective

Take the four-circle map from Module 1 and use it on real systems. By the end you will have:

1. **Classified** ten real AI products by technique and by breadth
2. **Demonstrated** three of the LLM behaviours from Section 1.7 with your own experiments — not taken on trust
3. **Written** a short teardown of an AI feature you use every day
4. *(Stretch)* **Trained** an actual machine learning model, in your browser, in five minutes

## Expected outcome

A completed [`worksheet.md`](worksheet.md) — filled in with your answers, your three verbatim chatbot transcripts, and your teardown. You should finish able to hear "we're adding AI to the product" and immediately ask the three questions that matter: *learned or hand-written? generative or discriminative? one turn or a loop with tools?*

## Requirements

| | |
|---|---|
| **Software** | A web browser. Nothing else. |
| **Accounts** | One free chatbot account (see below) |
| **Python** | Not needed — that's Module 2 |
| **API key** | Not needed |
| **Cost** | Free |
| **Time** | ~25 minutes (+10 for the stretch) |

**Pick any one chatbot** — all have a free tier:

- [ChatGPT](https://chat.openai.com)
- [Claude](https://claude.ai)
- [Google Gemini](https://gemini.google.com)

> **⚠️ One setup note for Part B:** if your chatbot has web search, browsing, or "extended thinking" turned on, **turn it off** for these experiments. Those features change the mechanism you're trying to observe. If you can't disable them, note it in your worksheet — Experiment 3 will behave differently and that's a genuinely interesting result in itself.

---

## Setup

1. Open [`worksheet.md`](worksheet.md) in a text editor (Notepad, TextEdit, VS Code — anything).
2. Save a copy as `worksheet-mine.md` so you can always see the blank original.
3. Have Module 1 open in another tab for reference. **This is not a closed-book test** — looking things up is the point.

---

## Part A — Classify ten systems (10 min)

For each of the ten systems in the worksheet, fill in three columns:

| Column | Choose from | The question you're answering |
|---|---|---|
| **Technique** | Rule-based · Classical ML · Deep Learning · Generative AI | Where does it sit in the four circles? |
| **Breadth** | Narrow · General · Super | How wide is its competence? |
| **Learning paradigm** | Supervised · Unsupervised · Reinforcement · Self-supervised · None | What feedback did it learn from? |

**Worked example** — the first row is done for you:

| System | Technique | Breadth | Paradigm |
|---|---|---|---|
| A thermostat that turns on heating below 18°C | Rule-based | Narrow | None |

*Why:* a human wrote the threshold, so nothing was learned (paradigm: none, technique: rule-based). It does exactly one thing, so it's narrow.

Some rows are deliberately ambiguous, and a few are genuinely arguable. **Write one sentence of reasoning for each.** The reasoning is what's being assessed — not matching the answer key.

## Part B — Three experiments (10 min)

Prove three claims from Module 1 §1.7 yourself. Paste the **actual** responses into your worksheet, including anything that contradicts what you expected. A contradicted prediction is a better result than a confirmed one.

### Experiment 1 — Does it sample, or always pick the top token?

**Claim being tested:** generation *samples* from a probability distribution, so identical inputs can produce different outputs.

1. Start a **new** chat.
2. Send exactly: `Write a one-sentence story about a lighthouse.`
3. Record the answer.
4. Start **another new chat** (not "regenerate" — a genuinely fresh conversation).
5. Send the **exact same** sentence.
6. Record that answer too.

**Record:** both sentences verbatim. Were they identical, similar, or completely different?

### Experiment 2 — Where does its knowledge stop?

**Claim being tested:** training ended on a fixed date, and the model has no live connection to anything.

1. In a new chat, ask: `What is today's date, and what is the most recent event you have reliable knowledge of?`
2. Then ask: `What was the closing price of Apple stock yesterday?`

**Record:** both answers. Did it know today's date? Did it give you a stock price, refuse, or explain that it couldn't know? **Note carefully which of the three it did** — the difference tells you whether a tool is attached.

### Experiment 3 — Does making reasoning visible change the answer?

**Claim being tested:** each token is predicted from the visible text, so forcing intermediate steps into the text gives later tokens better material.

1. New chat. Ask exactly:
   ```
   A baker makes 6 trays of cookies. Each tray holds 24 cookies.
   He sells 80 cookies. How many are left? Answer with just the number.
   ```
2. New chat again. Ask the same question, but replace the last sentence with:
   ```
   Think step by step, showing each calculation, then give the final number.
   ```

**Record:** both answers. Were they both correct? Was the reasoning visible in the second?

> **📌 If both answers were right:** that's a valid result, not a failed experiment. This arithmetic is easy enough that a modern model handles it either way. Your worksheet asks you to then try a *harder* version — the effect shows up as difficulty increases, and finding the threshold where it starts to matter is more informative than confirming it on an easy case.

## Part C — Autopsy an AI feature you use (5 min)

Pick one AI feature you personally use — Spotify recommendations, phone face unlock, Gmail spam filtering, autocomplete, Google Photos search, a code assistant, anything.

Answer the five questions in the worksheet:

1. What does it do?
2. Technique — rule-based, classical ML, deep learning, or generative?
3. What data must it have trained on?
4. Which learning paradigm, and why?
5. Generative or discriminative?

Aim for one or two sentences each. Guessing is fine — reasoning from evidence is the skill being practised, and Part 5 of the solution file walks through a full worked example.

---

## 🚀 Stretch Challenge — Train a real model in five minutes

**Optional.** Skip guilt-free — nothing later depends on it. But this makes "learning from examples" concrete in a way reading cannot.

1. Go to [Google Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Click **Get Started** → **Image Project** → **Standard image model**.
3. You'll see two classes. Rename `Class 1` to something like `holding a pen` and `Class 2` to `empty hand`.
4. Click **Webcam** under the first class and hold the record button for ~10 seconds while holding a pen. You'll capture a few hundred images.
5. Do the same for the second class with an empty hand.
6. Click **Train Model** and wait ~30 seconds. **You are now watching supervised learning happen.**
7. Test it in the Preview panel — hold the pen up, take it away.

Then investigate — this is the actual learning:

- **Vary the amount of data.** Retrain with only ~10 images per class. Does it get worse? By how much?
- **Break it deliberately.** Move to a different background or change the lighting. Does accuracy collapse? *Why?* (Hint: what did it actually learn to detect — the pen, or the whole scene?)
- **Introduce ambiguity.** Add a third class that overlaps the others, e.g. `holding a pencil`. Watch the confidence scores go uncertain.
- **Find the shortcut.** Record class 1 always in the left of frame and class 2 always in the right. Then test with the positions swapped. Does it follow the object or the position?

**Record in your worksheet:** what you tried, and what broke. That last experiment demonstrates **spurious correlation** — a model learning an accidental shortcut in your data rather than the thing you meant. It's one of the most common and most expensive failures in real ML systems, and you can produce it on purpose in about two minutes.

---

## When you're done

1. Attempt everything **before** opening the answers. A wrong answer you reasoned through teaches more than a right one you read.
2. Then check **[`SOLUTION.md`](SOLUTION.md)** — it explains the reasoning for every row, including why three of them are legitimately arguable.
3. Move on to **Module 2: Python & Your Environment**.

**Stuck or think an answer is wrong?** Open an issue. Several Part A rows have defensible alternative answers and the solution file says which — if you disagree with the rest, you may well be right.
