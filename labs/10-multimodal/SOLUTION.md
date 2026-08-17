# Lab 10 — Solutions & Discussion

> **Attempt `starter.py` first.** Runnable code is in [`solution.py`](solution.py); this file explains *why*.

---

## Task 1 — `estimate_image_tokens`

```python
if detail == "low":
    return BASE_TOKENS

w, h = float(width), float(height)

if max(w, h) > MAX_SIDE:                          # step 1: fit 2048x2048
    scale = MAX_SIDE / max(w, h)
    w, h = w * scale, h * scale

if min(w, h) > TARGET_SHORT_SIDE:                 # step 2: short side -> 768
    scale = TARGET_SHORT_SIDE / min(w, h)
    w, h = w * scale, h * scale

tiles = math.ceil(w / TILE_SIZE) * math.ceil(h / TILE_SIZE)
return BASE_TOKENS + TILE_TOKENS * tiles
```

Six checks verify this against OpenAI's published figures. **If they all pass, the model is right, not approximately right** — which is unusual for something you implement from a prose description, and worth the confidence it buys.

### The subtlety in step 2

```python
if min(w, h) > TARGET_SHORT_SIDE:      # greater-than: shrink ONLY
```

This condition only ever scales **down**. A 512×512 image is never enlarged to 768×768.

That asymmetry is the whole reason small images are cheap:

| Image | Step 2 fires? | Final size | Tiles | Tokens |
|---|---|---|---|---|
| 512 × 512 | No — 512 is not > 768 | 512 × 512 | 1 | **255** |
| 1024 × 1024 | Yes — scaled to 768 | 768 × 768 | 4 | **765** |

Write `!=` instead of `>` and the 512×512 case gets upscaled to 768×768, giving 765 instead of 255. The test catches it.

### Why floats, and why `math.ceil`

**Floats** because the two scaling steps compose. A 2048×4096 image is scaled twice (by 0.5, then 0.75), and rounding to int in between would drift.

**`math.ceil`** because a partial tile still costs a full tile. A 768×1536 image is 1.5 × 3 tiles geometrically, but you're billed for 2 × 3 = 6.

### Where the thresholds actually are

`solution.py`'s first demo walks up in size:

```
          size   tiles   tokens
       256x256       1      255
       512x512       1      255
       513x513       4      765  <-- JUMP
      1024x1024      4      765
      4096x4096      4      765
```

**There is exactly one jump, at 513.** Everything from 513 upward costs 765 tokens, because the short side is scaled to 768 regardless.

> **🔑 The practical consequence: the only size that saves you money is 512 or below on the short side.** Downscaling from 4096 to 1024 feels like a big optimisation and saves precisely zero tokens.

---

## Task 2 — `plan_downscale`

```python
if max(width, height) <= max_dimension:
    return (width, height, 1.0)

scale = max_dimension / max(width, height)
return (max(1, int(width * scale)), max(1, int(height * scale)), scale)
```

### Why `max(1, ...)`

There's a test for `plan_downscale(10000, 3, 100)`. Without the clamp:

```python
int(3 * (100 / 10000)) = int(0.03) = 0
```

A zero-pixel side is an invalid image, and it fails at the encoder with a confusing error rather than here with an obvious one. **Extreme aspect ratios are rare and real** — a scanned receipt strip, a panorama, a cropped banner.

### The two-effect finding

`solution.py` and experiment 2 make a point that isn't obvious from the algorithm:

```
    max side        result   tokens   saving    payload
        2048     1536x2048      765       0%        26%
        1024      768x1024      765       0%         7%
         512       384x512      255      67%         2%
```

**Tokens and payload are independent levers:**

| Downscale to | Token saving | Payload saving | Why you'd do it |
|---|---|---|---|
| 2048 | 0% | 74% | Latency |
| 1024 | 0% | 93% | Latency — the sweet spot |
| 512 | **67%** | 98% | Cost, if accuracy survives |

**Downscale to ~1024 for speed, to 512 for cost.** Most people conflate the two and then wonder why "making the image smaller" didn't reduce their bill.

---

## Task 3 — `to_data_uri`

```python
if not raw_bytes:
    raise ValueError("cannot encode empty bytes")

encoded = base64.b64encode(raw_bytes).decode("ascii")
return f"data:{mime_type};base64,{encoded}"
```

`.decode("ascii")` because `b64encode` returns bytes, and base64 output is ASCII by definition.

**Why raise on empty:** an empty image is always a bug — a failed read, a truncated download, a wrong path. Raising here gives you the error at the cause. Sending `data:image/png;base64,` produces a confusing API error several layers away.

### The 33% inflation

Experiment 4 quantifies it: every 3 bytes become 4 characters, so `4 * ceil(n/3)` characters for `n` bytes.

```
   file size   base64 chars   inflation
     5000 KB      6,826,668       1.33x
```

A 5 MB photo becomes a ~6.7 MB request body. Fine for one image; a real latency problem for twenty — and completely separate from token cost, which is why the payload column in experiment 2 matters.

---

## Task 4 — `build_content_blocks`

```python
blocks = [{"type": "text", "text": text}]
if not images:
    return blocks

needs_labels = len(images) > 1
for index, url in enumerate(images, start=1):
    if needs_labels:
        blocks.append({"type": "text", "text": f"Image {index}:"})
    blocks.append({"type": "image_url",
                   "image_url": {"url": url, "detail": detail}})
```

### Three deliberate choices

**1. Text first.** The instruction frames what the model should look for. Module 10 §10.1: once projected, image tokens sit in one sequence with text tokens, and attention connects nearby tokens most easily. An instruction *after* the image works, but framing first is more reliable.

**2. Labels only when there are several images.** Labelling a single image adds a token cost and no information. There are separate tests for the one-image and two-image cases.

**3. Labels are interleaved, not batched.** This:

```
"Compare"  "Image 1:"  [image]  "Image 2:"  [image]
```

not this:

```
"Compare"  [image]  [image]  "Image 1 is before, image 2 is after"
```

Each label sits immediately beside the image it names. In the second form the model has to work out which image the trailing description refers to — which it does less reliably, especially with more than two.

---

## Task 5 — `validate_image_input`

```python
if mime_type not in ALLOWED_MIME_TYPES:
    problems.append(f"mime type not allowed: {mime_type}")

if len(raw_bytes) == 0:
    problems.append("empty file")
elif len(raw_bytes) > MAX_IMAGE_BYTES:
    problems.append(f"too large: {len(raw_bytes):,} bytes")
```

### An allowlist, again

`ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}`

Same principle as Module 9's calculator: **name what is permitted and refuse the rest.** A blocklist of dangerous formats would have to anticipate every one.

**SVG is excluded deliberately**, and there's a test for it. SVG is XML that can carry `<script>` and external references — it belongs in a different risk category from a raster image, even though it's nominally "an image".

### Why `elif` for the size check

`empty` and `too large` are mutually exclusive, so the second is an `elif`. The test `empty AND wrong type` expects exactly **2** problems (bad mime + empty), not 3 — confirming the size branches don't both fire.

---

## Task 6 — `check_receipt_consistency`

The most valuable function in the module.

### Why it exists

Vision models misread digits with **complete confidence**. There is no uncertainty signal to consult — the model is exactly as sure about the wrong digit as the right one.

But arithmetic is checkable. **If the numbers don't add up, something was misread** — you don't learn *which* field, and you don't need to. You learn not to trust the record, which is the signal you need to route it to a human.

This is the same principle as Module 8's citation validation: **a cheap mechanical check that catches a failure the model cannot self-report.**

### Which check catches what

`solution.py`'s third demo runs six mutations of the same receipt:

| Scenario | Caught by |
|---|---|
| correct | (passes) |
| total misread, 8 for 1 | parts vs whole |
| whole receipt shifted 10× | **line items** |
| one line item price misread | line items |
| sign error | sign check |
| illegible | legible flag |

**Row 3 is why there are several checks.** A 10× shift keeps every field internally consistent:

```
  subtotal 105.00 + tax 21.00 = total 126.00     ✓ passes parts-vs-whole
  but 3 coffees at 3.50 is 10.50, not 105.00     ✗ caught by line items
```

A single "clever" check would miss it. **Several cheap independent checks beat one sophisticated one**, because each covers a different failure class.

### The `is False` subtlety

```python
if receipt.get("legible") is False:        # ✅
if not receipt.get("legible"):             # ❌
```

The second treats a **missing** `legible` key as illegible. But missing means "the model wasn't asked" — a different thing entirely, and it would flag every extraction from a schema that doesn't include the field.

There's a dedicated test: `a MISSING legible key is not treated as illegible`.

This is the same distinction as Module 2's `None`-versus-absent problem, and it recurs constantly when validating model output: **absent, null, and false are three different things.**

### Skipping items without prices

```python
priced = [item for item in line_items if item.get("price") is not None]
```

Treating a missing price as `0.0` would make every partially-legible receipt fail the sum check — a false positive on exactly the receipts where you most want a *real* signal. There's a test: `items with no price are skipped, not counted as zero`.

### The looser tolerance on line items

```python
if abs(line_total - subtotal) > tolerance * 3:
```

Per-item rounding accumulates. Ten items each rounded to the penny can drift several pence from the printed subtotal legitimately. A tolerance that's too tight generates false positives, which train people to ignore the check — the worst outcome for a guardrail.

---

## Part 2 — Discussion

### 1. Estimate versus actual

Your estimate should be short of the actual `prompt_tokens` by roughly the length of your text prompt — the estimate covers the *image* only.

**If it's wildly off**, your provider tiles differently. That's the point of §10.2's caveat: the algorithm is OpenAI's, and the *shape* (tiled, resolution-dependent, cheap low-detail mode) is universal while the constants aren't. Printing actual usage is how you calibrate for whichever provider you're on.

### 3. `high` versus `low`

For a **clean, well-lit receipt** you may find `low` works fine — a 9× saving.

For a **crumpled or photographed** one, `low` typically loses the small print: line items and tax lines go first, totals survive longest.

**Test on your data.** "Use high for anything with text" is a reasonable default and it's an expensive one if your text is large and clear.

### 5. The bad photograph

This tests whether your escape hatch works.

**Good outcome:** `legible: False`, most fields `null`. The model reported what it could and admitted what it couldn't.

**Bad outcome:** `legible: True` and confident values. Then either your prompt isn't emphatic enough, or the model is over-confident on your kind of image — and you need the consistency checks to catch it, because the model won't.

This is Module 5 §5.4's escape-hatch rule with the stakes made visible: **"reply exactly X when you can't"** only helps if the model actually uses it.

### 6. Removing "do not calculate"

Many models will start computing a `subtotal` that isn't printed on the receipt.

**Why that's bad:** you can no longer distinguish *observed* data from *inferred* data. A computed subtotal that happens to be right is indistinguishable in your database from one that was read off the paper — and if it's wrong, your consistency check now *passes*, because the model made the arithmetic self-consistent.

> **🔑 That last consequence is worth sitting with.** Letting the model calculate doesn't just add uncertainty — it **defeats your consistency check**, because a model that computes `total = subtotal + tax` guarantees they agree. The clause is load-bearing.

---

## 🚀 Stretch — Discussion

### 1. The scoring harness

The habit that separates a demo from a system.

A fluent, well-formatted extraction with one wrong digit **looks perfect**. Per-field accuracy against hand-labelled truth is the only way to know your real number, and ten labelled receipts is about 20 minutes.

Expect field-level accuracy to vary a lot by field: `merchant` and `total` are usually near-perfect; `date` suffers from format ambiguity; individual `line_items` are the weakest. **That breakdown tells you where to spend effort** — and it's invisible without per-field scoring.

### 2. `high` versus `low`, measured

Build the table. If `low` matches `high` on your data, you've found a 9× saving with evidence. If it doesn't, you know *how much* accuracy the saving costs, which is a decision you can actually make.

### 3. Retry with the problem fed back

Works reasonably often, because the error message points at a specific inconsistency the model can go back and check.

Two caveats: cap the retries (Module 9 §9.12), and note that a model can "fix" a consistency failure by adjusting the *wrong* field — making the arithmetic agree while moving further from the truth. **A passing check after a retry is weaker evidence than a passing check first time.**

### 4. Image-borne injection

Whether the injection succeeds depends on the model, but note what protects you either way:

| Layer | What it does |
|---|---|
| **Schema** | Output must be a `Receipt`. `"APPROVED"` isn't one. |
| **Consistency check** | `total: 0.01` against line items summing to 10.50 gets flagged |
| **No action tools** | Nothing to trigger even if the injection lands |

**None of these detect the injection.** They narrow what it can achieve — which is the right shape for a defence (Module 9 §9.12), because detection is unreliable and constraint isn't.

### 5. Charts

Harder than receipts, and the failure mode is different: the model **estimates values from pixel positions** rather than reading text. So it's confidently approximate rather than occasionally wrong.

Ask for axis labels and units explicitly, and treat extracted values as estimates. **If exact numbers matter, find the underlying data** — a chart is a lossy encoding of it.

### 6. The PDF fallback

This closes a real gap. Module 8 §8.3 warned that scanned PDFs return empty text and produce a silently useless index; §10.6's fallback detects the empty extraction and routes that page through vision instead.

It's a good example of the two modules composing: **the text path is cheap and handles most pages; the vision path is expensive and handles the ones that would otherwise be lost.**

---

## Ready for Module 11?

- [ ] You can explain why a 3000×4000 photo costs the same as 1024×1024
- [ ] You know which downscale target saves tokens, and which saves only payload
- [ ] You know when `detail: low` is appropriate, and its cost ratio
- [ ] You can explain why base64 inflates a request by a third
- [ ] You can say why schema fields should be optional with a `legible` flag
- [ ] You can explain why a whole-receipt decimal shift needs the line-items check
- [ ] You know why letting the model calculate defeats your consistency check
- [ ] You know why structured output is also an injection defence

**Next: Module 11 — Guardrails, Evaluation & Responsible AI.** The consistency checks here were a preview. Module 11 turns ad-hoc checks into a real evaluation harness, and adds the safety layer for everything you've built across Modules 8, 9 and 10.

---

<div align="center">

**[⬅ Back to Lab 10](README.md)** · **[📖 Module 10](../../modules/10-multimodal.md)** · **[🏠 README](../../README.md)**

</div>
