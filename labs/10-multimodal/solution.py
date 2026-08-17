"""
solution.py - Lab 10 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import base64
import math


# ======================================================================
# TASK 1 - estimate_image_tokens
# ======================================================================

BASE_TOKENS = 85
TILE_TOKENS = 170
MAX_SIDE = 2048
TARGET_SHORT_SIDE = 768
TILE_SIZE = 512


def estimate_image_tokens(width: int, height: int, detail: str = "high") -> int:
    """Estimate the token cost of sending an image (OpenAI's algorithm)."""
    if width <= 0 or height <= 0:
        raise ValueError(f"dimensions must be positive, got {width}x{height}")
    if detail not in {"high", "low"}:
        raise ValueError(f"detail must be 'high' or 'low', got {detail!r}")

    if detail == "low":
        # A flat rate, whatever the resolution. This is the 9x saving.
        return BASE_TOKENS

    # Work in floats so the two scaling steps compose without rounding drift.
    w, h = float(width), float(height)

    # STEP 1: fit inside 2048 x 2048.
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        w, h = w * scale, h * scale

    # STEP 2: bring the SHORTEST side down to 768.
    # Note "> ", not "!=" - this only ever shrinks. That asymmetry is why a
    # 512x512 image stays at one tile instead of being blown up to 768x768,
    # and it is what makes small images genuinely cheap.
    if min(w, h) > TARGET_SHORT_SIDE:
        scale = TARGET_SHORT_SIDE / min(w, h)
        w, h = w * scale, h * scale

    # STEP 3: count 512x512 tiles, rounding UP - a partial tile still costs.
    tiles = math.ceil(w / TILE_SIZE) * math.ceil(h / TILE_SIZE)

    return BASE_TOKENS + TILE_TOKENS * tiles


# ======================================================================
# TASK 2 - plan_downscale
# ======================================================================

def plan_downscale(width: int, height: int, max_dimension: int) -> tuple:
    """Compute target dimensions that fit max_dimension, preserving aspect."""
    if max(width, height) <= max_dimension:
        return (width, height, 1.0)

    scale = max_dimension / max(width, height)

    # max(1, ...) guards an extreme aspect ratio: 10000x3 scaled to fit 100
    # would otherwise give a height of 0, and a zero-pixel image is invalid.
    return (max(1, int(width * scale)), max(1, int(height * scale)), scale)


# ======================================================================
# TASK 3 - to_data_uri
# ======================================================================

def to_data_uri(raw_bytes: bytes, mime_type: str = "image/png") -> str:
    """Encode raw bytes as a data URI for inline sending."""
    if not raw_bytes:
        # Fail here rather than at the API, where the error is far less clear.
        raise ValueError("cannot encode empty bytes")

    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def base64_overhead(raw_size: int) -> int:
    """Return the encoded size in characters for raw_size bytes."""
    return 4 * math.ceil(raw_size / 3)


# ======================================================================
# TASK 4 - build_content_blocks
# ======================================================================

def build_content_blocks(text: str, images: list = None,
                         detail: str = "high") -> list:
    """Assemble a multimodal message content list."""
    # Text FIRST: the instruction frames what the model should look for, and
    # attention reaches nearby tokens most easily (Module 10, section 10.4).
    blocks = [{"type": "text", "text": text}]

    if not images:
        return blocks

    needs_labels = len(images) > 1

    for index, url in enumerate(images, start=1):
        if needs_labels:
            # Interleave the label with the image rather than listing all the
            # images then asking. The label sits immediately beside the image
            # it names, so the model can tell them apart.
            blocks.append({"type": "text", "text": f"Image {index}:"})
        blocks.append({"type": "image_url",
                       "image_url": {"url": url, "detail": detail}})

    return blocks


# ======================================================================
# TASK 5 - validate_image_input
# ======================================================================

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def validate_image_input(raw_bytes: bytes, mime_type: str) -> tuple:
    """Screen an image before it reaches the model."""
    problems = []

    # An ALLOWLIST, as in Module 9's calculator. SVG is excluded deliberately:
    # it is XML that can carry script and external references, so it does not
    # belong in the same category as a raster image.
    if mime_type not in ALLOWED_MIME_TYPES:
        problems.append(f"mime type not allowed: {mime_type}")

    if len(raw_bytes) == 0:
        problems.append("empty file")
    elif len(raw_bytes) > MAX_IMAGE_BYTES:
        problems.append(
            f"too large: {len(raw_bytes):,} bytes (max {MAX_IMAGE_BYTES:,})")

    return (not problems, problems)


# ======================================================================
# TASK 6 - check_receipt_consistency
# ======================================================================

def check_receipt_consistency(receipt: dict, tolerance: float = 0.02) -> list:
    """Cross-check an extracted receipt. Catches misread digits."""
    problems = []

    # "is False" rather than a falsy test: a MISSING legible key means the
    # model was not asked, which is different from "reported illegible".
    if receipt.get("legible") is False:
        problems.append("model reported the image as illegible")

    subtotal = receipt.get("subtotal")
    tax = receipt.get("tax")
    total = receipt.get("total")

    # CHECK 1: the parts should equal the whole.
    if subtotal is not None and tax is not None and total is not None:
        expected = subtotal + tax
        if abs(expected - total) > tolerance:
            problems.append(
                f"subtotal + tax = {expected:.2f} but total reads {total:.2f}")

    # CHECK 2: the line items should sum to the subtotal.
    # This is the one that catches a whole-receipt decimal shift, which
    # check 1 cannot see because every field shifts together.
    line_items = receipt.get("line_items") or []
    priced = [item for item in line_items if item.get("price") is not None]

    if priced and subtotal is not None:
        line_total = sum(
            item["price"] * (item.get("quantity") or 1) for item in priced)
        # A looser tolerance: per-item rounding accumulates across items.
        if abs(line_total - subtotal) > tolerance * 3:
            problems.append(
                f"line items sum to {line_total:.2f}, "
                f"subtotal reads {subtotal:.2f}")

    # CHECK 3: a negative total is never a real receipt reading.
    if total is not None and total < 0:
        problems.append(f"total is negative: {total:.2f}")

    return problems


# ======================================================================
# BONUS - a receipt schema and a scoring harness
# ======================================================================

RECEIPT_EXTRACTION_PROMPT = """Extract this receipt into the required schema.

RULES:
- Use null for any field that is not clearly visible. Do not guess.
- Do NOT calculate values that are not printed on the receipt.
- Transcribe figures exactly as printed, including the decimal point.
- Set legible to false if the image is too poor to read reliably.
"""


def score_extraction(predicted: dict, actual: dict, fields: list = None) -> dict:
    """Compare an extraction against hand-labelled ground truth, field by field.

    Eyeballing extraction quality does not work: a fluent, well-formatted
    result with one wrong digit looks perfect. Per-field scoring against
    ground truth gives you the number that matters.

    Args:
        predicted: The model's extraction.
        actual:    Hand-labelled truth.
        fields:    Which fields to score. Defaults to the common ones.

    Returns:
        {field: bool} plus an "_accuracy" summary.
    """
    fields = fields or ["merchant", "date", "subtotal", "tax", "total"]

    result = {}
    for field in fields:
        result[field] = predicted.get(field) == actual.get(field)

    correct = sum(1 for field in fields if result[field])
    result["_accuracy"] = correct / len(fields) if fields else 0.0
    return result


# ======================================================================
# Demonstrations
# ======================================================================

def demo_token_thresholds():
    print("=" * 76)
    print("  WHERE THE TOKEN THRESHOLDS ACTUALLY ARE")
    print("=" * 76)
    print()
    print("  Square images, walking up in size. Watch where the number JUMPS.")
    print()

    print(f"  {'size':>12}{'tiles':>8}{'tokens':>9}   {'':<4}")
    print("  " + "-" * 34)

    previous = None
    for side in [256, 384, 512, 513, 640, 768, 769, 1024, 2048, 4096]:
        tokens = estimate_image_tokens(side, side, "high")
        tiles = (tokens - BASE_TOKENS) // TILE_TOKENS
        marker = "  <-- JUMP" if previous is not None and tokens != previous else ""
        print(f"  {f'{side}x{side}':>12}{tiles:>8}{tokens:>9}{marker}")
        previous = tokens

    print()
    print("  Only ONE jump, at 513. Above 768 on the short side everything is")
    print("  scaled to 768x768 = 4 tiles, so 1024, 2048 and 4096 are identical.")
    print()
    print("  Practical rule: the only size that saves you money is 512 or")
    print("  below on the SHORT side. Everything else is the same 765 tokens.")
    print()


def demo_cost_at_volume():
    print("=" * 76)
    print("  COST AT VOLUME")
    print("=" * 76)
    print()

    # Illustrative rate - check your provider's current pricing.
    price_per_million = 0.15

    print("  Illustrative input rate: $0.15 per million tokens.")
    print()
    print(f"  {'images':>12}{'1024px high':>14}{'512px high':>13}{'low detail':>13}")
    print("  " + "-" * 52)

    for count in [100, 1_000, 100_000, 1_000_000]:
        big = estimate_image_tokens(1024, 1024, "high") * count
        small = estimate_image_tokens(512, 512, "high") * count
        low = estimate_image_tokens(1024, 1024, "low") * count
        print(f"  {count:>12,}"
              f"{f'${big/1e6*price_per_million:,.2f}':>14}"
              f"{f'${small/1e6*price_per_million:,.2f}':>13}"
              f"{f'${low/1e6*price_per_million:,.2f}':>13}")

    print()
    print("  At 100 images none of this matters. At a million it decides")
    print("  whether the product is viable. Do the arithmetic before you")
    print("  deploy, not after the invoice.")
    print()


def demo_which_check_catches_what():
    print("=" * 76)
    print("  WHY THERE ARE SEVERAL CONSISTENCY CHECKS")
    print("=" * 76)
    print()

    base = {
        "legible": True, "subtotal": 10.50, "tax": 2.10, "total": 12.60,
        "line_items": [{"name": "coffee", "price": 3.50, "quantity": 3}],
    }

    mutations = [
        ("correct", base),
        ("total misread (8 for 1)", {**base, "total": 82.60}),
        ("whole receipt shifted by 10x",
         {**base, "subtotal": 105.00, "tax": 21.00, "total": 126.00}),
        ("one line item price misread",
         {**base, "line_items": [{"name": "coffee", "price": 35.00, "quantity": 3}]}),
        ("sign error", {**base, "total": -12.60}),
        ("illegible", {**base, "legible": False}),
    ]

    print(f"  {'scenario':<32}{'caught by':<24}")
    print("  " + "-" * 56)

    for label, receipt in mutations:
        problems = check_receipt_consistency(receipt)
        if not problems:
            caught = "(nothing - passes)"
        else:
            names = []
            for problem in problems:
                if "illegible" in problem:
                    names.append("legible flag")
                elif "subtotal + tax" in problem:
                    names.append("parts vs whole")
                elif "line items sum" in problem:
                    names.append("line items")
                elif "negative" in problem:
                    names.append("sign check")
            caught = ", ".join(names)
        print(f"  {label:<32}{caught:<24}")

    print()
    print("  Row 3 is the important one. A 10x shift keeps every field")
    print("  internally consistent - 105 + 21 really is 126 - so the")
    print("  parts-vs-whole check passes cleanly. Only the LINE ITEMS check")
    print("  notices that 3 coffees at 3.50 is not 105.")
    print()
    print("  Each check catches a different class of misreading. That is why")
    print("  you write several cheap ones rather than one clever one.")
    print()


def demo_content_block_shapes():
    print("=" * 76)
    print("  WHAT A MULTIMODAL MESSAGE LOOKS LIKE")
    print("=" * 76)
    print()

    for label, args in [
        ("text only", ("Summarise this.", None)),
        ("one image", ("What is in this image?", ["data:image/png;base64,AAAA"])),
        ("two images", ("What changed?", ["data:...BEFORE", "data:...AFTER"])),
    ]:
        blocks = build_content_blocks(*args)
        print(f"  {label} -> {len(blocks)} blocks")
        for block in blocks:
            if block["type"] == "text":
                print(f"    text      : {block['text']!r}")
            else:
                url = block["image_url"]["url"]
                shown = url if len(url) <= 30 else url[:27] + "..."
                print(f"    image_url : {shown}  detail={block['image_url']['detail']}")
        print()

    print("  Note the two-image case interleaves labels with images rather than")
    print("  listing both images and then asking. Each label sits immediately")
    print("  beside the image it names, which is what lets the model reason")
    print("  about 'image 1' versus 'image 2' reliably.")
    print()


if __name__ == "__main__":
    demo_token_thresholds()
    demo_cost_at_volume()
    demo_which_check_catches_what()
    demo_content_block_shapes()
