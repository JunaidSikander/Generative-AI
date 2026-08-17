"""
starter.py - Lab 10: Vision Extraction With Guardrails

Replace each TODO with working code. The self-test checks your work.

    python starter.py

PART 1 (tasks 1-6) is pure standard library - no packages, no API key, no
images needed. You will implement the image-token cost model and verify it
against published figures, build content blocks and data URIs, and write the
input validation and consistency checks that make vision extraction safe.

PART 2 (in the lab brief) runs real extraction against an image.
"""

import base64
import math


# ======================================================================
# TASK 1 - estimate_image_tokens
# Module 10, section 10.2
# ======================================================================

BASE_TOKENS = 85
TILE_TOKENS = 170
MAX_SIDE = 2048
TARGET_SHORT_SIDE = 768
TILE_SIZE = 512


def estimate_image_tokens(width: int, height: int, detail: str = "high") -> int:
    """Estimate the token cost of sending an image.

    This is OpenAI's documented tiling algorithm:

      1. If either side exceeds 2048px, scale to fit within 2048 x 2048.
      2. If the SHORTEST side exceeds 768px, scale so the shortest side is 768.
         Scale DOWN only - a small image is never enlarged.
      3. tiles = ceil(w / 512) * ceil(h / 512)
      4. tokens = 85 + 170 * tiles

    For detail="low" it is a flat 85 tokens, whatever the size.

    Other providers tile differently, so treat the exact numbers as
    provider-specific. The SHAPE - tiled, resolution-dependent, with a cheap
    low-detail mode - is universal.

    Args:
        width:  Image width in pixels.
        height: Image height in pixels.
        detail: "high" or "low".

    Returns:
        Estimated prompt tokens for this image alone.

    Raises:
        ValueError: on non-positive dimensions or an unknown detail level.

    Examples:
        >>> estimate_image_tokens(1024, 1024, "high")
        765
        >>> estimate_image_tokens(512, 512, "high")
        255
        >>> estimate_image_tokens(4096, 8192, "low")
        85
    """
    # TODO:
    #   1. Raise ValueError if width <= 0 or height <= 0.
    #   2. Raise ValueError if detail not in {"high", "low"}.
    #   3. detail == "low"  -> return BASE_TOKENS.
    #   4. Work in floats. If max(w, h) > MAX_SIDE, scale both by
    #      MAX_SIDE / max(w, h).
    #   5. If min(w, h) > TARGET_SHORT_SIDE, scale both by
    #      TARGET_SHORT_SIDE / min(w, h).
    #      NOTE the "> " - this only ever shrinks. That is why a 512x512 image
    #      stays at 1 tile instead of being blown up to 768x768.
    #   6. tiles = math.ceil(w / TILE_SIZE) * math.ceil(h / TILE_SIZE)
    #   7. return BASE_TOKENS + TILE_TOKENS * tiles
    return 0


# ======================================================================
# TASK 2 - plan_downscale
# Module 10, section 10.2
# ======================================================================

def plan_downscale(width: int, height: int, max_dimension: int) -> tuple:
    """Compute target dimensions that fit max_dimension, preserving aspect.

    Args:
        width:         Current width.
        height:        Current height.
        max_dimension: Neither side may exceed this.

    Returns:
        (new_width, new_height, scale) where scale is 1.0 if no change was
        needed. Dimensions are ints (truncated), and never below 1.

    Examples:
        >>> plan_downscale(4000, 3000, 1024)
        (1024, 768, 0.256)
        >>> plan_downscale(800, 600, 1024)
        (800, 600, 1.0)
    """
    # TODO:
    #   1. If max(width, height) <= max_dimension, return (width, height, 1.0).
    #   2. scale = max_dimension / max(width, height)
    #   3. Return int(width * scale), int(height * scale), scale -
    #      but clamp each dimension to a minimum of 1, so an extreme
    #      aspect ratio cannot produce a zero-pixel side.
    return (width, height, 1.0)


# ======================================================================
# TASK 3 - to_data_uri
# Module 10, section 10.3
# ======================================================================

def to_data_uri(raw_bytes: bytes, mime_type: str = "image/png") -> str:
    """Encode raw bytes as a data URI for inline sending.

    Args:
        raw_bytes: The file contents.
        mime_type: e.g. "image/png", "image/jpeg".

    Returns:
        "data:<mime_type>;base64,<base64 of raw_bytes>"

    Raises:
        ValueError: if raw_bytes is empty.

    Note that base64 inflates the payload by roughly a third: every 3 bytes
    become 4 characters (section 10.3).

    Examples:
        >>> to_data_uri(b"hello", "text/plain")
        'data:text/plain;base64,aGVsbG8='
    """
    # TODO:
    #   1. Raise ValueError on empty bytes - sending an empty image is
    #      always a bug, and it fails confusingly at the API instead.
    #   2. base64.b64encode(raw_bytes).decode("ascii")
    #   3. Return the "data:...;base64,..." string.
    return ""


def base64_overhead(raw_size: int) -> int:
    """Return the encoded size in characters for raw_size bytes.

    base64 encodes each group of 3 bytes as 4 characters, padding the final
    group. Provided so the demo can quantify the inflation.
    """
    return 4 * math.ceil(raw_size / 3)


# ======================================================================
# TASK 4 - build_content_blocks
# Module 10, sections 10.3 and 10.4
# ======================================================================

def build_content_blocks(text: str, images: list = None,
                         detail: str = "high") -> list:
    """Assemble a multimodal message content list.

    Args:
        text:   The instruction. Placed FIRST, so it frames what to look for.
        images: A list of data URIs or http(s) URLs. May be None or empty.
        detail: "high" or "low", applied to every image.

    Returns:
        A list of content blocks:
          [{"type": "text", "text": text},
           {"type": "image_url", "image_url": {"url": ..., "detail": ...}},
           ...]

        With more than ONE image, insert a label before each so the model can
        tell them apart (section 10.4):
          {"type": "text", "text": "Image 1:"}

        With exactly one image, no label is added.

    Examples:
        >>> build_content_blocks("What is this?")
        [{'type': 'text', 'text': 'What is this?'}]
        >>> blocks = build_content_blocks("Compare", ["a", "b"])
        >>> [b["type"] for b in blocks]
        ['text', 'text', 'image_url', 'text', 'image_url']
    """
    # TODO:
    #   1. Start with the text block.
    #   2. If images is falsy, return just that.
    #   3. For each image, 1-indexed:
    #        - if there is more than one image, append a
    #          {"type": "text", "text": f"Image {n}:"} label first
    #        - append {"type": "image_url",
    #                  "image_url": {"url": url, "detail": detail}}
    #   4. Return the list.
    return []


# ======================================================================
# TASK 5 - validate_image_input
# Module 10, section 10.11
# ======================================================================

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024      # 5 MB


def validate_image_input(raw_bytes: bytes, mime_type: str) -> tuple:
    """Screen an image before it reaches the model.

    The first line of defence against image-borne prompt injection and
    against accidentally sending something enormous (section 10.11).

    Args:
        raw_bytes: The file contents.
        mime_type: The declared MIME type.

    Returns:
        (is_valid, problems) - problems is a list of human-readable strings.

    Three checks: the type is allowed, the size is within budget, and the
    file is not empty.

    Examples:
        >>> validate_image_input(b"x" * 100, "image/png")
        (True, [])
        >>> ok, problems = validate_image_input(b"x" * 100, "image/svg+xml")
        >>> ok, len(problems)
        (False, 1)
    """
    problems = []

    # TODO:
    #   1. mime_type not in ALLOWED_MIME_TYPES -> one problem.
    #      An ALLOWLIST, as in Module 9: name what is permitted, refuse the
    #      rest. SVG is excluded deliberately - it can carry script.
    #   2. len(raw_bytes) == 0 -> one problem.
    #   3. len(raw_bytes) > MAX_IMAGE_BYTES -> one problem.
    #   4. Return (not problems, problems).

    return (not problems, problems)


# ======================================================================
# TASK 6 - check_receipt_consistency
# Module 10, section 10.5 - the most valuable check in the module
# ======================================================================

def check_receipt_consistency(receipt: dict, tolerance: float = 0.02) -> list:
    """Cross-check an extracted receipt. Catches misread digits.

    Vision models misread digits - 8 for 3, a decimal point in the wrong
    place - with complete confidence. Arithmetic that does not add up is a
    free signal that SOMETHING was misread, even though it cannot tell you
    which field.

    Args:
        receipt:   A dict with optional keys: legible (bool), subtotal, tax,
                   total (numbers or None), line_items (list of dicts with
                   optional name/quantity/price).
        tolerance: Allowed absolute difference, for rounding.

    Returns:
        A list of problem strings. Empty means everything checked out.

    Checks, each only when the values needed are present:
      1. legible is present and False                  -> problem
      2. subtotal + tax differs from total             -> problem
      3. sum of (price * quantity) differs from subtotal -> problem
         (a missing quantity counts as 1; items with no price are skipped)
      4. total is negative                             -> problem

    Examples:
        >>> check_receipt_consistency({"subtotal": 10.0, "tax": 2.0, "total": 12.0})
        []
        >>> len(check_receipt_consistency({"subtotal": 10.0, "tax": 2.0, "total": 99.0}))
        1
    """
    problems = []

    # TODO:
    #   1. If receipt.get("legible") is False, append a problem.
    #      Use "is False" rather than "not receipt.get(...)" - a MISSING
    #      legible key should not count as illegible.
    #   2. If subtotal, tax and total are all not None:
    #        expected = subtotal + tax
    #        if abs(expected - total) > tolerance: append a problem naming
    #        both numbers.
    #   3. Line items: collect those with a price that is not None. If any
    #      exist AND subtotal is not None:
    #        line_total = sum(price * (quantity or 1))
    #        if abs(line_total - subtotal) > tolerance * 3: append a problem.
    #        (A looser tolerance: per-item rounding accumulates.)
    #   4. If total is not None and total < 0: append a problem.
    #   5. Return problems.

    return problems


# ======================================================================
# SELF-TEST - do not edit
# ======================================================================

def _run_self_test() -> int:
    checks = []

    def check(name, got, expected):
        checks.append((name, got, expected))

    def check_raises(name, fn, exception=ValueError):
        try:
            fn()
            checks.append((name, "did not raise", f"raised {exception.__name__}"))
        except exception:
            checks.append((name, f"raised {exception.__name__}",
                           f"raised {exception.__name__}"))
        except Exception as exc:
            checks.append((name, f"raised {type(exc).__name__}",
                           f"raised {exception.__name__}"))

    # --- TASK 1: verified against OpenAI's published figures ---
    for width, height, detail, expected in [
        (1024, 1024, "high", 765),
        (2048, 4096, "high", 1105),
        (512, 512, "high", 255),
        (512, 512, "low", 85),
        (2048, 2048, "high", 765),
        (4096, 8192, "low", 85),
    ]:
        check(f"1. {width}x{height} {detail} == documented {expected}",
              estimate_image_tokens(width, height, detail), expected)

    check("1. small image is not upscaled (150x150 -> 1 tile)",
          estimate_image_tokens(150, 150, "high"), 255)
    check("1. THE KEY RESULT: 3000x4000 costs the same as 1024x1024",
          estimate_image_tokens(3000, 4000, "high"),
          estimate_image_tokens(1024, 1024, "high"))
    check("1. dropping the short side to 512 DOES save",
          estimate_image_tokens(512, 512, "high") <
          estimate_image_tokens(1024, 1024, "high"), True)
    check("1. low detail ignores size entirely",
          estimate_image_tokens(10000, 10000, "low"), 85)
    check_raises("1. rejects zero width", lambda: estimate_image_tokens(0, 100))
    check_raises("1. rejects negative height", lambda: estimate_image_tokens(100, -1))
    check_raises("1. rejects an unknown detail level",
                 lambda: estimate_image_tokens(100, 100, "medium"))

    # --- TASK 2 ---
    check("2. downscale 4000x3000 to max 1024",
          plan_downscale(4000, 3000, 1024)[:2], (1024, 768))
    check("2. downscale reports the scale factor",
          round(plan_downscale(4000, 3000, 1024)[2], 4), 0.256)
    check("2. no change when already small enough",
          plan_downscale(800, 600, 1024), (800, 600, 1.0))
    check("2. exactly at the limit is unchanged",
          plan_downscale(1024, 768, 1024), (1024, 768, 1.0))
    check("2. extreme aspect ratio never yields a zero side",
          plan_downscale(10000, 3, 100)[1] >= 1, True)

    # --- TASK 3 ---
    check("3. to_data_uri encodes correctly",
          to_data_uri(b"hello", "text/plain"),
          "data:text/plain;base64,aGVsbG8=")
    check("3. to_data_uri defaults to image/png",
          to_data_uri(b"hello").startswith("data:image/png;base64,"), True)
    check("3. to_data_uri round-trips",
          base64.b64decode(to_data_uri(b"\x89PNG\r\n", "image/png")
                           .split(",", 1)[1]), b"\x89PNG\r\n")
    check_raises("3. to_data_uri rejects empty bytes", lambda: to_data_uri(b""))

    # --- TASK 4 ---
    check("4. text only", build_content_blocks("What is this?"),
          [{"type": "text", "text": "What is this?"}])
    check("4. one image gets NO label",
          [b["type"] for b in build_content_blocks("What?", ["uri1"])],
          ["text", "image_url"])
    check("4. two images each get a label",
          [b["type"] for b in build_content_blocks("Compare", ["a", "b"])],
          ["text", "text", "image_url", "text", "image_url"])
    blocks = build_content_blocks("Compare", ["a", "b"])
    check("4. labels are 1-indexed",
          [b.get("text") for b in blocks if b["type"] == "text"],
          ["Compare", "Image 1:", "Image 2:"])
    check("4. detail is applied to the image block",
          build_content_blocks("What?", ["uri1"], detail="low")[1]["image_url"],
          {"url": "uri1", "detail": "low"})
    check("4. empty image list behaves like text only",
          build_content_blocks("What?", []),
          [{"type": "text", "text": "What?"}])
    check("4. None image list behaves like text only",
          build_content_blocks("What?", None),
          [{"type": "text", "text": "What?"}])

    # --- TASK 5 ---
    check("5. accepts a valid png", validate_image_input(b"x" * 100, "image/png"),
          (True, []))
    for label, raw, mime, expected_count in [
        ("disallowed mime (svg can carry script)", b"x" * 100, "image/svg+xml", 1),
        ("empty file", b"", "image/png", 1),
        ("oversized", b"x" * (MAX_IMAGE_BYTES + 1), "image/png", 1),
        ("empty AND wrong type", b"", "application/pdf", 2),
    ]:
        ok, problems = validate_image_input(raw, mime)
        check(f"5. rejects: {label}", (ok, len(problems)), (False, expected_count))

    check("5. accepts jpeg and webp",
          all(validate_image_input(b"x" * 10, m)[0]
              for m in ["image/jpeg", "image/webp"]), True)

    # --- TASK 6 ---
    check("6. consistent receipt has no problems",
          check_receipt_consistency({"subtotal": 10.0, "tax": 2.0, "total": 12.0}),
          [])
    check("6. tolerates rounding",
          check_receipt_consistency({"subtotal": 10.0, "tax": 2.0, "total": 12.01}),
          [])
    check("6. catches a misread total",
          len(check_receipt_consistency(
              {"subtotal": 10.0, "tax": 2.0, "total": 99.0})), 1)
    check("6. catches an illegible flag",
          len(check_receipt_consistency({"legible": False})), 1)
    check("6. a MISSING legible key is not treated as illegible",
          check_receipt_consistency({"subtotal": 10.0, "tax": 2.0, "total": 12.0}),
          [])
    check("6. legible True is fine",
          check_receipt_consistency(
              {"legible": True, "subtotal": 1.0, "tax": 0.0, "total": 1.0}), [])
    check("6. catches a negative total",
          len(check_receipt_consistency({"total": -5.0})), 1)
    check("6. skips checks when values are missing",
          check_receipt_consistency({"subtotal": 10.0, "total": None}), [])

    check("6. line items that sum correctly pass",
          check_receipt_consistency({
              "subtotal": 10.0, "tax": 2.0, "total": 12.0,
              "line_items": [{"name": "a", "price": 4.0, "quantity": 1},
                             {"name": "b", "price": 3.0, "quantity": 2}],
          }), [])
    check("6. catches line items that do not sum to the subtotal",
          len(check_receipt_consistency({
              "subtotal": 10.0,
              "line_items": [{"name": "a", "price": 4.0, "quantity": 1}],
          })), 1)
    check("6. a missing quantity counts as 1",
          check_receipt_consistency({
              "subtotal": 7.0,
              "line_items": [{"name": "a", "price": 4.0},
                             {"name": "b", "price": 3.0}],
          }), [])
    check("6. items with no price are skipped, not counted as zero",
          check_receipt_consistency({
              "subtotal": 4.0,
              "line_items": [{"name": "a", "price": 4.0},
                             {"name": "b", "price": None}],
          }), [])
    check("6. reports MULTIPLE problems together",
          len(check_receipt_consistency({
              "legible": False, "subtotal": 10.0, "tax": 2.0, "total": -99.0,
          })) >= 2, True)

    # --- report ---
    print()
    print("=" * 76)
    print("  LAB 10 SELF-TEST - vision extraction with guardrails")
    print("=" * 76)
    print()

    failures = 0
    for name, got, expected in checks:
        if got == expected:
            print(f"[ OK ]  {name}")
        else:
            failures += 1
            print(f"[FAIL]  {name}")
            print(f"          got:      {got!r}")
            print(f"          expected: {expected!r}")

    print()
    print("-" * 76)
    if failures == 0:
        print(f"  All {len(checks)} checks passed.")
        print("  Your token model matches OpenAI's published figures exactly,")
        print("  and you have the two guardrails that matter: input validation")
        print("  and arithmetic cross-checking.")
    else:
        print(f"  {failures} of {len(checks)} failing.")
        print("  Order: 1 (tokens), 2, 3, 4, 5, 6.")
    print("-" * 76)
    print()
    return failures


# ======================================================================
# EXPERIMENTS
# ======================================================================

def experiment_image_costs():
    print("=" * 76)
    print("  EXPERIMENT 1: what images actually cost")
    print("=" * 76)
    print()

    sizes = [(150, 150), (512, 512), (768, 768), (1024, 1024),
             (1536, 1536), (2048, 2048), (3000, 4000), (6000, 8000)]

    print(f"  {'dimensions':>14}{'megapixels':>12}{'high':>8}{'low':>7}{'ratio':>8}")
    print("  " + "-" * 49)

    for width, height in sizes:
        high = estimate_image_tokens(width, height, "high")
        low = estimate_image_tokens(width, height, "low")
        if high == 0:
            print("  estimate_image_tokens not implemented yet.")
            print()
            return
        megapixels = width * height / 1e6
        print(f"  {f'{width}x{height}':>14}{megapixels:>12.1f}"
              f"{high:>8}{low:>7}{high/low:>7.1f}x")

    print()
    print("  Read the last four rows. 1024x1024 through 6000x8000 all cost")
    print("  exactly 765 tokens - a 46x difference in pixels for ZERO")
    print("  difference in price, because the short side is scaled to 768")
    print("  regardless.")
    print()
    print("  So uploading a big photo buys you no extra detail and no saving.")
    print("  It only costs you upload time and payload size.")
    print()
    print("  Now look UP the table. Getting the short side to 512 or below")
    print("  drops you from 765 to 255. THAT is the lever that saves money.")
    print()


def experiment_downscale_savings():
    print("=" * 76)
    print("  EXPERIMENT 2: where downscaling actually saves anything")
    print("=" * 76)
    print()

    original = (3000, 4000)
    before = estimate_image_tokens(*original, "high")
    if before == 0:
        print("  not implemented yet.")
        print()
        return

    print(f"  original: {original[0]}x{original[1]} = {before} tokens")
    print()
    print(f"  {'max side':>10}{'result':>14}{'tokens':>9}{'saving':>9}"
          f"{'payload':>11}")
    print("  " + "-" * 53)

    # Rough proxy for file size: bytes scale with pixel count.
    original_pixels = original[0] * original[1]

    for max_dimension in [4000, 2048, 1400, 1024, 768, 512, 384]:
        width, height, scale = plan_downscale(*original, max_dimension)
        after = estimate_image_tokens(width, height, "high")
        saving = 100 * (1 - after / before)
        payload = 100 * (width * height) / original_pixels
        print(f"  {max_dimension:>10}{f'{width}x{height}':>14}{after:>9}"
              f"{saving:>8.0f}%{payload:>10.0f}%")

    print()
    print("  Two separate effects, and they do not move together:")
    print()
    print("   - TOKENS only drop at a tiling threshold. 2048 and 1400 save")
    print("     nothing at all, because the short side is still above 768.")
    print()
    print("   - PAYLOAD drops immediately and continuously. At max side 1024")
    print("     you are shipping ~7% of the pixels for the same token cost -")
    print("     which is a large latency win for zero quality loss.")
    print()
    print("  So downscale to about 1024 for SPEED, and to 512 for COST.")
    print()


def experiment_consistency_checks():
    print("=" * 76)
    print("  EXPERIMENT 3: catching misread digits")
    print("=" * 76)
    print()
    print("  Four extractions a vision model might plausibly return. Only")
    print("  arithmetic distinguishes them - all four are well-formed JSON")
    print("  and none of them look wrong.")
    print()

    extractions = [
        ("clean extraction", {
            "legible": True, "merchant": "Cafe Rio", "total": 12.60,
            "subtotal": 10.50, "tax": 2.10,
            "line_items": [{"name": "coffee", "price": 3.50, "quantity": 3}],
        }),
        ("misread a digit in the total", {
            "legible": True, "merchant": "Cafe Rio", "total": 82.60,
            "subtotal": 10.50, "tax": 2.10,
            "line_items": [{"name": "coffee", "price": 3.50, "quantity": 3}],
        }),
        ("decimal point in the wrong place", {
            "legible": True, "merchant": "Cafe Rio", "total": 126.00,
            "subtotal": 105.00, "tax": 21.00,
            "line_items": [{"name": "coffee", "price": 3.50, "quantity": 3}],
        }),
        ("model admits it could not read the image", {
            "legible": False, "merchant": None, "total": None,
            "subtotal": None, "tax": None, "line_items": [],
        }),
    ]

    for label, receipt in extractions:
        problems = check_receipt_consistency(receipt)
        verdict = "PASS" if not problems else "FLAGGED"
        print(f"  [{verdict:>7}] {label}")
        for problem in problems:
            print(f"            - {problem}")
        print()

    print("  The third case is the one to study. Every field is internally")
    print("  consistent - 105 + 21 = 126 - so the subtotal/tax/total check")
    print("  passes. It was caught by the LINE ITEMS check instead: 3 coffees")
    print("  at 3.50 is 10.50, not 105.00.")
    print()
    print("  That is why there are several independent checks. Each catches a")
    print("  different class of misreading, and a whole-receipt decimal shift")
    print("  is invisible to any single one of them.")
    print()


def experiment_base64_overhead():
    print("=" * 76)
    print("  EXPERIMENT 4: base64 payload inflation")
    print("=" * 76)
    print()

    print(f"  {'file size':>12}{'base64 chars':>15}{'inflation':>12}")
    print("  " + "-" * 39)
    for kilobytes in [10, 100, 500, 2000, 5000]:
        raw = kilobytes * 1024
        encoded = base64_overhead(raw)
        print(f"  {f'{kilobytes} KB':>12}{encoded:>15,}{encoded/raw:>11.2f}x")

    print()
    print("  Every 3 bytes become 4 characters, so a base64 request body is")
    print("  about 33% larger than the file. Fine for one image; a real")
    print("  latency problem for twenty.")
    print()
    print("  Note this is entirely separate from token cost. Downscaling a")
    print("  3000x4000 photo to 1024x768 saves zero tokens (experiment 2) and")
    print("  still shrinks the request by roughly 93%.")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        experiment_image_costs()
        experiment_downscale_savings()
        experiment_consistency_checks()
        experiment_base64_overhead()
    else:
        print("  Fix the self-test first, then the experiments will run.")
        print()
