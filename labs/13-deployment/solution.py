"""
solution.py - Lab 13 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import hashlib
import json
import time
from collections import OrderedDict


class FakeClock:
    """A clock you control. Injected wherever real time would be used."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ======================================================================
# TASK 1 - TokenBucket
# ======================================================================

class TokenBucket:
    """Allow bursts up to `capacity`, sustained at `refill_per_second`."""

    def __init__(self, capacity: float, refill_per_second: float, clock=None):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        # A zero refill rate is a permanent block dressed up as a rate limit:
        # once the initial tokens are gone they never come back.
        if refill_per_second <= 0:
            raise ValueError(
                f"refill_per_second must be positive, got {refill_per_second}")

        self.capacity = float(capacity)
        self.refill_per_second = float(refill_per_second)
        self.clock = clock or time.monotonic

        # Start FULL. A brand new user should not be rate-limited on their
        # very first request.
        self.tokens = float(capacity)
        self.last_refill = self.clock()

    def _refill(self) -> None:
        """Add tokens for the time elapsed since the last check."""
        now = self.clock()
        elapsed = now - self.last_refill

        if elapsed > 0:
            # min() caps the bucket. Without it, an idle bucket would
            # accumulate unlimited burst capacity - so a user who was quiet
            # for an hour could then fire thousands of requests at once.
            self.tokens = min(self.capacity,
                              self.tokens + elapsed * self.refill_per_second)
            self.last_refill = now

    def allow(self, cost: float = 1.0) -> bool:
        """Consume `cost` tokens if available."""
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def time_until_available(self, cost: float = 1.0) -> float:
        """Seconds until `cost` tokens will be available. 0.0 if now."""
        self._refill()
        if self.tokens >= cost:
            return 0.0
        return (cost - self.tokens) / self.refill_per_second


# ======================================================================
# TASK 2 - make_cache_key
# ======================================================================

def make_cache_key(model: str, messages: list, temperature: float = 0.0,
                   **extra) -> str:
    """Build a stable cache key for a request."""
    payload = {"model": model, "messages": messages,
               "temperature": temperature, **extra}

    # sort_keys makes the serialisation canonical, so two logically identical
    # requests hash the same regardless of dict ordering.
    # default=str stops the whole thing raising on an unexpected type.
    blob = json.dumps(payload, sort_keys=True, default=str)

    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ======================================================================
# TASK 3 - ResponseCache
# ======================================================================

class ResponseCache:
    """An LRU cache with TTL expiry."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0,
                 clock=None):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")

        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.clock = clock or time.monotonic

        # OrderedDict gives LRU for free: move_to_end marks recently used,
        # popitem(last=False) removes the oldest.
        self.entries = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def get(self, key: str):
        """Look up a key. Returns None on a miss or an expired entry."""
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None

        value, stored_at = entry

        if self.clock() - stored_at >= self.ttl_seconds:
            # Expired. Delete it now rather than leaving it to be evicted -
            # a stale entry occupying a slot is worse than an empty slot.
            del self.entries[key]
            self.expirations += 1
            self.misses += 1
            return None

        self.entries.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value) -> None:
        """Store a value, evicting the least recently used entry if full."""
        if key in self.entries:
            # An overwrite counts as a use, so move it before reassigning.
            self.entries.move_to_end(key)

        self.entries[key] = (value, self.clock())

        # while, not if: a smaller max_size set later should shrink properly.
        while len(self.entries) > self.max_size:
            self.entries.popitem(last=False)
            self.evictions += 1

    def stats(self) -> dict:
        """Return cache statistics."""
        lookups = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.entries),
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": self.hits / lookups if lookups else 0.0,
        }


# ======================================================================
# TASK 4 - BudgetGuard
# ======================================================================

SECONDS_PER_DAY = 86_400


class BudgetGuard:
    """Track spend against a daily limit, resetting at day boundaries."""

    def __init__(self, daily_limit: float, clock=None):
        if daily_limit <= 0:
            raise ValueError(f"daily_limit must be positive, got {daily_limit}")

        self.daily_limit = float(daily_limit)
        # time.time, not monotonic: we need wall-clock days, and monotonic
        # has an arbitrary origin.
        self.clock = clock or time.time

        self.spent_today = 0.0
        self.current_day = self._day()

    def _day(self) -> int:
        """Which day are we in? Integer days since the epoch."""
        return int(self.clock() // SECONDS_PER_DAY)

    def _maybe_reset(self) -> None:
        """Reset the running total if the day has changed."""
        day = self._day()
        if day != self.current_day:
            self.current_day = day
            self.spent_today = 0.0

    def can_spend(self, estimated_cost: float) -> bool:
        """Would this request keep us within budget?"""
        self._maybe_reset()
        # <= so that spending exactly to the limit is permitted.
        return self.spent_today + estimated_cost <= self.daily_limit

    def record(self, actual_cost: float) -> None:
        """Record what a request actually cost."""
        self._maybe_reset()
        self.spent_today += actual_cost

    def remaining(self) -> float:
        """Budget left today. Never negative."""
        self._maybe_reset()
        # max(0.0, ...) because an overspend can slip through: you check with
        # an ESTIMATE and record the ACTUAL, and actuals can be higher.
        return max(0.0, self.daily_limit - self.spent_today)


# ======================================================================
# TASK 5 - preflight_checks
# ======================================================================

def preflight_checks(config: dict) -> dict:
    """Check a deployment configuration before it goes public."""
    blockers = []
    warnings = []

    # --- BLOCKERS: these bound your worst case ---

    if config.get("api_key_source") != "env":
        blockers.append(
            f"api key source is {config.get('api_key_source')!r}, not 'env': "
            f"a hardcoded or file-based key ends up in your image or repo")

    if not config.get("env_in_gitignore"):
        blockers.append(
            ".env is not in .gitignore: your key will be committed")

    if config.get("provider_spending_cap") is None:
        blockers.append(
            "no provider spending cap: an unbounded bill. A retry loop does "
            "as much damage as an attacker")

    if not config.get("rate_limit_enabled"):
        blockers.append(
            "no rate limiting: one user or one bug can exhaust your budget")

    if config.get("max_tokens") is None:
        blockers.append(
            "no max_tokens: a runaway generation is unbounded, and output "
            "tokens are the expensive half")

    if config.get("returns_stack_traces"):
        blockers.append(
            "returns stack traces to callers: leaks file paths, library "
            "versions and internals")

    if config.get("debug_mode"):
        blockers.append("debug mode is on: verbose logging may include secrets")

    # --- WARNINGS: should fix, but they do not bound catastrophe ---

    if not config.get("env_in_dockerignore"):
        warnings.append(
            ".env is not in .dockerignore: COPY . . would bake the key into "
            "an image layer")

    if config.get("daily_budget_limit") is None:
        warnings.append(
            "no application budget guard: you will hit the provider cap and "
            "start returning errors instead of degrading gracefully")

    if config.get("max_input_chars") is None:
        warnings.append(
            "no input length cap: cost per request is unbounded, and context "
            "flooding is possible")

    if not config.get("health_endpoint"):
        warnings.append(
            "no health endpoint: a hung process looks healthy to the platform "
            "and keeps receiving traffic")

    if not config.get("structured_logging"):
        warnings.append(
            "no structured logging: you will have no idea what is happening")

    if config.get("logs_user_input"):
        warnings.append(
            "logs user input verbatim: a privacy decision, not a default. "
            "Log length and a hash unless you have decided otherwise")

    if not config.get("dependencies_pinned"):
        warnings.append(
            "dependencies not pinned: the next rebuild may not work")

    # "ready" depends on BLOCKERS only. A checker that blocks on everything
    # gets bypassed entirely, and then it protects nothing.
    return {"ready": not blockers, "blockers": blockers, "warnings": warnings}


# ======================================================================
# BONUS - putting them together
# ======================================================================

class ProtectedEndpoint:
    """The four components wired into one request path.

    Note the ORDER, and it matters: rate limit, then cache, then budget,
    then the expensive call. Each layer is cheaper than the one after it,
    so the cheapest way to survive a spike is not to reach the expensive part.
    """

    def __init__(self, call_model, cost_per_request: float = 0.002,
                 daily_limit: float = 1.0, clock=None):
        self.call_model = call_model
        self.cost_per_request = cost_per_request
        self.clock = clock or time.monotonic

        self.buckets = {}
        self.cache = ResponseCache(max_size=500, ttl_seconds=3600.0,
                                   clock=self.clock)
        self.budget = BudgetGuard(daily_limit, clock=self.clock)

        self.stats = {"served": 0, "rate_limited": 0, "cached": 0,
                      "budget_blocked": 0}

    def _bucket_for(self, user_id: str) -> TokenBucket:
        if user_id not in self.buckets:
            self.buckets[user_id] = TokenBucket(
                capacity=10, refill_per_second=0.2, clock=self.clock)
        return self.buckets[user_id]

    def handle(self, user_id: str, question: str, model: str = "gpt-4o-mini"):
        """Process one request through every protection layer."""
        # LAYER 1: rate limit. Costs nothing, so it goes first.
        bucket = self._bucket_for(user_id)
        if not bucket.allow():
            self.stats["rate_limited"] += 1
            return {"status": "rate_limited",
                    "retry_after": round(bucket.time_until_available(), 1)}

        # LAYER 2: cache. Note user_id is IN the key - without it you would
        # serve one user's answer to another.
        key = make_cache_key(model, [{"role": "user", "content": question}],
                             0.0, user=user_id)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats["cached"] += 1
            return {"status": "ok", "answer": cached, "cached": True}

        # LAYER 3: budget. Check with an estimate BEFORE spending.
        if not self.budget.can_spend(self.cost_per_request):
            self.stats["budget_blocked"] += 1
            return {"status": "budget_exhausted",
                    "message": "Daily limit reached. Try again tomorrow."}

        # LAYER 4: the expensive call.
        answer = self.call_model(question)
        self.budget.record(self.cost_per_request)   # record the ACTUAL cost
        self.cache.set(key, answer)
        self.stats["served"] += 1

        return {"status": "ok", "answer": answer, "cached": False}


# ======================================================================
# Demonstrations
# ======================================================================

def demo_layered_protection():
    print("=" * 76)
    print("  ALL FOUR LAYERS, ON ONE REQUEST PATH")
    print("=" * 76)
    print()

    clock = FakeClock()
    calls_made = {"count": 0}

    def fake_model(question):
        calls_made["count"] += 1
        return f"Answer to: {question}"

    endpoint = ProtectedEndpoint(fake_model, cost_per_request=0.02,
                                 daily_limit=0.20, clock=clock)

    print("  Setup: 10 burst then 1 per 5s per user, $0.20/day, $0.02/request.")
    print("  So the budget allows 10 uncached calls per day.")
    print()

    scenario = (
        [("alice", "What is the refund policy?")] * 3      # repeats -> cached
        + [("alice", f"Question {i}") for i in range(12)]  # burst -> limited
        + [("bob", "What is the refund policy?")] * 2      # different user!
    )

    for user, question in scenario:
        endpoint.handle(user, question)
        clock.advance(0.1)

    print(f"  {'outcome':<20}{'count':>7}")
    print("  " + "-" * 27)
    for outcome, count in endpoint.stats.items():
        print(f"  {outcome:<20}{count:>7}")
    print()
    print(f"  actual model calls: {calls_made['count']}")
    print(f"  total requests    : {len(scenario)}")
    print()
    print("  Alice's repeated question was served from cache after the first")
    print("  call. Her burst hit the rate limiter. And Bob asking the SAME")
    print("  question still triggered a real call - because user_id is part of")
    print("  the cache key.")
    print()
    print("  That last point is the safety one. Drop user_id from the key and")
    print("  Bob would have received Alice's answer. For a personalised app")
    print("  that is a data breach, not an optimisation.")
    print()


def demo_cache_key_safety():
    print("=" * 76)
    print("  THE CACHE KEY MISTAKE THAT LEAKS DATA")
    print("=" * 76)
    print()

    messages = [{"role": "user", "content": "What is my account balance?"}]

    unsafe_alice = make_cache_key("gpt-4o-mini", messages, 0.0)
    unsafe_bob = make_cache_key("gpt-4o-mini", messages, 0.0)

    safe_alice = make_cache_key("gpt-4o-mini", messages, 0.0, user="alice")
    safe_bob = make_cache_key("gpt-4o-mini", messages, 0.0, user="bob")

    print("  Question: 'What is my account balance?'")
    print()
    print("  WITHOUT the user in the key:")
    print(f"    alice -> {unsafe_alice[:24]}...")
    print(f"    bob   -> {unsafe_bob[:24]}...")
    print(f"    same key: {unsafe_alice == unsafe_bob}   <-- BOB GETS ALICE'S BALANCE")
    print()
    print("  WITH the user in the key:")
    print(f"    alice -> {safe_alice[:24]}...")
    print(f"    bob   -> {safe_bob[:24]}...")
    print(f"    same key: {safe_alice == safe_bob}")
    print()
    print("  Nothing errors in the first case. The cache works exactly as")
    print("  designed - it just has the wrong idea of what makes two requests")
    print("  identical.")
    print()
    print("  RULE: if the response depends on WHO is asking, the identity must")
    print("  be in the key. If you cannot be sure, do not cache.")
    print()


def demo_cost_of_each_layer():
    print("=" * 76)
    print("  WHAT EACH LAYER IS WORTH")
    print("=" * 76)
    print()

    import random

    COST = 0.002
    REQUESTS = 2000
    rng = random.Random(11)

    popular = ["refund policy", "office hours", "password reset"]
    tail = [f"niche {i}" for i in range(60)]

    traffic = []
    for _ in range(REQUESTS):
        user = f"user{rng.randint(0, 20)}"
        question = rng.choice(popular) if rng.random() < 0.6 else rng.choice(tail)
        traffic.append((user, question))

    # --- no protection at all ---
    baseline = REQUESTS * COST

    # --- cache only ---
    cache = ResponseCache(max_size=500, ttl_seconds=3600.0, clock=FakeClock())
    calls = 0
    for user, question in traffic:
        key = make_cache_key("m", [{"role": "user", "content": question}], 0.0,
                             user=user)
        if cache.get(key) is None:
            calls += 1
            cache.set(key, "answer")
    cache_only = calls * COST

    # --- cache with a SHARED key (no user) - much higher hit rate, and only
    #     safe when responses are not personalised ---
    shared_cache = ResponseCache(max_size=500, ttl_seconds=3600.0,
                                 clock=FakeClock())
    shared_calls = 0
    for _, question in traffic:
        key = make_cache_key("m", [{"role": "user", "content": question}], 0.0)
        if shared_cache.get(key) is None:
            shared_calls += 1
            shared_cache.set(key, "answer")
    shared_only = shared_calls * COST

    print(f"  {REQUESTS:,} requests from 21 users, 60% on three popular questions.")
    print()
    print(f"  {'configuration':<36}{'API calls':>11}{'cost':>10}{'saved':>9}")
    print("  " + "-" * 66)
    print(f"  {'no protection':<36}{REQUESTS:>11,}{f'${baseline:.2f}':>10}{'-':>9}")
    print(f"  {'cache keyed per user (safe)':<36}{calls:>11,}"
          f"{f'${cache_only:.2f}':>10}{f'{100*(1-cache_only/baseline):.0f}%':>9}")
    print(f"  {'cache shared (if impersonal)':<36}{shared_calls:>11,}"
          f"{f'${shared_only:.2f}':>10}{f'{100*(1-shared_only/baseline):.0f}%':>9}")
    print()
    print("  The shared cache saves far more - and it is only correct when the")
    print("  answer genuinely does not depend on who asked.")
    print()
    print("  That is the trade-off worth naming explicitly. A per-user cache is")
    print("  always safe and saves less. A shared cache saves much more and is")
    print("  a data breach if you are wrong about personalisation.")
    print()
    print("  When in doubt, key per user. The saving is not worth the risk.")
    print()


if __name__ == "__main__":
    demo_layered_protection()
    demo_cache_key_safety()
    demo_cost_of_each_layer()
