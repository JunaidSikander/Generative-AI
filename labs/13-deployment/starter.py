"""
starter.py - Lab 13: Ship It Safely

Replace each TODO with working code. The self-test checks your work.

    python starter.py

PART 1 (tasks 1-5) is pure standard library - no packages, no API key.
You will build the four components that stand between a public URL and an
unbounded bill: a token-bucket rate limiter, an LRU cache with TTL, a budget
guard, and a pre-launch checker. All take an injected clock, so they are
actually testable.

PART 2 (in the lab brief) deploys a real Gradio app to a public URL.
"""

import hashlib
import json
from collections import OrderedDict


# ======================================================================
# A controllable clock, so every component below is testable
# ======================================================================

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
# Module 13, section 13.8
# ======================================================================

class TokenBucket:
    """Allow bursts up to `capacity`, sustained at `refill_per_second`.

    The standard rate-limiting algorithm, and it has the property you want:
    a user can make a few requests back to back, but cannot sustain that rate.

        capacity 5, refill 1/second

        tokens 5 -> five rapid requests -> 0
                 -> refills at 1/second -> 3 after three seconds
    """

    def __init__(self, capacity: float, refill_per_second: float, clock=None):
        """
        Args:
            capacity:          Maximum tokens, and therefore the burst size.
            refill_per_second: Sustained rate.
            clock:             Callable returning a monotonic time. Injected
                               so tests need not sleep. Defaults to
                               time.monotonic.

        Raises:
            ValueError: on a non-positive capacity or refill rate.
        """
        import time

        # TODO:
        #   1. Raise ValueError if capacity <= 0.
        #   2. Raise ValueError if refill_per_second <= 0 - a zero rate would
        #      mean the bucket never refills, which is a permanent block
        #      dressed up as a rate limit.
        #   3. Store capacity and refill_per_second as floats.
        #   4. self.clock = clock or time.monotonic
        #   5. Start FULL: self.tokens = capacity. A new user should not be
        #      rate-limited on their first request.
        #   6. self.last_refill = self.clock()
        raise NotImplementedError

    def _refill(self) -> None:
        """Add tokens for the time elapsed since the last check.

        Lazy refill - no background timer, no thread. Tokens are computed
        from elapsed time whenever someone asks.
        """
        # TODO:
        #   now = self.clock()
        #   elapsed = now - self.last_refill
        #   if elapsed > 0:
        #       add elapsed * refill_per_second tokens, CAPPED at capacity
        #       (use min), then set last_refill = now
        #
        # The cap is what stops an idle bucket accumulating unlimited burst.
        raise NotImplementedError

    def allow(self, cost: float = 1.0) -> bool:
        """Consume `cost` tokens if available.

        Args:
            cost: How much this request is worth. Weighting by expected cost
                  bounds SPEND rather than request count (section 13.8).

        Returns:
            True if the request may proceed.

        Examples:
            >>> clock = FakeClock()
            >>> bucket = TokenBucket(2, 1.0, clock)
            >>> bucket.allow(), bucket.allow(), bucket.allow()
            (True, True, False)
        """
        # TODO: refill first, then consume if there are enough tokens.
        raise NotImplementedError

    def time_until_available(self, cost: float = 1.0) -> float:
        """Seconds until `cost` tokens will be available. 0.0 if now.

        This is what makes the limiter usable: returning it in a Retry-After
        header lets a well-behaved client back off correctly instead of
        hammering you.
        """
        # TODO: refill, return 0.0 if we already have enough, otherwise
        # (cost - self.tokens) / self.refill_per_second
        raise NotImplementedError


# ======================================================================
# TASK 2 - make_cache_key
# Module 13, section 13.7
# ======================================================================

def make_cache_key(model: str, messages: list, temperature: float = 0.0,
                   **extra) -> str:
    """Build a stable cache key for a request.

    Args:
        model:       Model identifier.
        messages:    The messages list.
        temperature: Sampling temperature.
        **extra:     Anything else that changes the response - INCLUDING the
                     user identity if responses are personalised.

    Returns:
        A 64-character hex SHA-256 digest.

    Two logically identical requests must produce the SAME key regardless of
    dict ordering, and any difference that changes the response must produce
    a DIFFERENT one.

    Examples:
        >>> a = make_cache_key("m", [{"role": "user", "content": "hi"}])
        >>> b = make_cache_key("m", [{"role": "user", "content": "hi"}])
        >>> a == b
        True
        >>> len(a)
        64
    """
    # TODO:
    #   1. Build a dict: {"model": ..., "messages": ..., "temperature": ...,
    #      plus everything in **extra}
    #   2. Serialise with json.dumps(payload, sort_keys=True, default=str).
    #      sort_keys makes it canonical so key ORDER does not change the hash.
    #      default=str stops it raising on an unexpected type.
    #   3. Return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return ""


# ======================================================================
# TASK 3 - ResponseCache
# Module 13, section 13.7
# ======================================================================

class ResponseCache:
    """An LRU cache with TTL expiry.

    Two eviction rules, and they are independent:
      - TTL:  an entry older than ttl_seconds is stale, however popular
      - LRU:  when full, the LEAST RECENTLY USED entry is dropped
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0,
                 clock=None):
        """
        Raises:
            ValueError: on a non-positive max_size or ttl_seconds.
        """
        import time

        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.clock = clock or time.monotonic

        # An OrderedDict gives us LRU for free: move_to_end marks an entry as
        # recently used, popitem(last=False) removes the oldest.
        self.entries = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def get(self, key: str):
        """Look up a key.

        Returns:
            The cached value, or None on a miss or an expired entry.

        Examples:
            >>> cache = ResponseCache(clock=FakeClock())
            >>> cache.get("a") is None
            True
            >>> cache.set("a", "A")
            >>> cache.get("a")
            'A'
        """
        # TODO:
        #   1. entry = self.entries.get(key). If None: count a miss, return None.
        #   2. Unpack (value, stored_at).
        #   3. If self.clock() - stored_at >= self.ttl_seconds:
        #        delete the entry, count an EXPIRATION and a MISS, return None.
        #   4. Otherwise: self.entries.move_to_end(key) to mark it recently
        #      used, count a hit, return the value.
        raise NotImplementedError

    def set(self, key: str, value) -> None:
        """Store a value, evicting the least recently used entry if full."""
        # TODO:
        #   1. If key already present, move_to_end first so an overwrite
        #      counts as a use.
        #   2. self.entries[key] = (value, self.clock())
        #   3. While len(self.entries) > self.max_size:
        #        self.entries.popitem(last=False)   # oldest = least recent
        #        count an eviction
        raise NotImplementedError

    def stats(self) -> dict:
        """Return cache statistics.

        Returns:
            {"hits", "misses", "size", "evictions", "expirations", "hit_rate"}
            hit_rate is 0.0 when there have been no lookups at all.
        """
        # TODO: compute hit_rate = hits / (hits + misses), guarding zero.
        return {"hits": 0, "misses": 0, "size": 0, "evictions": 0,
                "expirations": 0, "hit_rate": 0.0}


# ======================================================================
# TASK 4 - BudgetGuard
# Module 13, section 13.6
# ======================================================================

SECONDS_PER_DAY = 86_400


class BudgetGuard:
    """Track spend against a daily limit, resetting at day boundaries.

    A hard cap at the provider protects your wallet by BREAKING your app.
    A guard in your own code lets you degrade gracefully instead - serve a
    cached answer, or a polite message, rather than a 500.
    """

    def __init__(self, daily_limit: float, clock=None):
        """
        Args:
            daily_limit: Maximum spend per day, in your currency.
            clock:       Callable returning seconds since the epoch.

        Raises:
            ValueError: if daily_limit <= 0.
        """
        import time

        # TODO:
        #   1. Raise ValueError if daily_limit <= 0.
        #   2. Store daily_limit; self.clock = clock or time.time
        #   3. self.spent_today = 0.0
        #   4. self.current_day = self._day()
        raise NotImplementedError

    def _day(self) -> int:
        """Which day are we in? Integer days since the epoch."""
        # TODO: return int(self.clock() // SECONDS_PER_DAY)
        raise NotImplementedError

    def _maybe_reset(self) -> None:
        """Reset the running total if the day has changed."""
        # TODO: if self._day() differs from self.current_day, update
        # current_day and zero spent_today.
        raise NotImplementedError

    def can_spend(self, estimated_cost: float) -> bool:
        """Would this request keep us within budget?

        Examples:
            >>> guard = BudgetGuard(1.0, FakeClock())
            >>> guard.can_spend(0.5)
            True
            >>> guard.record(0.9)
            >>> guard.can_spend(0.5)
            False
        """
        # TODO: reset if needed, then return
        # spent_today + estimated_cost <= daily_limit
        raise NotImplementedError

    def record(self, actual_cost: float) -> None:
        """Record what a request actually cost.

        Check BEFORE with can_spend, record AFTER with the real figure -
        estimates and actuals differ (Module 10, section 10.2).
        """
        # TODO: reset if needed, then add actual_cost to spent_today.
        raise NotImplementedError

    def remaining(self) -> float:
        """Budget left today. Never negative."""
        # TODO: reset if needed, return max(0.0, daily_limit - spent_today)
        raise NotImplementedError


# ======================================================================
# TASK 5 - preflight_checks
# Module 13, section 13.12
# ======================================================================

def preflight_checks(config: dict) -> dict:
    """Check a deployment configuration before it goes public.

    Args:
        config: Any of these keys (missing keys are treated as unset):
            api_key_source: "env" | "hardcoded" | "file"
            env_in_gitignore: bool
            env_in_dockerignore: bool
            provider_spending_cap: float or None
            daily_budget_limit: float or None
            rate_limit_enabled: bool
            max_tokens: int or None
            max_input_chars: int or None
            health_endpoint: bool
            structured_logging: bool
            logs_user_input: bool
            dependencies_pinned: bool
            debug_mode: bool
            returns_stack_traces: bool

    Returns:
        {
          "ready": bool,          False if there are any BLOCKERS
          "blockers": list[str],  must fix before going public
          "warnings": list[str],  should fix
        }

    BLOCKERS - these bound your worst case, so nothing ships without them:
      - api_key_source is not "env"
      - env_in_gitignore is not True
      - provider_spending_cap is None
      - rate_limit_enabled is not True
      - max_tokens is None
      - returns_stack_traces is True
      - debug_mode is True

    WARNINGS - everything else:
      - env_in_dockerignore not True
      - daily_budget_limit is None
      - max_input_chars is None
      - health_endpoint not True
      - structured_logging not True
      - logs_user_input is True   (a privacy decision, not a bug)
      - dependencies_pinned not True

    Examples:
        >>> preflight_checks({})["ready"]
        False
    """
    blockers = []
    warnings = []

    # TODO:
    #   Work through the two lists above, appending a human-readable string
    #   for each failing condition. Say WHY, not just what - "no provider
    #   spending cap: an unbounded bill" beats "spending_cap missing".
    #
    #   Then return {"ready": not blockers, "blockers": ..., "warnings": ...}
    #
    # Note "ready" depends on BLOCKERS only. Warnings should not stop a
    # deploy - a checker that blocks on everything gets bypassed entirely,
    # and then it protects nothing.

    return {"ready": not blockers, "blockers": blockers, "warnings": warnings}


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
        except NotImplementedError:
            checks.append((name, "not implemented", f"raised {exception.__name__}"))
        except Exception as exc:
            checks.append((name, f"raised {type(exc).__name__}",
                           f"raised {exception.__name__}"))

    def safely(fn, default="not implemented"):
        try:
            return fn()
        except NotImplementedError:
            return default
        except Exception as exc:
            return f"raised {type(exc).__name__}: {exc}"

    # --- TASK 1 ---
    check_raises("1. rejects a non-positive capacity",
                 lambda: TokenBucket(0, 1.0, FakeClock()))
    check_raises("1. rejects a non-positive refill rate",
                 lambda: TokenBucket(5, 0, FakeClock()))

    def burst():
        clock = FakeClock()
        bucket = TokenBucket(capacity=5, refill_per_second=1.0, clock=clock)
        return [bucket.allow() for _ in range(6)]
    check("1. starts full and allows a burst of capacity", safely(burst),
          [True, True, True, True, True, False])

    def refills():
        clock = FakeClock()
        bucket = TokenBucket(5, 1.0, clock)
        for _ in range(5):
            bucket.allow()
        first = bucket.allow()          # empty
        clock.advance(1.0)
        second = bucket.allow()         # one token refilled
        third = bucket.allow()          # empty again
        return (first, second, third)
    check("1. refills over time", safely(refills), (False, True, False))

    def caps():
        clock = FakeClock()
        bucket = TokenBucket(5, 1.0, clock)
        bucket.allow()
        clock.advance(1000)             # a long idle period
        return [bucket.allow() for _ in range(6)]
    check("1. an idle bucket caps at capacity, not more", safely(caps),
          [True, True, True, True, True, False])

    def weighted():
        bucket = TokenBucket(10, 1.0, FakeClock())
        return (bucket.allow(cost=7.0), bucket.allow(cost=7.0),
                bucket.allow(cost=3.0))
    check("1. cost weighting works", safely(weighted), (True, False, True))

    def wait_time():
        clock = FakeClock()
        bucket = TokenBucket(10, 2.0, clock)
        for _ in range(10):
            bucket.allow()
        return (bucket.time_until_available(1.0), bucket.time_until_available(4.0))
    check("1. time_until_available", safely(wait_time), (0.5, 2.0))

    def no_wait_when_available():
        return TokenBucket(5, 1.0, FakeClock()).time_until_available(1.0)
    check("1. no wait when tokens are available", safely(no_wait_when_available), 0.0)

    # --- TASK 2 ---
    messages = [{"role": "user", "content": "hi"}]
    key = make_cache_key("gpt-4o-mini", messages, 0.0)
    check("2. key is a 64-char sha256 hex digest", len(key), 64)
    check("2. identical requests give identical keys",
          make_cache_key("m", messages, 0.0) == make_cache_key("m", messages, 0.0),
          True)
    check("2. a different model gives a different key",
          make_cache_key("a", messages, 0.0) != make_cache_key("b", messages, 0.0),
          True)
    check("2. a different temperature gives a different key",
          make_cache_key("m", messages, 0.0) != make_cache_key("m", messages, 0.7),
          True)
    check("2. different messages give a different key",
          make_cache_key("m", [{"role": "user", "content": "a"}])
          != make_cache_key("m", [{"role": "user", "content": "b"}]), True)
    check("2. keyword ORDER does not change the key",
          make_cache_key("m", messages, 0.0, user="u1", region="eu")
          == make_cache_key("m", messages, 0.0, region="eu", user="u1"), True)
    check("2. THE SAFETY ONE: a different user gives a different key",
          make_cache_key("m", messages, 0.0, user="alice")
          != make_cache_key("m", messages, 0.0, user="bob"), True)

    # --- TASK 3 ---
    check_raises("3. rejects a non-positive max_size",
                 lambda: ResponseCache(max_size=0, clock=FakeClock()))
    check_raises("3. rejects a non-positive ttl",
                 lambda: ResponseCache(ttl_seconds=0, clock=FakeClock()))

    def basic_cache():
        cache = ResponseCache(max_size=3, ttl_seconds=10.0, clock=FakeClock())
        miss = cache.get("a")
        cache.set("a", "A")
        return (miss, cache.get("a"))
    check("3. miss then hit", safely(basic_cache), (None, "A"))

    def ttl_expiry():
        clock = FakeClock()
        cache = ResponseCache(max_size=3, ttl_seconds=10.0, clock=clock)
        cache.set("a", "A")
        clock.advance(9.9)
        before = cache.get("a")
        clock.advance(0.2)
        after = cache.get("a")
        return (before, after, cache.expirations)
    check("3. entries expire at the ttl", safely(ttl_expiry), ("A", None, 1))

    def lru_eviction():
        cache = ResponseCache(max_size=3, ttl_seconds=1000.0, clock=FakeClock())
        for k in "abc":
            cache.set(k, k.upper())
        cache.get("a")                 # 'a' is now the most recently used
        cache.set("d", "D")            # should evict 'b', the least recent
        return (sorted(cache.entries.keys()), cache.evictions)
    check("3. evicts the LEAST RECENTLY USED entry", safely(lru_eviction),
          (["a", "c", "d"], 1))

    def stats():
        cache = ResponseCache(max_size=10, ttl_seconds=1000.0, clock=FakeClock())
        cache.set("a", "A")
        cache.get("a")
        cache.get("a")
        cache.get("missing")
        result = cache.stats()
        return (result["hits"], result["misses"], round(result["hit_rate"], 4))
    check("3. stats", safely(stats), (2, 1, 0.6667))

    def empty_stats():
        return ResponseCache(clock=FakeClock()).stats()["hit_rate"]
    check("3. hit_rate is 0.0 with no lookups", safely(empty_stats), 0.0)

    def overwrite():
        cache = ResponseCache(max_size=2, ttl_seconds=1000.0, clock=FakeClock())
        cache.set("a", "A1")
        cache.set("a", "A2")
        return (cache.get("a"), len(cache.entries))
    check("3. overwriting does not grow the cache", safely(overwrite), ("A2", 1))

    # --- TASK 4 ---
    check_raises("4. rejects a non-positive daily limit",
                 lambda: BudgetGuard(0, FakeClock()))

    def under_budget():
        guard = BudgetGuard(1.0, FakeClock())
        return (guard.can_spend(0.5), guard.remaining())
    check("4. allows spending under the limit", safely(under_budget), (True, 1.0))

    def hits_limit():
        guard = BudgetGuard(1.0, FakeClock())
        guard.record(0.9)
        return (guard.can_spend(0.05), guard.can_spend(0.5),
                round(guard.remaining(), 4))
    check("4. blocks a request that would exceed the limit",
          safely(hits_limit), (True, False, 0.1))

    def exact_limit():
        guard = BudgetGuard(1.0, FakeClock())
        guard.record(0.5)
        return guard.can_spend(0.5)     # exactly reaching the limit is allowed
    check("4. spending exactly to the limit is allowed", safely(exact_limit), True)

    def daily_reset():
        clock = FakeClock()
        guard = BudgetGuard(1.0, clock)
        guard.record(1.0)
        before = guard.can_spend(0.1)
        clock.advance(SECONDS_PER_DAY + 1)
        return (before, guard.can_spend(0.1), guard.remaining())
    check("4. resets at a day boundary", safely(daily_reset), (False, True, 1.0))

    def no_early_reset():
        clock = FakeClock()
        guard = BudgetGuard(1.0, clock)
        guard.record(1.0)
        clock.advance(3600)             # one hour, same day
        return guard.can_spend(0.1)
    check("4. does NOT reset within the same day", safely(no_early_reset), False)

    def never_negative():
        guard = BudgetGuard(1.0, FakeClock())
        guard.record(5.0)               # an overspend slipped through
        return guard.remaining()
    check("4. remaining is never negative", safely(never_negative), 0.0)

    # --- TASK 5 ---
    ready_config = {
        "api_key_source": "env",
        "env_in_gitignore": True,
        "env_in_dockerignore": True,
        "provider_spending_cap": 10.0,
        "daily_budget_limit": 1.0,
        "rate_limit_enabled": True,
        "max_tokens": 500,
        "max_input_chars": 2000,
        "health_endpoint": True,
        "structured_logging": True,
        "logs_user_input": False,
        "dependencies_pinned": True,
        "debug_mode": False,
        "returns_stack_traces": False,
    }
    result = preflight_checks(ready_config)
    check("5. a fully configured deployment is ready",
          (result["ready"], len(result["blockers"]), len(result["warnings"])),
          (True, 0, 0))

    check("5. an empty config is not ready", preflight_checks({})["ready"], False)

    for label, key, bad_value in [
        ("hardcoded key", "api_key_source", "hardcoded"),
        (".env not git-ignored", "env_in_gitignore", False),
        ("no provider spending cap", "provider_spending_cap", None),
        ("no rate limiting", "rate_limit_enabled", False),
        ("no max_tokens", "max_tokens", None),
        ("returns stack traces", "returns_stack_traces", True),
        ("debug mode on", "debug_mode", True),
    ]:
        broken = {**ready_config, key: bad_value}
        result = preflight_checks(broken)
        check(f"5. BLOCKS: {label}",
              (result["ready"], len(result["blockers"])), (False, 1))

    for label, key, bad_value in [
        (".env not docker-ignored", "env_in_dockerignore", False),
        ("no daily budget", "daily_budget_limit", None),
        ("no input length cap", "max_input_chars", None),
        ("no health endpoint", "health_endpoint", False),
        ("no structured logging", "structured_logging", False),
        ("logs user input", "logs_user_input", True),
        ("unpinned dependencies", "dependencies_pinned", False),
    ]:
        broken = {**ready_config, key: bad_value}
        result = preflight_checks(broken)
        check(f"5. WARNS (does not block): {label}",
              (result["ready"], len(result["warnings"])), (True, 1))

    # --- report ---
    print()
    print("=" * 76)
    print("  LAB 13 SELF-TEST - ship it safely")
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
        print("  You have the four components that stand between a public URL")
        print("  and an unbounded bill. Part 2 puts them behind a real app.")
    else:
        print(f"  {failures} of {len(checks)} failing.")
        print("  Order: 1 (bucket), 2 (key), 3 (cache), 4 (budget), 5 (checks).")
    print("-" * 76)
    print()
    return failures


# ======================================================================
# EXPERIMENTS
# ======================================================================

def experiment_rate_limit_under_attack():
    print("=" * 76)
    print("  EXPERIMENT 1: what a rate limiter actually saves you")
    print("=" * 76)
    print()

    COST_PER_REQUEST = 0.002        # a fraction of a cent
    clock = FakeClock()
    bucket = TokenBucket(capacity=10, refill_per_second=0.2, clock=clock)

    print("  Limit: 10 burst, then 1 request every 5 seconds.")
    print("  A script fires one request every second for 5 minutes.")
    print()

    allowed = blocked = 0
    for _ in range(300):
        if bucket.allow():
            allowed += 1
        else:
            blocked += 1
        clock.advance(1.0)

    print(f"  requests attempted : {allowed + blocked}")
    print(f"  allowed            : {allowed}")
    print(f"  blocked            : {blocked}")
    print()
    print(f"  cost WITH limiter    : ${allowed * COST_PER_REQUEST:.2f}")
    print(f"  cost WITHOUT limiter : ${(allowed + blocked) * COST_PER_REQUEST:.2f}")
    print(f"  saved                : "
          f"${blocked * COST_PER_REQUEST:.2f} "
          f"({100 * blocked / (allowed + blocked):.0f}%)")
    print()
    print("  Five minutes. Now imagine the script runs overnight, or a retry")
    print("  loop ships in a release on a Friday afternoon.")
    print()
    print("  Note the limiter did NOT block the first ten requests. A real")
    print("  user clicking around is unaffected; only sustained volume is")
    print("  bounded. That is the property that makes bursts survivable.")
    print()


def experiment_cache_value():
    print("=" * 76)
    print("  EXPERIMENT 2: why caching pays more than you expect")
    print("=" * 76)
    print()

    COST_PER_REQUEST = 0.002

    # Realistic demo traffic: a few questions asked over and over.
    popular = ["What is the refund policy?", "How do I reset my password?",
               "What are your office hours?"]
    long_tail = [f"Niche question {i}" for i in range(40)]

    import random
    rng = random.Random(7)

    traffic = []
    for _ in range(500):
        # 70% of requests hit one of three popular questions.
        if rng.random() < 0.7:
            traffic.append(rng.choice(popular))
        else:
            traffic.append(rng.choice(long_tail))

    cache = ResponseCache(max_size=100, ttl_seconds=3600.0, clock=FakeClock())

    api_calls = 0
    for question in traffic:
        key = make_cache_key("gpt-4o-mini",
                             [{"role": "user", "content": question}], 0.0)
        if cache.get(key) is None:
            api_calls += 1
            cache.set(key, f"answer to {question}")

    stats = cache.stats()
    if stats["hits"] + stats["misses"] == 0:
        print("  cache not implemented yet.")
        print()
        return

    print(f"  requests    : {len(traffic)}")
    print(f"  API calls   : {api_calls}")
    print(f"  cache hits  : {stats['hits']}")
    print(f"  hit rate    : {stats['hit_rate']:.0%}")
    print()
    print(f"  cost without cache : ${len(traffic) * COST_PER_REQUEST:.2f}")
    print(f"  cost with cache    : ${api_calls * COST_PER_REQUEST:.2f}")
    print(f"  saved              : "
          f"${(len(traffic) - api_calls) * COST_PER_REQUEST:.2f}")
    print()
    print("  Real traffic is not uniform. A handful of questions dominate,")
    print("  because people ask the obvious things and demo visitors all click")
    print("  the same example.")
    print()
    print("  That skew is exactly what a cache exploits - which is why it is")
    print("  usually the highest-return change you can make.")
    print()


def experiment_budget_guard():
    print("=" * 76)
    print("  EXPERIMENT 3: degrading gracefully instead of breaking")
    print("=" * 76)
    print()

    clock = FakeClock()
    guard = BudgetGuard(daily_limit=1.00, clock=clock)
    COST = 0.02

    print("  Daily budget: $1.00. Each request costs $0.02, so 50 requests.")
    print()

    served = refused = 0
    for request in range(1, 81):
        if guard.can_spend(COST):
            guard.record(COST)
            served += 1
        else:
            refused += 1
        if request in (25, 50, 51, 80):
            print(f"  after {request:>3} requests: served {served:>2}, "
                  f"refused {refused:>2}, remaining ${guard.remaining():.2f}")

    print()
    print("  Once the budget is gone, further requests are refused. Your app")
    print("  is still UP - it serves a cached answer, or a polite 'daily limit")
    print("  reached'. What it does not do is keep spending.")
    print()
    print("  Look closely at the numbers: 49 served, not 50, with $0.02 left.")
    print("  That is FLOATING-POINT DRIFT. Adding 0.02 to itself 49 times")
    print("  gives 0.9800000000000005, not 0.98 - so the 50th request appears")
    print("  to exceed the limit by a fraction of a cent.")
    print()
    print("  Harmless here, and not harmless in a billing system. The standard")
    print("  fix is to never store money as a float: use integer minor units")
    print("  (cents) or decimal.Decimal. Worth knowing before you write")
    print("  something that actually charges people.")
    print()

    clock.advance(SECONDS_PER_DAY)
    print(f"  next day: remaining ${guard.remaining():.2f}, "
          f"can_spend(0.02) = {guard.can_spend(0.02)}")
    print()
    print("  Compare this to hitting the PROVIDER's hard cap. There, every")
    print("  request starts failing with an API error and your app returns")
    print("  500s until someone notices.")
    print()
    print("  Both limits matter. The provider cap is the backstop that bounds")
    print("  your worst case; the app guard is what keeps the product usable")
    print("  when you approach it.")
    print()


def experiment_preflight():
    print("=" * 76)
    print("  EXPERIMENT 4: the pre-launch check on three real configurations")
    print("=" * 76)
    print()

    configurations = {
        "Friday afternoon demo": {
            "api_key_source": "hardcoded",
            "env_in_gitignore": False,
            "provider_spending_cap": None,
            "rate_limit_enabled": False,
            "max_tokens": None,
            "debug_mode": True,
            "returns_stack_traces": True,
        },
        "Nearly there": {
            "api_key_source": "env",
            "env_in_gitignore": True,
            "env_in_dockerignore": False,
            "provider_spending_cap": 20.0,
            "daily_budget_limit": None,
            "rate_limit_enabled": True,
            "max_tokens": 500,
            "max_input_chars": None,
            "health_endpoint": False,
            "structured_logging": True,
            "logs_user_input": False,
            "dependencies_pinned": True,
            "debug_mode": False,
            "returns_stack_traces": False,
        },
        "Ready to ship": {
            "api_key_source": "env", "env_in_gitignore": True,
            "env_in_dockerignore": True, "provider_spending_cap": 10.0,
            "daily_budget_limit": 1.0, "rate_limit_enabled": True,
            "max_tokens": 500, "max_input_chars": 2000,
            "health_endpoint": True, "structured_logging": True,
            "logs_user_input": False, "dependencies_pinned": True,
            "debug_mode": False, "returns_stack_traces": False,
        },
    }

    for label, config in configurations.items():
        result = preflight_checks(config)
        verdict = "READY" if result["ready"] else "NOT READY"
        print(f"  [{verdict:>9}] {label}")
        for blocker in result["blockers"]:
            print(f"              BLOCKER: {blocker}")
        for warning in result["warnings"]:
            print(f"              warning: {warning}")
        print()

    print("  The first configuration is not a straw man. Every one of those")
    print("  settings is the DEFAULT - you get a hardcoded key by pasting it")
    print("  in, no spending cap by not visiting billing, and stack traces by")
    print("  not catching anything.")
    print()
    print("  Nothing in that list requires effort to get wrong. Which is why")
    print("  the checklist exists.")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        experiment_rate_limit_under_attack()
        experiment_cache_value()
        experiment_budget_guard()
        experiment_preflight()
    else:
        print("  Fix the self-test first, then the experiments will run.")
        print()
