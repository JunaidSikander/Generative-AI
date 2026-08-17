"""
starter.py - Lab 6: Build Your Own Chain Framework

Replace each TODO with working code. The self-test checks your work.

    python starter.py

PART 1 (tasks 1-7) is pure Python - no packages, no API key, always runs.
By the end you will have implemented the `|` operator, parallel execution,
fallbacks and conversation memory yourself. LCEL stops being magic.

PART 2 (in the lab brief) rebuilds the same pipeline in real LangChain.
"""


# ======================================================================
# TASK 1 - the Runnable base class and the | operator
# Module 6, section 6.4
# ======================================================================

class Runnable:
    """Base class for anything that can be a step in a chain.

    The whole framework rests on one idea: every step implements invoke(),
    so any step can be composed with any other.
    """

    def invoke(self, value):
        """Run this step on a single input. Subclasses must implement it."""
        raise NotImplementedError(f"{type(self).__name__} must implement invoke()")

    def __or__(self, other):
        """Implement the | operator: `a | b` builds a sequence.

        Python calls this method for the expression `a | b`. That is the
        entire mechanism behind LCEL - there is nothing else to it.

        Returns:
            A RunnableSequence running self, then other.

        Examples:
            >>> add1 = RunnableLambda(lambda x: x + 1)
            >>> double = RunnableLambda(lambda x: x * 2)
            >>> (add1 | double).invoke(1)
            4
        """
        # TODO: return RunnableSequence([self, coerce(other)])
        # Use coerce() so plain functions and dicts work too.
        raise NotImplementedError

    def __ror__(self, other):
        """Implement `other | self` when `other` is NOT a Runnable.

        Python tries the LEFT operand's __or__ first. For the RAG shape
        `{"context": retriever, "question": passthrough} | prompt`, that means
        dict.__or__(prompt) - which fails, because dict has no idea what a
        Runnable is. Python then falls back to the RIGHT operand's __ror__.

        Without this method that RAG shape raises TypeError. Real LangChain
        defines __ror__ for exactly this reason.

        Examples:
            >>> prompt = PromptRunnable("{a}-{b}")
            >>> chain = {"a": lambda x: x, "b": lambda x: x.upper()} | prompt
            >>> chain.invoke("hi")
            'hi-HI'
        """
        # TODO: return RunnableSequence([coerce(other), self])
        # Note the ORDER - `other` runs first, because it was on the left.
        raise NotImplementedError

    def batch(self, values):
        """Run this step on many inputs.

        Examples:
            >>> RunnableLambda(lambda x: x * 2).batch([1, 2, 3])
            [2, 4, 6]
        """
        # TODO: return a list of results, one per input.
        raise NotImplementedError

    def with_fallbacks(self, fallbacks):
        """Return a version of this step that tries fallbacks on failure."""
        return RunnableWithFallbacks(self, fallbacks)


def coerce(thing):
    """Turn a plain function or dict into a Runnable.

    This is why `chain | some_function` works without wrapping it by hand,
    and why `chain | {"a": x, "b": y}` becomes a parallel step.

    Args:
        thing: A Runnable, a dict of Runnables, or a callable.

    Returns:
        A Runnable.

    Raises:
        TypeError: if it cannot be converted.
    """
    # TODO:
    #   - already a Runnable      -> return it unchanged
    #   - a dict                  -> RunnableParallel(**thing)
    #   - callable (function)     -> RunnableLambda(thing)
    #   - anything else           -> raise TypeError
    # Check Runnable FIRST: a Runnable instance is not callable here, but
    # checking in the wrong order still makes the logic harder to follow.
    raise NotImplementedError


# ======================================================================
# TASK 2 - RunnableSequence
# Module 6, section 6.4
# ======================================================================

class RunnableSequence(Runnable):
    """Runs steps one after another, threading output into input."""

    def __init__(self, steps):
        self.steps = list(steps)

    def invoke(self, value):
        """Run every step in order.

        Examples:
            >>> add1 = RunnableLambda(lambda x: x + 1)
            >>> RunnableSequence([add1, add1, add1]).invoke(0)
            3
        """
        # TODO: pass `value` through each step in self.steps, feeding each
        # step's output into the next, then return the final result.
        raise NotImplementedError

    def __or__(self, other):
        """Extend this sequence instead of nesting a new one inside it.

        Without this, `a | b | c` builds Sequence([Sequence([a, b]), c]) -
        which works but is harder to inspect and debug.

        Examples:
            >>> add1 = RunnableLambda(lambda x: x + 1)
            >>> len((add1 | add1 | add1).steps)
            3
        """
        # TODO: return RunnableSequence(self.steps + [coerce(other)])
        raise NotImplementedError

    def __ror__(self, other):
        """Prepend a step, rather than nesting this sequence inside a new one."""
        # TODO: return RunnableSequence([coerce(other)] + self.steps)
        raise NotImplementedError


# ======================================================================
# TASK 3 - RunnableLambda
# Module 6, section 6.7
# ======================================================================

class RunnableLambda(Runnable):
    """Wraps any plain Python function as a chain step.

    This is the escape hatch: anything the framework does not provide,
    you drop in as a function.
    """

    def __init__(self, function, name: str = None):
        self.function = function
        self.name = name or getattr(function, "__name__", "lambda")

    def invoke(self, value):
        """Call the wrapped function on the input.

        Examples:
            >>> RunnableLambda(str.upper).invoke("hi")
            'HI'
        """
        # TODO: call self.function on value and return the result.
        raise NotImplementedError


# ======================================================================
# TASK 4 - RunnableParallel
# Module 6, section 6.8
# ======================================================================

class RunnableParallel(Runnable):
    """Runs several steps on the SAME input and collects results by key.

    Real LangChain runs these concurrently. We run them sequentially, which
    keeps the code readable - the important part is the fan-out SHAPE, not
    the threading.
    """

    def __init__(self, **steps):
        # Each value gets coerced, so you can pass plain functions.
        self.steps = {key: coerce(value) for key, value in steps.items()}

    def invoke(self, value):
        """Run every step on the same input; return a dict of results.

        Examples:
            >>> add1 = RunnableLambda(lambda x: x + 1)
            >>> double = RunnableLambda(lambda x: x * 2)
            >>> RunnableParallel(a=add1, b=double).invoke(3)
            {'a': 4, 'b': 6}
        """
        # TODO: return {key: step.invoke(value) for each key/step}
        # NOTE: every step gets the SAME original input, not each other's
        # output. That is what makes this parallel rather than sequential.
        raise NotImplementedError


class RunnablePassthrough(Runnable):
    """Returns its input unchanged.

    Useful for carrying the original input alongside computed values -
    the shape RAG pipelines use (Module 6, section 6.8, pattern 3).
    """

    def invoke(self, value):
        return value


# ======================================================================
# TASK 5 - PromptRunnable
# Module 6, section 6.6
# ======================================================================

class PromptRunnable(Runnable):
    """Fills a template string from a dict of variables."""

    def __init__(self, template: str):
        self.template = template

    def invoke(self, value):
        """Render the template.

        Args:
            value: A dict of variables.

        Returns:
            The rendered string.

        Raises:
            KeyError: if a variable is missing. Let this propagate - a
                prompt containing a literal "{topic}" produces plausible
                nonsense, which is far worse than a crash (Lab 5, task 2).

        Examples:
            >>> PromptRunnable("Explain {topic}.").invoke({"topic": "RAG"})
            'Explain RAG.'
        """
        # TODO: return self.template.format(**value)
        raise NotImplementedError


class FakeModel(Runnable):
    """A deterministic stand-in for a real model, so tests need no API key.

    Real models are stochastic, which makes them useless for unit tests.
    Swapping in a fake at this boundary is how you test a chain's WIRING
    separately from the model's behaviour.
    """

    def __init__(self, prefix: str = "AI"):
        self.prefix = prefix
        self.call_count = 0          # lets tests assert whether it was called

    def invoke(self, value):
        self.call_count += 1
        return f"{self.prefix}: {value}"


class FailingModel(Runnable):
    """Always raises. Used to test fallback behaviour."""

    def __init__(self, message: str = "model unavailable"):
        self.message = message
        self.call_count = 0

    def invoke(self, value):
        self.call_count += 1
        raise RuntimeError(self.message)


# ======================================================================
# TASK 6 - RunnableWithFallbacks
# Module 6, section 6.8
# ======================================================================

class RunnableWithFallbacks(Runnable):
    """Tries the primary step, then each fallback in turn."""

    def __init__(self, primary, fallbacks):
        self.primary = primary
        self.fallbacks = list(fallbacks)

    def invoke(self, value):
        """Try primary, then fallbacks in order.

        Returns:
            The first successful result.

        Raises:
            The LAST exception, if every option fails. Raising the last one
            (not the first) is deliberate: it reflects the final state of
            the system after all recovery attempts.

        Examples:
            >>> ok = RunnableLambda(lambda x: "worked")
            >>> RunnableWithFallbacks(FailingModel(), [ok]).invoke("x")
            'worked'
        """
        # TODO:
        #   1. Try self.primary.invoke(value) in a try/except Exception.
        #   2. On failure, try each fallback in order, same pattern.
        #   3. Keep track of the most recent exception.
        #   4. If nothing succeeded, raise the last exception.
        raise NotImplementedError


# ======================================================================
# TASK 7 - memory
# Module 6, section 6.9
# ======================================================================

class BufferMemory:
    """Stores every turn, verbatim, forever.

    Maximum fidelity, unbounded token cost. Fine for short conversations,
    a liability in production (Module 6, section 6.9).
    """

    def __init__(self):
        self.messages = []

    def save(self, human: str, ai: str) -> None:
        """Record one exchange as two messages.

        Examples:
            >>> m = BufferMemory()
            >>> m.save("hi", "hello")
            >>> m.messages
            [{'role': 'human', 'content': 'hi'}, {'role': 'ai', 'content': 'hello'}]
        """
        # TODO: append TWO dicts - the human message then the ai message,
        # each shaped {"role": ..., "content": ...}
        raise NotImplementedError

    def load(self) -> list:
        """Return the messages to inject into the next prompt.

        Returns:
            A list of message dicts. Buffer memory returns everything.
        """
        # TODO: return all messages. Return a COPY (list(...)) so callers
        # cannot mutate our internal state by accident.
        raise NotImplementedError


class WindowMemory(BufferMemory):
    """Keeps only the most recent k exchanges.

    Bounded cost, predictable behaviour. The sensible default.
    """

    def __init__(self, k: int = 2):
        super().__init__()
        self.k = k

    def load(self) -> list:
        """Return only the last k exchanges.

        One exchange is TWO messages (human + ai), so this returns at most
        2 * k messages.

        Examples:
            >>> m = WindowMemory(k=1)
            >>> m.save("a", "1"); m.save("b", "2")
            >>> m.load()
            [{'role': 'human', 'content': 'b'}, {'role': 'ai', 'content': '2'}]
        """
        # TODO: return the last 2 * self.k messages.
        # Hint: negative slicing. Careful - self.messages[-0:] returns
        # EVERYTHING, so handle k == 0 explicitly.
        raise NotImplementedError


# ======================================================================
# SELF-TEST - do not edit
# ======================================================================

def _run_self_test() -> int:
    checks = []

    def add1_fn(x):
        return x + 1

    def double_fn(x):
        return x * 2

    # --- TASK 3 first: everything else depends on RunnableLambda ---
    try:
        add1 = RunnableLambda(add1_fn)
        double = RunnableLambda(double_fn)
        checks.append(("3. RunnableLambda.invoke", add1.invoke(1), 2))
        checks.append(("3. RunnableLambda wraps a builtin",
                       RunnableLambda(str.upper).invoke("hi"), "HI"))
    except NotImplementedError:
        add1 = double = None
        checks.append(("3. RunnableLambda.invoke", "not implemented", 2))

    if add1 is not None:
        # --- TASK 1 + 2: composition ---
        def try_check(name, fn, expected):
            try:
                checks.append((name, fn(), expected))
            except NotImplementedError:
                checks.append((name, "not implemented", expected))
            except Exception as exc:
                checks.append((name, f"raised {type(exc).__name__}: {exc}", expected))

        try_check("1+2. pipe two steps", lambda: (add1 | double).invoke(1), 4)
        try_check("1+2. pipe order matters", lambda: (double | add1).invoke(1), 3)
        try_check("1+2. coerce a plain function",
                  lambda: (add1 | (lambda x: x * 10)).invoke(1), 20)
        try_check("1+2. three steps flatten to one sequence",
                  lambda: len((add1 | add1 | add1).steps), 3)
        try_check("1+2. three steps compute correctly",
                  lambda: (add1 | add1 | add1).invoke(0), 3)
        try_check("1. batch", lambda: (add1 | double).batch([1, 2, 3]), [4, 6, 8])
        try_check("2. RunnableSequence direct construction",
                  lambda: RunnableSequence([add1, add1, add1]).invoke(0), 3)

        # --- TASK 4: parallel ---
        try_check("4. RunnableParallel fans out",
                  lambda: RunnableParallel(a=add1, b=double).invoke(3),
                  {"a": 4, "b": 6})
        try_check("4. RunnableParallel accepts plain functions",
                  lambda: RunnableParallel(a=add1_fn, b=double_fn).invoke(3),
                  {"a": 4, "b": 6})
        try_check("4. dict coerces to parallel in a pipe",
                  lambda: (add1 | {"a": add1, "b": double}).invoke(1),
                  {"a": 3, "b": 4})
        try_check("4. RunnablePassthrough keeps the input",
                  lambda: RunnableParallel(same=RunnablePassthrough(),
                                           plus=add1).invoke(5),
                  {"same": 5, "plus": 6})

        # --- TASK 5: prompt + model ---
        try_check("5. PromptRunnable renders",
                  lambda: PromptRunnable("Explain {topic}.").invoke({"topic": "RAG"}),
                  "Explain RAG.")

        def missing_variable():
            try:
                PromptRunnable("Explain {topic}.").invoke({"wrong": "x"})
                return "did not raise"
            except KeyError:
                return "raised KeyError"
            except NotImplementedError:
                raise
        try_check("5. PromptRunnable raises on missing variable",
                  missing_variable, "raised KeyError")

        try_check("1. __ror__: dict on the LEFT of a pipe",
                  lambda: ({"a": (lambda x: x), "b": (lambda x: x.upper())}
                           | PromptRunnable("{a}-{b}")).invoke("hi"),
                  "hi-HI")
        # 1 -> (+1) -> 2 -> add1 -> 3 -> double -> 6
        try_check("1. __ror__ preserves order (left runs first)",
                  lambda: ((lambda x: x + 1) | (add1 | double)).invoke(1), 6)

        try_check("5. full chain: prompt | model | parser",
                  lambda: (PromptRunnable("Explain {topic}.")
                           | FakeModel()
                           | (lambda s: s.upper())).invoke({"topic": "RAG"}),
                  "AI: EXPLAIN RAG.")

        # --- TASK 6: fallbacks ---
        def fallback_used():
            return RunnableWithFallbacks(FailingModel(), [FakeModel("BACKUP")]).invoke("x")
        try_check("6. fallback runs when primary fails", fallback_used, "BACKUP: x")

        def primary_wins():
            backup = FakeModel("BACKUP")
            result = RunnableWithFallbacks(FakeModel("PRIMARY"), [backup]).invoke("x")
            # The backup must not be touched when the primary succeeds.
            return (result, backup.call_count)
        try_check("6. fallback NOT used when primary succeeds",
                  primary_wins, ("PRIMARY: x", 0))

        def all_fail():
            try:
                RunnableWithFallbacks(FailingModel("first"),
                                      [FailingModel("second")]).invoke("x")
                return "did not raise"
            except NotImplementedError:
                raise
            except RuntimeError as exc:
                return str(exc)
        try_check("6. raises the LAST error when all fail", all_fail, "second")

        try_check("6. .with_fallbacks() helper works",
                  lambda: FailingModel().with_fallbacks([FakeModel("B")]).invoke("x"),
                  "B: x")

    # --- TASK 7: memory ---
    def memory_check(name, fn, expected):
        try:
            checks.append((name, fn(), expected))
        except NotImplementedError:
            checks.append((name, "not implemented", expected))
        except Exception as exc:
            checks.append((name, f"raised {type(exc).__name__}: {exc}", expected))

    def buffer_shape():
        memory = BufferMemory()
        memory.save("hi", "hello")
        return memory.messages
    memory_check("7. BufferMemory.save shape", buffer_shape,
                 [{"role": "human", "content": "hi"},
                  {"role": "ai", "content": "hello"}])

    def buffer_grows():
        memory = BufferMemory()
        for i in range(3):
            memory.save(f"q{i}", f"a{i}")
        return len(memory.load())
    memory_check("7. BufferMemory keeps everything (3 turns -> 6 msgs)",
                 buffer_grows, 6)

    def buffer_returns_copy():
        memory = BufferMemory()
        memory.save("hi", "hello")
        loaded = memory.load()
        loaded.append({"role": "human", "content": "injected"})
        return len(memory.messages)      # must still be 2
    memory_check("7. BufferMemory.load returns a copy", buffer_returns_copy, 2)

    def window_bounds():
        memory = WindowMemory(k=2)
        for i in range(5):
            memory.save(f"q{i}", f"a{i}")
        return len(memory.load())
    memory_check("7. WindowMemory(k=2) after 5 turns -> 4 msgs",
                 window_bounds, 4)

    def window_keeps_recent():
        memory = WindowMemory(k=1)
        memory.save("a", "1")
        memory.save("b", "2")
        return memory.load()
    memory_check("7. WindowMemory keeps the MOST RECENT turn",
                 window_keeps_recent,
                 [{"role": "human", "content": "b"},
                  {"role": "ai", "content": "2"}])

    def window_k_zero():
        memory = WindowMemory(k=0)
        memory.save("a", "1")
        return memory.load()
    memory_check("7. WindowMemory(k=0) returns nothing", window_k_zero, [])

    def window_short_history():
        memory = WindowMemory(k=5)
        memory.save("a", "1")
        return len(memory.load())
    memory_check("7. WindowMemory handles history shorter than k",
                 window_short_history, 2)

    # --- report ---
    print()
    print("=" * 74)
    print("  LAB 6 SELF-TEST - your own chain framework")
    print("=" * 74)
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
    print("-" * 74)
    if failures == 0:
        print(f"  All {len(checks)} checks passed.")
        print("  You just implemented LCEL. The | operator is Python's __or__ and")
        print("  nothing more - which is the whole point of building it yourself.")
    else:
        print(f"  {failures} of {len(checks)} failing.")
        print("  Order to work in: task 3 (RunnableLambda), then 1 and 2")
        print("  (coerce + sequence), then 4-7 in any order.")
    print("-" * 74)
    print()
    return failures


# ======================================================================
# DEMO - a realistic pipeline, no API needed
# ======================================================================

def demo_pipeline():
    print("=" * 74)
    print("  DEMO: a multi-step pipeline with your own framework")
    print("=" * 74)
    print()

    # A pipeline that classifies, then drafts, then formats -
    # using FakeModel so it is deterministic and free.
    classify = PromptRunnable("Classify this ticket: {ticket}") | FakeModel("CLASSIFIER")
    summarise = PromptRunnable("Summarise this ticket: {ticket}") | FakeModel("SUMMARISER")

    # Fan out to both, keeping the original input too.
    analyse = RunnableParallel(
        category=classify,
        summary=summarise,
        original=RunnablePassthrough(),
    )

    result = analyse.invoke({"ticket": "I was charged twice."})
    for key, value in result.items():
        print(f"  {key:>9}: {value}")
    print()

    # Now chain the parallel result into a formatting step.
    def format_report(data: dict) -> str:
        return f"[{data['category']}] {data['summary']}"

    pipeline = analyse | format_report
    print(f"  chained -> {pipeline.invoke({'ticket': 'App crashes on upload.'})}")
    print()
    print("  Note the shape: a parallel step produces a dict, and the next step")
    print("  consumes that dict. Type mismatches between steps are the most")
    print("  common chain bug (Module 6, section 6.8).")
    print()


def demo_memory():
    print("=" * 74)
    print("  DEMO: buffer vs window memory over 6 turns")
    print("=" * 74)
    print()

    buffer = BufferMemory()
    window = WindowMemory(k=2)

    print(f"  {'turn':>5}{'buffer msgs':>14}{'window msgs':>14}   buffer token cost")
    print("  " + "-" * 62)

    running_cost = 0
    for turn in range(1, 7):
        buffer.save(f"question {turn}", f"answer {turn}")
        window.save(f"question {turn}", f"answer {turn}")

        # Every turn re-sends the whole history, so cost accumulates.
        chars_this_turn = sum(len(m["content"]) for m in buffer.load())
        running_cost += chars_this_turn

        print(f"  {turn:>5}{len(buffer.load()):>14}{len(window.load()):>14}"
              f"{running_cost:>20} chars sent so far")

    print()
    print("  Buffer memory grows without bound. Because each turn re-sends the")
    print("  ENTIRE history, cost per turn grows linearly and TOTAL cost grows")
    print("  quadratically. Window memory stays flat at 2*k messages.")
    print()
    print(f"  Window memory currently holds:")
    for message in window.load():
        print(f"    {message['role']:>6}: {message['content']}")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        demo_pipeline()
        demo_memory()
    else:
        print("  Fix the self-test first, then the demos will run.")
        print()
