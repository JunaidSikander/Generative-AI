"""
solution.py - Lab 6 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""


# ======================================================================
# TASK 1 - Runnable and the | operator
# ======================================================================

class Runnable:
    """Base class for anything that can be a step in a chain."""

    def invoke(self, value):
        raise NotImplementedError(f"{type(self).__name__} must implement invoke()")

    def __or__(self, other):
        """Implement `a | b`.

        Python calls this for the | expression. That single dunder method is
        the entire mechanism behind LCEL - there is no other machinery.
        """
        return RunnableSequence([self, coerce(other)])

    def __ror__(self, other):
        """Implement `other | self` when `other` is NOT a Runnable.

        Python tries the LEFT operand's __or__ first. For `{"a": x} | prompt`
        that means dict.__or__(prompt), which fails - so Python falls back to
        the RIGHT operand's __ror__. Without this method, the RAG shape
        `{"context": retriever, ...} | prompt` raises TypeError.

        Real LangChain defines __ror__ for exactly this reason.
        """
        return RunnableSequence([coerce(other), self])

    def batch(self, values):
        """Run on many inputs.

        Real LangChain runs these concurrently; we keep it sequential so the
        code stays readable. The INTERFACE is the point - callers write
        .batch() once and get whatever concurrency the framework provides.
        """
        return [self.invoke(value) for value in values]

    def with_fallbacks(self, fallbacks):
        return RunnableWithFallbacks(self, fallbacks)


def coerce(thing):
    """Turn a plain function or dict into a Runnable."""
    # Check Runnable first - it is the common case and the cheapest test.
    if isinstance(thing, Runnable):
        return thing

    # A dict of steps becomes a parallel fan-out. This is why real LCEL lets
    # you write {"context": retriever, "question": passthrough} | prompt.
    if isinstance(thing, dict):
        return RunnableParallel(**thing)

    # Any callable becomes a lambda step. This is the escape hatch that means
    # you are never stuck inside the abstraction.
    if callable(thing):
        return RunnableLambda(thing)

    raise TypeError(
        f"cannot use {type(thing).__name__} as a chain step; "
        "expected a Runnable, dict, or callable"
    )


# ======================================================================
# TASK 2 - RunnableSequence
# ======================================================================

class RunnableSequence(Runnable):
    """Runs steps one after another, threading output into input."""

    def __init__(self, steps):
        self.steps = list(steps)

    def invoke(self, value):
        # The whole of "chaining": reassign `value` at each step.
        for step in self.steps:
            value = step.invoke(value)
        return value

    def __or__(self, other):
        # Flatten rather than nest. Without this, a | b | c would build
        # Sequence([Sequence([a, b]), c]) - correct, but harder to inspect
        # and it makes tracing output confusingly deep.
        return RunnableSequence(self.steps + [coerce(other)])

    def __ror__(self, other):
        # Prepend rather than nest, for the same flattening reason as above.
        return RunnableSequence([coerce(other)] + self.steps)

    def __repr__(self):
        return " | ".join(type(step).__name__ for step in self.steps)


# ======================================================================
# TASK 3 - RunnableLambda
# ======================================================================

class RunnableLambda(Runnable):
    """Wraps any plain Python function as a chain step."""

    def __init__(self, function, name: str = None):
        self.function = function
        self.name = name or getattr(function, "__name__", "lambda")

    def invoke(self, value):
        return self.function(value)

    def __repr__(self):
        return f"RunnableLambda({self.name})"


# ======================================================================
# TASK 4 - RunnableParallel and RunnablePassthrough
# ======================================================================

class RunnableParallel(Runnable):
    """Runs several steps on the SAME input and collects results by key."""

    def __init__(self, **steps):
        self.steps = {key: coerce(value) for key, value in steps.items()}

    def invoke(self, value):
        # Every step receives the SAME original input. That is the difference
        # between a fan-out and a sequence, and it is why the steps are
        # independent enough to run concurrently in a real implementation.
        return {key: step.invoke(value) for key, step in self.steps.items()}

    def __repr__(self):
        return f"RunnableParallel({', '.join(self.steps)})"


class RunnablePassthrough(Runnable):
    """Returns its input unchanged."""

    def invoke(self, value):
        return value


# ======================================================================
# TASK 5 - PromptRunnable and the test doubles
# ======================================================================

class PromptRunnable(Runnable):
    """Fills a template string from a dict of variables."""

    def __init__(self, template: str):
        self.template = template

    def invoke(self, value):
        # str.format raises KeyError on a missing variable, and we let it.
        # A prompt shipped with a literal "{topic}" in it does not crash the
        # model - it produces confident nonsense, which is far harder to spot.
        return self.template.format(**value)


class FakeModel(Runnable):
    """A deterministic stand-in for a real model."""

    def __init__(self, prefix: str = "AI"):
        self.prefix = prefix
        self.call_count = 0

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
# ======================================================================

class RunnableWithFallbacks(Runnable):
    """Tries the primary step, then each fallback in turn."""

    def __init__(self, primary, fallbacks):
        self.primary = primary
        self.fallbacks = list(fallbacks)

    def invoke(self, value):
        last_error = None

        # Try everything in order, primary first.
        for candidate in [self.primary, *self.fallbacks]:
            try:
                return candidate.invoke(value)
            except Exception as exc:  # noqa: BLE001 - any failure means "try the next one"
                last_error = exc

        # Everything failed. Raise the LAST error, not the first: it describes
        # the final state of the system after all recovery attempts, which is
        # what someone reading the traceback needs to know.
        raise last_error


# ======================================================================
# TASK 7 - memory
# ======================================================================

class BufferMemory:
    """Stores every turn, verbatim, forever."""

    def __init__(self):
        self.messages = []

    def save(self, human: str, ai: str) -> None:
        self.messages.append({"role": "human", "content": human})
        self.messages.append({"role": "ai", "content": ai})

    def load(self) -> list:
        # Return a copy so a caller cannot mutate our history by accident -
        # an easy bug when the returned list gets spliced into a prompt.
        return list(self.messages)


class WindowMemory(BufferMemory):
    """Keeps only the most recent k exchanges."""

    def __init__(self, k: int = 2):
        super().__init__()
        self.k = k

    def load(self) -> list:
        # Guard k == 0 explicitly: self.messages[-0:] is self.messages[0:],
        # which returns EVERYTHING. A genuinely nasty off-by-one.
        if self.k <= 0:
            return []

        # One exchange is two messages, so keep the last 2*k.
        # Slicing past the start is safe in Python - a short history just
        # returns everything it has.
        return list(self.messages[-2 * self.k:])


class SummaryMemory(BufferMemory):
    """Keeps recent turns verbatim and compresses older ones into a summary.

    The usual production compromise: bounded cost, but older context is not
    lost entirely. A real implementation would call an LLM to write the
    summary; this one concatenates topics so the demo stays free and
    deterministic.
    """

    def __init__(self, keep_recent: int = 2):
        super().__init__()
        self.keep_recent = keep_recent
        self.summary = ""

    def save(self, human: str, ai: str) -> None:
        super().save(human, ai)

        # Once we exceed the window, fold the oldest exchange into the summary.
        max_messages = 2 * self.keep_recent
        while len(self.messages) > max_messages:
            old_human = self.messages.pop(0)
            self.messages.pop(0)          # its paired ai message
            fragment = old_human["content"]
            self.summary = f"{self.summary}; {fragment}" if self.summary else fragment

    def load(self) -> list:
        messages = []
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Earlier in this conversation: {self.summary}",
            })
        messages.extend(self.messages)
        return messages


# ======================================================================
# Demonstrations
# ======================================================================

def demo_pipe_is_just_dunder_or():
    print("=" * 74)
    print("  THE | OPERATOR IS JUST __or__")
    print("=" * 74)
    print()

    add1 = RunnableLambda(lambda x: x + 1, name="add1")
    double = RunnableLambda(lambda x: x * 2, name="double")

    piped = add1 | double
    explicit = RunnableSequence([add1, double])
    dunder = add1.__or__(double)

    print(f"  add1 | double                    -> {piped.invoke(1)}")
    print(f"  RunnableSequence([add1, double])  -> {explicit.invoke(1)}")
    print(f"  add1.__or__(double)              -> {dunder.invoke(1)}")
    print()
    print("  All three are the same thing. `|` is syntax; __or__ is the method.")
    print()
    print(f"  repr of a 3-step chain: {add1 | double | add1}")
    print(f"  steps are FLATTENED, not nested: {len((add1 | double | add1).steps)} steps")
    print()


def demo_parallel_vs_sequential():
    print("=" * 74)
    print("  PARALLEL vs SEQUENTIAL - the same steps, different wiring")
    print("=" * 74)
    print()

    add1 = RunnableLambda(lambda x: x + 1)
    double = RunnableLambda(lambda x: x * 2)

    print("  SEQUENTIAL (add1 | double), input 3:")
    print(f"    3 -> add1 -> 4 -> double -> {(add1 | double).invoke(3)}")
    print("    each step consumes the PREVIOUS step's output")
    print()

    print("  PARALLEL (RunnableParallel(a=add1, b=double)), input 3:")
    print(f"    3 -> both -> {RunnableParallel(a=add1, b=double).invoke(3)}")
    print("    each step consumes the SAME original input")
    print()
    print("  Sequential builds on itself; parallel fans out. Getting these")
    print("  confused is a common source of surprising chain results.")
    print()


def demo_fallbacks():
    print("=" * 74)
    print("  FALLBACKS")
    print("=" * 74)
    print()

    primary_ok = FakeModel("PRIMARY")
    backup = FakeModel("BACKUP")
    chain = primary_ok.with_fallbacks([backup])
    print(f"  primary succeeds -> {chain.invoke('hello')}")
    print(f"    backup call_count = {backup.call_count}  (never touched)")
    print()

    broken = FailingModel("rate limit exceeded")
    backup2 = FakeModel("BACKUP")
    chain2 = broken.with_fallbacks([backup2])
    print(f"  primary fails    -> {chain2.invoke('hello')}")
    print(f"    primary call_count = {broken.call_count}, "
          f"backup call_count = {backup2.call_count}")
    print()

    try:
        FailingModel("first").with_fallbacks([FailingModel("second")]).invoke("x")
    except RuntimeError as exc:
        print(f"  everything fails -> RuntimeError({exc!r})")
        print("    note it is the LAST error, not the first")
    print()


def demo_memory_strategies():
    print("=" * 74)
    print("  THREE MEMORY STRATEGIES OVER 6 TURNS")
    print("=" * 74)
    print()

    strategies = {
        "buffer": BufferMemory(),
        "window(k=2)": WindowMemory(k=2),
        "summary(keep=2)": SummaryMemory(keep_recent=2),
    }

    for turn in range(1, 7):
        for memory in strategies.values():
            memory.save(f"question {turn}", f"answer {turn}")

    print(f"  {'strategy':<18}{'messages':>10}{'chars':>8}")
    print("  " + "-" * 36)
    for name, memory in strategies.items():
        loaded = memory.load()
        chars = sum(len(m["content"]) for m in loaded)
        print(f"  {name:<18}{len(loaded):>10}{chars:>8}")
    print()

    print("  What summary memory actually holds:")
    for message in strategies["summary(keep=2)"].load():
        print(f"    {message['role']:>6}: {message['content']}")
    print()
    print("  Buffer keeps everything and grows without bound. Window is flat")
    print("  but forgets. Summary is flat AND retains a trace of what came")
    print("  before - which is why it is the usual production choice.")
    print()


def demo_rag_shape():
    print("=" * 74)
    print("  THE RAG SHAPE (a preview of Module 8)")
    print("=" * 74)
    print()

    # A stand-in retriever: in Module 8 this becomes a real vector search.
    def fake_retriever(question: str) -> str:
        return f"[retrieved 3 documents about '{question}']"

    rag = (
        {"context": fake_retriever, "question": RunnablePassthrough()}
        | PromptRunnable("Context: {context}\n\nQuestion: {question}\n\nAnswer:")
        | FakeModel("ANSWER")
    )

    print("  rag = ({'context': retriever, 'question': passthrough}")
    print("         | prompt | model)")
    print()
    print(f"  {rag.invoke('What is chunking?')}")
    print()
    print("  The dict is the key trick: the question goes to BOTH the retriever")
    print("  and the prompt. That exact shape is the backbone of Module 8.")
    print()


if __name__ == "__main__":
    demo_pipe_is_just_dunder_or()
    demo_parallel_vs_sequential()
    demo_fallbacks()
    demo_memory_strategies()
    demo_rag_shape()
