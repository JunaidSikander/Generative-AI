"""
solution.py - Lab 9 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import ast
import inspect
import operator


# ======================================================================
# TASK 1 - make_tool_schema
# ======================================================================

TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def make_tool_schema(func) -> dict:
    """Build a tool schema by inspecting a Python function."""
    signature = inspect.signature(func)

    # getdoc() dedents the docstring, which raw __doc__ does not.
    doc = inspect.getdoc(func) or ""

    # The FIRST paragraph is the description. Everything after a blank line is
    # Args/Returns detail the model does not need for routing, and which would
    # waste context on every request.
    description = doc.split("\n\n")[0].strip().replace("\n", " ")

    properties = {}
    required = []

    for name, parameter in signature.parameters.items():
        # Unannotated parameters fall back to "string" - the model will send
        # something, and a string is the least surprising thing to receive.
        properties[name] = {"type": TYPE_MAP.get(parameter.annotation, "string")}

        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = parameter.default

    return {
        "name": func.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ======================================================================
# TASK 3 - validate_arguments (defined before the registry, which uses it)
# ======================================================================

def _type_matches(value, expected: str) -> bool:
    """Does a value match a JSON Schema type name?"""
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        # bool subclasses int in Python, so True would otherwise pass as 1.
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def validate_arguments(schema: dict, arguments: dict) -> tuple:
    """Check model-generated arguments against a tool's schema."""
    # Fail fast on the wrong shape: every check below assumes a mapping.
    if not isinstance(arguments, dict):
        return (False, [f"expected an object, got {type(arguments).__name__}"])

    problems = []
    parameters = schema.get("parameters", {})
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])

    for name in required:
        if name not in arguments:
            problems.append(f"missing required argument: {name}")

    for name, value in arguments.items():
        if name not in properties:
            # An unexpected argument is a signal, not a nuisance: the model
            # invented a parameter, which may mean it picked the wrong tool.
            problems.append(f"unexpected argument: {name}")
            continue

        expected = properties[name].get("type", "string")
        if not _type_matches(value, expected):
            problems.append(
                f"argument {name} must be {expected}, "
                f"got {type(value).__name__}")

    return (not problems, problems)


# ======================================================================
# TASK 2 - ToolRegistry
# ======================================================================

class ToolRegistry:
    """Holds the tools an agent may call, and dispatches to them by name."""

    def __init__(self):
        self.tools = {}
        self.schemas = {}

    def register(self, func):
        """Add a function as a tool. Returns it unchanged, so it can decorate."""
        schema = make_tool_schema(func)
        self.tools[schema["name"]] = func
        self.schemas[schema["name"]] = schema
        return func

    def list_schemas(self) -> list:
        """Return every registered schema, for sending to the model."""
        return list(self.schemas.values())

    def call(self, name: str, arguments: dict):
        """Look up and invoke a tool.

        Raises:
            KeyError:   unknown tool.
            ValueError: arguments failed validation.
        """
        # ORDER MATTERS. Exists -> valid -> execute. Validating after executing
        # is not validating, and this is the single control point where a
        # malformed or malicious tool call can be stopped (section 9.12).
        if name not in self.tools:
            raise KeyError(f"unknown tool: {name}")

        is_valid, problems = validate_arguments(self.schemas[name], arguments)
        if not is_valid:
            raise ValueError(f"invalid arguments: {'; '.join(problems)}")

        return self.tools[name](**arguments)


# ======================================================================
# TASK 4 - safe_calculate
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
    """Evaluate an arithmetic expression WITHOUT executing arbitrary code."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")

    try:
        # mode="eval" parses a single expression, so statements (assignments,
        # imports, function definitions) are rejected by the parser itself.
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        # Normalise to ValueError so callers handle one exception type.
        raise ValueError(f"could not parse expression: {exc}") from exc

    return _evaluate_node(tree.body)


def _evaluate_node(node):
    """Recursively evaluate an allowlisted AST node."""
    if isinstance(node, ast.Constant):
        # bool first: it subclasses int, so True would otherwise be accepted.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(
                f"only numbers are allowed, got {type(node.value).__name__}")
        return node.value

    if isinstance(node, ast.BinOp):
        operation = type(node.op)
        if operation not in ALLOWED_BINARY:
            raise ValueError(f"operator not allowed: {operation.__name__}")

        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        # 2 ** 100000000 is a denial-of-service, not a calculation: Python
        # will happily try to build the integer and exhaust memory.
        if operation is ast.Pow and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent too large: {right}")

        if operation in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("division by zero")

        return ALLOWED_BINARY[operation](left, right)

    if isinstance(node, ast.UnaryOp):
        operation = type(node.op)
        if operation not in ALLOWED_UNARY:
            raise ValueError(f"unary operator not allowed: {operation.__name__}")
        return ALLOWED_UNARY[operation](_evaluate_node(node.operand))

    # THE ALLOWLIST BOUNDARY. Calls, names, attributes, subscripts, comparisons,
    # comprehensions, f-strings - everything not handled above is refused.
    # A blocklist would have to anticipate each of those; this does not.
    raise ValueError(f"expression element not allowed: {type(node).__name__}")


# ======================================================================
# TASK 5 - run_agent_loop
# ======================================================================

class ScriptedModel:
    """A deterministic stand-in for a real model."""

    def __init__(self, script: list):
        self.script = list(script)
        self.call_count = 0
        self.seen_messages = []

    def decide(self, messages: list) -> dict:
        self.call_count += 1
        self.seen_messages.append(list(messages))
        if not self.script:
            return {"type": "tool_call", "name": "noop", "arguments": {}}
        return self.script.pop(0)


def run_agent_loop(model, registry: ToolRegistry, question: str,
                   max_iterations: int = 5) -> dict:
    """Run the tool-calling loop until the model answers or we stop it."""
    messages = [{"role": "user", "content": question}]
    trace = []

    for iteration in range(1, max_iterations + 1):
        decision = model.decide(messages)

        if decision.get("type") == "final":
            return {
                "answer": decision.get("content", ""),
                "trace": trace,
                "iterations": iteration,
                "stopped_early": False,
            }

        name = decision.get("name", "")
        arguments = decision.get("arguments", {})

        try:
            observation = str(registry.call(name, arguments))
        except Exception as exc:  # noqa: BLE001 - every failure becomes an observation
            # Return the MESSAGE, not the traceback. A traceback in the
            # transcript leaks file paths, table names and library versions
            # into the model's context (section 9.12).
            observation = f"Error: {type(exc).__name__}: {exc}"

        trace.append({"tool": name, "arguments": arguments,
                      "observation": observation})

        # Grow the transcript so the next iteration can see what was tried.
        # This is the scratchpad, and it is why context grows every step.
        messages.append({"role": "assistant",
                         "content": f"Calling {name} with {arguments}"})
        messages.append({"role": "tool", "name": name, "content": observation})

    # Fell out of the loop without a final answer. THE GUARDRAIL.
    return {
        "answer": "Stopped: reached the maximum number of iterations.",
        "trace": trace,
        "iterations": max_iterations,
        "stopped_early": True,
    }


# ======================================================================
# Example tools
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
# BONUS - guardrails from section 9.12 that the tasks do not cover
# ======================================================================

IRREVERSIBLE_TOOLS = {"send_email", "delete_record", "make_payment", "deploy"}


class GuardedRegistry(ToolRegistry):
    """A registry that gates irreversible actions and enforces a budget.

    Demonstrates two guardrails beyond validation and the iteration cap:
    human approval for actions you cannot undo, and a total call budget.
    """

    def __init__(self, max_calls: int = 20, auto_approve: bool = False):
        super().__init__()
        self.max_calls = max_calls
        self.auto_approve = auto_approve
        self.call_count = 0
        self.audit_log = []

    def call(self, name: str, arguments: dict):
        # A total budget bounds cost even if a single loop stays short - an
        # agent invoked in a retry wrapper can otherwise still run away.
        if self.call_count >= self.max_calls:
            raise RuntimeError(f"tool call budget exhausted ({self.max_calls})")

        # Draw the line at REVERSIBILITY. Reads are safe to automate; sending
        # an email or taking a payment should have a human in the loop.
        if name in IRREVERSIBLE_TOOLS and not self.auto_approve:
            self.audit_log.append({"tool": name, "arguments": arguments,
                                   "outcome": "blocked_pending_approval"})
            raise PermissionError(
                f"{name} is irreversible and requires human approval")

        self.call_count += 1
        try:
            result = super().call(name, arguments)
        except Exception as exc:
            self.audit_log.append({"tool": name, "arguments": arguments,
                                   "outcome": f"error: {exc}"})
            raise
        # Log every call. When an agent does something surprising, the trace
        # is the only way to reconstruct why.
        self.audit_log.append({"tool": name, "arguments": arguments,
                               "outcome": "ok"})
        return result


# ======================================================================
# Demonstrations
# ======================================================================

def demo_allowlist_vs_blocklist():
    print("=" * 76)
    print("  ALLOWLIST vs BLOCKLIST")
    print("=" * 76)
    print()

    # A plausible-looking blocklist, of the kind people actually write.
    BANNED = ["__import__", "eval", "exec", "open", "os.", "subprocess", "system"]

    def blocklist_calculate(expression: str):
        for pattern in BANNED:
            if pattern in expression:
                raise ValueError(f"blocked pattern: {pattern}")
        return eval(expression)          # NEVER do this

    bypasses = [
        "().__class__.__bases__[0].__subclasses__()",
        "getattr(__builtins__, 'ev' + 'al')",
        # Deliberately modest. 2 ** 100000000 also slips this blocklist, but it
        # allocates a 12 MB integer and takes seconds - a real denial-of-service,
        # and not something a teaching demo should inflict on your machine.
        # The point is only that a pattern blocklist does not stop resource
        # exhaustion at all; the exponent size does not change that.
        "2 ** 1000000",
    ]

    print("  A blocklist that bans __import__, eval, exec, open, os., subprocess:")
    for expression in bypasses:
        try:
            blocklist_calculate(expression)
            print(f"    [SLIPPED THROUGH] {expression[:52]}")
        except ValueError as exc:
            print(f"    [blocked        ] {expression[:52]}  ({exc})")
        except Exception as exc:
            print(f"    [ran, then errored] {expression[:44]} ({type(exc).__name__})")
    print()

    print("  The same expressions against the allowlist:")
    for expression in bypasses:
        try:
            safe_calculate(expression)
            print(f"    [SLIPPED THROUGH] {expression[:52]}")
        except ValueError as exc:
            print(f"    [blocked        ] {expression[:52]}")
    print()
    print("  The first bypass reaches arbitrary classes without using any banned")
    print("  word. A blocklist must anticipate every route; the allowlist only")
    print("  has to know what arithmetic looks like.")
    print()


def demo_guardrails():
    print("=" * 76)
    print("  GUARDRAILS: approval gates and budgets")
    print("=" * 76)
    print()

    registry = GuardedRegistry(max_calls=3)

    @registry.register
    def get_revenue(year: int) -> int:
        """Look up revenue for a year."""
        return {2024: 4_200_000}.get(year, 0)

    @registry.register
    def send_email(to: str, body: str) -> str:
        """Send an email to a recipient.

        Args:
            to: Recipient address.
            body: Message body.
        """
        return f"sent to {to}"

    print("  A read-only tool call:")
    print(f"    get_revenue(2024) -> {registry.call('get_revenue', {'year': 2024})}")
    print()

    print("  An irreversible tool call:")
    try:
        registry.call("send_email", {"to": "customers@example.com", "body": "hi"})
    except PermissionError as exc:
        print(f"    BLOCKED: {exc}")
    print()

    print("  Exhausting the call budget:")
    for attempt in range(4):
        try:
            registry.call("get_revenue", {"year": 2024})
            print(f"    call {attempt + 1}: ok")
        except RuntimeError as exc:
            print(f"    call {attempt + 1}: BLOCKED - {exc}")
    print()

    print("  Audit log:")
    for entry in registry.audit_log:
        print(f"    {entry['tool']:<14} {entry['outcome']}")
    print()
    print("  Note the email attempt is logged as blocked_pending_approval. If")
    print("  a prompt-injected document ever persuades an agent to email your")
    print("  customer list, this is the line that stops it - and the log entry")
    print("  that tells you it was attempted.")
    print()


def demo_context_growth():
    print("=" * 76)
    print("  WHY AGENT LOOPS GET EXPENSIVE")
    print("=" * 76)
    print()

    registry = build_demo_registry()
    script = [{"type": "tool_call", "name": "get_revenue",
               "arguments": {"year": 2024}} for _ in range(6)]

    model = ScriptedModel(script)
    run_agent_loop(model, registry, "How much did we make?", max_iterations=6)

    print(f"  {'iteration':>10}{'messages sent':>16}{'chars sent':>13}"
          f"{'cumulative':>13}")
    print("  " + "-" * 52)

    cumulative = 0
    for iteration, messages in enumerate(model.seen_messages, 1):
        chars = sum(len(str(m.get("content", ""))) for m in messages)
        cumulative += chars
        print(f"  {iteration:>10}{len(messages):>16}{chars:>13}{cumulative:>13}")

    print()
    print("  Every iteration re-sends the whole scratchpad, so cost per step")
    print("  grows linearly and TOTAL cost grows quadratically with loop length -")
    print("  exactly the pattern from Module 6's buffer memory.")
    print()
    print("  Which is a second, purely economic reason to cap iterations.")
    print()


if __name__ == "__main__":
    demo_allowlist_vs_blocklist()
    demo_guardrails()
    demo_context_growth()
