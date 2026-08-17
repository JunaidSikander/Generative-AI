"""
check_setup.py - Environment diagnostic for GenAI From Scratch.

Run this any time something breaks. It checks your Python version, whether a
virtual environment is active, which packages are installed, and whether your
API key is loading - then tells you exactly what to fix.

    python check_setup.py

Deliberately written to have NO required dependencies and to never crash:
a diagnostic that fails to run is worse than no diagnostic at all. It also uses
plain ASCII markers rather than emoji, because Windows terminals using the
legacy cp1252 code page raise UnicodeEncodeError on emoji.
"""

import importlib
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Small helpers for consistent, readable output
# ----------------------------------------------------------------------

# Tally of results so we can print a summary at the end.
results = {"ok": 0, "warn": 0, "fail": 0}


def header(title):
    """Print a section heading."""
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def report(status, label, detail="", fix=""):
    """Print one check result and record it.

    Args:
        status: "ok", "warn" or "fail".
        label:  What was checked.
        detail: What we found.
        fix:    What to do about it, if anything.
    """
    marks = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
    results[status] += 1

    line = f"{marks[status]}  {label}"
    if detail:
        line += f": {detail}"
    print(line)

    if fix:
        # Indent the suggested fix so it reads as a sub-note.
        print(f"         -> {fix}")


# ----------------------------------------------------------------------
# Check 1 - Python version
# ----------------------------------------------------------------------

def check_python_version():
    """This course needs Python 3.10 or newer."""
    header("1. Python version")

    # sys.version_info is a tuple like (3, 12, 1, 'final', 0)
    major, minor = sys.version_info[0], sys.version_info[1]
    version = f"{major}.{minor}.{sys.version_info[2]}"

    if (major, minor) >= (3, 10):
        report("ok", "Python version", version)
    elif major == 3:
        report(
            "fail",
            "Python version",
            f"{version} is too old",
            "Install Python 3.10+ from python.org (Module 2, section 2.2)",
        )
    else:
        report(
            "fail",
            "Python version",
            f"{version} is Python 2, a different language",
            "Install Python 3.10+ from python.org",
        )

    # Knowing WHICH python is running resolves most confusing import errors.
    print(f"         interpreter: {sys.executable}")


# ----------------------------------------------------------------------
# Check 2 - Virtual environment
# ----------------------------------------------------------------------

def check_virtual_environment():
    """Warn if packages would install system-wide instead of per-project."""
    header("2. Virtual environment")

    # The reliable modern test: inside a venv these two differ.
    in_venv = sys.prefix != sys.base_prefix

    if in_venv:
        report("ok", "Virtual environment active", Path(sys.prefix).name)
    else:
        activate = (
            r".\.venv\Scripts\Activate.ps1"
            if os.name == "nt"
            else "source .venv/bin/activate"
        )
        report(
            "warn",
            "No virtual environment detected",
            "packages would install system-wide",
            f"Create one:  python -m venv .venv    then activate:  {activate}",
        )


# ----------------------------------------------------------------------
# Check 3 - Packages
# ----------------------------------------------------------------------

# (import name, pip name, needed from which module, required for Module 2?)
PACKAGES = [
    ("dotenv",   "python-dotenv", "Module 2",  True),
    ("openai",   "openai",        "Module 2",  True),
    ("numpy",    "numpy",         "Module 3",  False),
    ("tiktoken", "tiktoken",      "Module 3",  False),
    ("pandas",   "pandas",        "Module 3",  False),
    ("langchain", "langchain",    "Module 6",  False),
    ("faiss",    "faiss-cpu",     "Module 7",  False),
    ("chromadb", "chromadb",      "Module 7",  False),
    ("gradio",   "gradio",        "Module 13", False),
]


def check_packages():
    """Report which course packages are importable."""
    header("3. Packages")

    missing_required = []
    missing_optional = []

    for import_name, pip_name, needed_by, required in PACKAGES:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "installed")
            report("ok", pip_name, version)
        except ImportError:
            if required:
                missing_required.append(pip_name)
                report("fail", pip_name, "not installed", f"needed now ({needed_by})")
            else:
                missing_optional.append(pip_name)
                report("warn", pip_name, "not installed", f"needed from {needed_by}")
        except Exception as exc:  # noqa: BLE001 - a broken install shouldn't stop the report
            report("warn", pip_name, f"installed but failed to import ({exc})",
                   f"Try: pip install --force-reinstall {pip_name}")

    if missing_required:
        print()
        print("  Install what you need right now:")
        print(f"    pip install {' '.join(missing_required)}")

    if missing_optional:
        print()
        print("  Optional for later modules (or install everything at once):")
        print("    pip install -r requirements.txt")


# ----------------------------------------------------------------------
# Check 4 - .env file and secret hygiene
# ----------------------------------------------------------------------

def find_repo_root():
    """Walk upward looking for the repo root (the folder holding .env.example)."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / ".env.example").exists():
            return candidate
    # Fall back to two levels up: labs/02-python-environment -> repo root
    return here.parent.parent.parent


def check_env_file():
    """Check .env exists, is git-ignored, and holds at least one key."""
    header("4. API keys and secret hygiene")

    root = find_repo_root()
    env_path = root / ".env"
    gitignore_path = root / ".gitignore"

    print(f"  Looking in: {root}")
    print()

    # --- Does .env exist? ---
    if not env_path.exists():
        copy_cmd = (
            "Copy-Item .env.example .env" if os.name == "nt" else "cp .env.example .env"
        )
        report(
            "warn",
            ".env file",
            "not found",
            f"Create it:  {copy_cmd}    then add your key (Module 2, section 2.10)",
        )
        return

    report("ok", ".env file", "found")

    # --- Is it git-ignored? This is the check that protects your money. ---
    if gitignore_path.exists():
        ignored = ".env" in gitignore_path.read_text(encoding="utf-8", errors="ignore")
        if ignored:
            report("ok", ".env is git-ignored", "your key will not be committed")
        else:
            report(
                "fail",
                ".env is NOT in .gitignore",
                "your key could be committed and leaked",
                "Add a line containing  .env  to .gitignore before committing anything",
            )
    else:
        report("warn", ".gitignore", "not found", "Add one containing  .env")

    # --- Load it and see which keys are present ---
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        report(
            "warn",
            "python-dotenv",
            "not installed, cannot load .env",
            "pip install python-dotenv",
        )
        return

    known_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
    ]

    found_any = False
    for key_name in known_keys:
        value = os.getenv(key_name)
        if value and value.strip():
            found_any = True
            # Show ONLY a short prefix and the length. Never the whole key.
            preview = value[:7]
            report("ok", key_name, f"loaded ({preview}..., {len(value)} chars)")
        else:
            report("warn", key_name, "not set")

    if not found_any:
        print()
        print("  No API keys set. That is fine for now - you can either:")
        print("    a) add a key to .env  (see Module 2, section 2.10), or")
        print("    b) use Ollama for free local models  (see Appendix A)")


# ----------------------------------------------------------------------
# Check 5 - Ollama (the free local route)
# ----------------------------------------------------------------------

def check_ollama():
    """Report whether a local Ollama server is reachable. Entirely optional."""
    header("5. Ollama (optional, free local models)")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        # urllib is in the standard library, so this needs nothing installed.
        import json
        import urllib.error
        import urllib.request

        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))

        models = [m.get("name", "?") for m in payload.get("models", [])]
        if models:
            report("ok", "Ollama running", f"{len(models)} model(s): {', '.join(models[:5])}")
        else:
            report(
                "warn",
                "Ollama running but no models",
                "",
                "Pull one:  ollama pull llama3",
            )
    except Exception:  # noqa: BLE001 - any failure just means "not available"
        report(
            "warn",
            "Ollama not reachable",
            f"nothing listening at {base_url}",
            "Only needed if you want the free local route - see Appendix A",
        )


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

def print_summary():
    header("Summary")

    print(f"  Passed:   {results['ok']}")
    print(f"  Warnings: {results['warn']}")
    print(f"  Failed:   {results['fail']}")
    print()

    if results["fail"] == 0:
        print("  No blocking problems. You are ready for the lab.")
        print("  Warnings are fine - they flag things needed by later modules.")
    else:
        print("  Fix the [FAIL] items above before continuing.")
        print("  Each one lists the command to run.")

    print()
    print("  Still stuck? See appendix/D-troubleshooting.md")
    print()

    # Non-zero exit code on failure, so this can be used in automation.
    return 1 if results["fail"] else 0


def main():
    print()
    print("GenAI From Scratch - environment check")

    check_python_version()
    check_virtual_environment()
    check_packages()
    check_env_file()
    check_ollama()

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
