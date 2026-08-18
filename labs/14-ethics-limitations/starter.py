"""
starter.py - Lab 14: Measure It, Then Decide

Replace each TODO with working code. The self-test checks your work.

    python starter.py

Pure standard library - no packages, no API key.

You will implement three fairness metrics and demonstrate for yourself that
they DISAGREE, compute the energy break-even between training and inference,
and build a structured impact assessment you can run on something you
actually want to build.
"""

from collections import defaultdict


# ======================================================================
# TASK 1 - selection_rates
# Module 14, section 14.2
# ======================================================================

def selection_rates(predictions: list, groups: list) -> dict:
    """Fraction of each group receiving a positive outcome.

    Args:
        predictions: Truthy for a positive outcome (selected, approved,
                     flagged), falsy otherwise.
        groups:      Group label for each prediction, same length.

    Returns:
        {group: rate} where rate is positives / total for that group.
        An empty input gives an empty dict.

    Raises:
        ValueError: if the two lists differ in length - a silent zip() would
                    quietly drop the tail and give you a wrong answer.

    Examples:
        >>> selection_rates([1, 1, 0, 0], ["A", "A", "B", "B"])
        {'A': 1.0, 'B': 0.0}
    """
    # TODO:
    #   1. Raise ValueError if len(predictions) != len(groups).
    #   2. Count totals and positives per group (defaultdict(int) helps).
    #   3. Return {group: positives/total}. Guard against a zero total.
    return {}


# ======================================================================
# TASK 2 - disparate_impact_ratio
# Module 14, section 14.2
# ======================================================================

FOUR_FIFTHS = 0.8


def disparate_impact_ratio(predictions: list, groups: list) -> float:
    """Ratio of the lowest selection rate to the highest.

        ratio = min(selection_rate) / max(selection_rate)

    A ratio below 0.8 is the "four-fifths rule", used as a screening
    threshold in US employment discrimination practice. It is a flag for
    investigation, not a verdict.

    Returns:
        1.0 (perfect parity) when there are no groups, only one group, or
        nobody was selected anywhere. Otherwise min/max.

    Examples:
        >>> round(disparate_impact_ratio([1]*50 + [0]*50 + [1]*30 + [0]*70,
        ...                              ["A"]*100 + ["B"]*100), 4)
        0.6
    """
    # TODO:
    #   1. rates = selection_rates(predictions, groups)
    #   2. If there are no rates, return 1.0 - nothing to compare.
    #   3. If max(rates) == 0, return 1.0. Nobody was selected anywhere,
    #      which is equally bad for everyone, and it avoids dividing by zero.
    #   4. Otherwise return min / max.
    return 1.0


def demographic_parity_difference(predictions: list, groups: list) -> float:
    """Largest gap between any two groups' selection rates.

    Returns:
        max(rate) - min(rate). 0.0 means every group is selected at the
        same rate. 0.0 for fewer than two groups.

    Examples:
        >>> round(demographic_parity_difference(
        ...     [1]*50 + [0]*50 + [1]*30 + [0]*70, ["A"]*100 + ["B"]*100), 4)
        0.2
    """
    # TODO: rates = selection_rates(...); return max - min, or 0.0 if empty.
    return 0.0


# ======================================================================
# TASK 3 - true_positive_rates and equal opportunity
# Module 14, section 14.2
# ======================================================================

def true_positive_rates(predictions: list, actuals: list, groups: list) -> dict:
    """Of those who SHOULD be positive, what fraction were?

        TPR(group) = correctly-positive in group / actually-positive in group

    This is the metric that needs ground truth: you must know who should
    have been selected, not just who was.

    Args:
        predictions: What the system decided.
        actuals:     What the correct decision was.
        groups:      Group label for each.

    Returns:
        {group: tpr}. A group with NO actually-positive members is OMITTED
        entirely rather than recorded as 0.0 - there is no rate to compute,
        and a 0.0 would look like total failure instead of no data.

    Raises:
        ValueError: if the three lists differ in length.

    Examples:
        >>> true_positive_rates([1, 0, 1, 1], [1, 1, 1, 1],
        ...                     ["A", "A", "B", "B"])
        {'A': 0.5, 'B': 1.0}
    """
    # TODO:
    #   1. Raise ValueError unless all three lengths match.
    #   2. For each (prediction, actual, group): only count rows where
    #      `actual` is truthy. Track how many there are, and how many of
    #      those were also predicted positive.
    #   3. Return {group: caught/should} ONLY for groups where should > 0.
    return {}


def equal_opportunity_difference(predictions: list, actuals: list,
                                 groups: list) -> float:
    """Largest gap between any two groups' true positive rates.

    Returns:
        max(TPR) - min(TPR). 0.0 for fewer than two measurable groups.

    Examples:
        >>> round(equal_opportunity_difference(
        ...     [1, 0, 1, 1], [1, 1, 1, 1], ["A", "A", "B", "B"]), 4)
        0.5
    """
    # TODO: rates = true_positive_rates(...); return max - min, or 0.0.
    return 0.0


# ======================================================================
# TASK 4 - fairness_report
# Module 14, section 14.2
# ======================================================================

def fairness_report(predictions: list, groups: list,
                    actuals: list = None) -> dict:
    """Compute several fairness metrics together, and flag disagreement.

    The whole point of this function is that the metrics can DISAGREE. A
    system can show perfect demographic parity while being far worse at
    identifying qualified people in one group.

    Args:
        predictions: What the system decided.
        groups:      Group label for each.
        actuals:     Ground truth. Optional - without it, only the
                     selection-rate metrics can be computed.

    Returns:
        {
          "selection_rates": dict,
          "disparate_impact_ratio": float,
          "passes_four_fifths": bool,
          "demographic_parity_difference": float,
          "true_positive_rates": dict,      empty without actuals
          "equal_opportunity_difference": float or None,
          "metrics_disagree": bool,
        }

    metrics_disagree is True when the selection-rate view looks acceptable
    (passes the four-fifths rule) but the opportunity view does not
    (equal_opportunity_difference > 0.1). It is False when actuals were
    not supplied - you cannot detect disagreement you cannot measure.
    """
    # TODO:
    #   1. Compute the selection-rate metrics (tasks 1 and 2).
    #   2. passes_four_fifths = disparate_impact_ratio >= FOUR_FIFTHS
    #   3. If actuals is None: true_positive_rates {}, equal opportunity None,
    #      metrics_disagree False.
    #   4. Otherwise compute the opportunity metrics, and set
    #      metrics_disagree = passes_four_fifths and eo_difference > 0.1
    #   5. Return the dict above.
    return {}


# ======================================================================
# TASK 5 - energy_break_even
# Module 14, section 14.4
# ======================================================================

def energy_break_even(training_mwh: float, inference_wh_per_query: float,
                      queries_per_day: int) -> dict:
    """When does cumulative inference energy overtake training energy?

    The figure people usually get backwards. Training is one-off and large;
    inference is tiny and multiplied by every request forever.

    Args:
        training_mwh:           One-off training energy, megawatt-hours.
        inference_wh_per_query: Per-query energy, watt-hours.
        queries_per_day:        Serving volume.

    Returns:
        {
          "training_kwh": float,
          "inference_kwh_per_query": float,
          "break_even_queries": int or None,
          "break_even_days": float or None,
          "inference_kwh_per_year": float,
        }

        break_even_* are None when inference_wh_per_query is 0 - it never
        catches up.

    Units: 1 MWh = 1000 kWh; 1 kWh = 1000 Wh.

    Examples:
        >>> r = energy_break_even(1000.0, 1.0, 1_000_000)
        >>> r["training_kwh"], r["break_even_queries"]
        (1000000.0, 1000000)
    """
    # TODO:
    #   1. training_kwh = training_mwh * 1000
    #   2. inference_kwh_per_query = inference_wh_per_query / 1000
    #   3. If inference_kwh_per_query <= 0: break-even fields are None.
    #   4. break_even_queries = ceil(training_kwh / inference_kwh_per_query)
    #   5. break_even_days = break_even_queries / queries_per_day
    #      (guard queries_per_day <= 0 -> None)
    #   6. inference_kwh_per_year = inference_kwh_per_query * queries_per_day * 365
    return {"training_kwh": 0.0, "inference_kwh_per_query": 0.0,
            "break_even_queries": None, "break_even_days": None,
            "inference_kwh_per_year": 0.0}


# ======================================================================
# TASK 6 - impact_assessment
# Module 14, section 14.9
# ======================================================================

AUTONOMY_LEVELS = {
    "personal_tool": 0,       # affects nobody but the operator
    "informational": 1,       # a user reads output and decides
    "recommendation": 2,      # someone acts on a suggestion
    "automated_decision": 3,  # the system decides, unsupervised
}

HIGHER_BAR_DOMAINS = {
    "health", "legal", "financial", "employment",
    "education", "criminal_justice", "children",
}


def impact_assessment(system: dict) -> dict:
    """Assess how much care a proposed system needs before building it.

    Args:
        system: Any of these keys (missing = the safest assumption):
            autonomy: one of AUTONOMY_LEVELS (default "personal_tool")
            domain: a string; see HIGHER_BAR_DOMAINS
            affects_people: bool
            errors_visible_to_affected_person: bool
            errors_recoverable: bool
            human_review: bool
            human_review_has_time: bool     is the review REAL?
            can_be_contested: bool
            measures_fairness: bool
            has_evaluation_set: bool
            discloses_ai_use: bool

    Returns:
        {
          "risk_level": "low" | "medium" | "high" | "very high",
          "score": int,
          "concerns": list[str],
          "required_before_building": list[str],
        }

    Scoring - each adds to the risk score:
        autonomy level                                   +0 to +3
        domain in HIGHER_BAR_DOMAINS                     +3
        affects_people                                   +1
        errors NOT visible to the affected person        +2
        errors NOT recoverable                           +2
        no human review, when autonomy >= recommendation +2
        human review present but WITHOUT time            +1
        cannot be contested, when affects_people         +2

    Bands: 0-2 low, 3-5 medium, 6-9 high, 10+ very high.

    required_before_building is raised at "high" or above, and lists the
    missing items among: an evaluation set, fairness measurement,
    AI disclosure, a contest route, and real human review.
    """
    # TODO:
    #   Work through the scoring table, appending a human-readable concern
    #   for each condition that fires. Say WHY it matters, not just what.
    #
    #   Then band the score, build required_before_building for high-risk
    #   systems, and return the dict.
    #
    # Note the human_review_has_time condition. "Human in the loop" that
    # rubber-stamps a hundred recommendations an hour is not review
    # (Module 14, section 14.9) - so claiming review without time scores
    # WORSE than being honest about having none, because it creates false
    # assurance.
    return {"risk_level": "low", "score": 0, "concerns": [],
            "required_before_building": []}


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

    # --- TASK 1 ---
    check("1. selection rates", selection_rates([1, 1, 0, 0], ["A", "A", "B", "B"]),
          {"A": 1.0, "B": 0.0})
    check("1. partial rates",
          selection_rates([1, 0, 1, 1], ["A", "A", "B", "B"]), {"A": 0.5, "B": 1.0})
    check("1. single group", selection_rates([1, 0], ["A", "A"]), {"A": 0.5})
    check("1. empty input", selection_rates([], []), {})
    check_raises("1. rejects mismatched lengths",
                 lambda: selection_rates([1, 0], ["A"]))

    # --- TASK 2 ---
    predictions = [1] * 50 + [0] * 50 + [1] * 30 + [0] * 70
    groups = ["A"] * 100 + ["B"] * 100
    check("2. disparate impact ratio",
          round(disparate_impact_ratio(predictions, groups), 4), 0.6)
    check("2. demographic parity difference",
          round(demographic_parity_difference(predictions, groups), 4), 0.2)

    at_threshold = [1] * 50 + [0] * 50 + [1] * 40 + [0] * 60
    check("2. exactly the four-fifths threshold",
          round(disparate_impact_ratio(at_threshold, groups), 4), 0.8)

    check("2. perfect parity gives 1.0",
          disparate_impact_ratio([1, 0, 1, 0], ["A", "A", "B", "B"]), 1.0)
    check("2. nobody selected anywhere gives 1.0, not a crash",
          disparate_impact_ratio([0, 0, 0, 0], ["A", "A", "B", "B"]), 1.0)
    check("2. empty input gives 1.0", disparate_impact_ratio([], []), 1.0)
    check("2. single group gives 1.0",
          disparate_impact_ratio([1, 0], ["A", "A"]), 1.0)
    check("2. parity difference is 0.0 for one group",
          demographic_parity_difference([1, 0], ["A", "A"]), 0.0)

    # --- TASK 3 ---
    check("3. true positive rates",
          true_positive_rates([1, 0, 1, 1], [1, 1, 1, 1], ["A", "A", "B", "B"]),
          {"A": 0.5, "B": 1.0})
    check("3. equal opportunity difference",
          round(equal_opportunity_difference(
              [1, 0, 1, 1], [1, 1, 1, 1], ["A", "A", "B", "B"]), 4), 0.5)
    check("3. rows where actual is negative are ignored",
          true_positive_rates([1, 1], [1, 0], ["A", "A"]), {"A": 1.0})
    check("3. a group with no positives is OMITTED, not scored 0.0",
          true_positive_rates([0, 0], [0, 0], ["A", "A"]), {})
    check_raises("3. rejects mismatched lengths",
                 lambda: true_positive_rates([1, 0], [1], ["A", "A"]))

    # --- TASK 4: THE DEMONSTRATION ---
    # Two groups, 100 each, 50 genuinely qualified in each.
    # Both groups selected at exactly 50%. But group A's picks are mostly
    # qualified people and group B's are mostly not.
    demo_predictions = (
        [1] * 45 + [0] * 5 + [1] * 5 + [0] * 45      # A: TPR 45/50 = 0.9
        + [1] * 25 + [0] * 25 + [1] * 25 + [0] * 25  # B: TPR 25/50 = 0.5
    )
    demo_actuals = [1] * 50 + [0] * 50 + [1] * 50 + [0] * 50
    demo_groups = ["A"] * 100 + ["B"] * 100

    report = fairness_report(demo_predictions, demo_groups, demo_actuals)
    check("4. selection rates are IDENTICAL",
          report.get("selection_rates"), {"A": 0.5, "B": 0.5})
    check("4. disparate impact looks PERFECT",
          report.get("disparate_impact_ratio"), 1.0)
    check("4. and it passes the four-fifths rule",
          report.get("passes_four_fifths"), True)
    check("4. but the true positive rates differ badly",
          report.get("true_positive_rates"), {"A": 0.9, "B": 0.5})
    check("4. equal opportunity gap",
          round(report.get("equal_opportunity_difference", 0), 4), 0.4)
    check("4. THE POINT: the metrics disagree, and it is detected",
          report.get("metrics_disagree"), True)

    agreeing = fairness_report([1] * 45 + [0] * 5 + [1] * 5 + [0] * 45,
                               ["A"] * 100, [1] * 50 + [0] * 50)
    check("4. one group does not trigger disagreement",
          agreeing.get("metrics_disagree"), False)

    no_truth = fairness_report(demo_predictions, demo_groups)
    check("4. without ground truth, opportunity metrics are unavailable",
          (no_truth.get("true_positive_rates"),
           no_truth.get("equal_opportunity_difference")), ({}, None))
    check("4. and disagreement cannot be claimed without measuring it",
          no_truth.get("metrics_disagree"), False)

    unfair = fairness_report(predictions, groups)
    check("4. a genuinely unequal system fails four-fifths",
          unfair.get("passes_four_fifths"), False)

    # --- TASK 5 ---
    result = energy_break_even(1000.0, 1.0, 1_000_000)
    check("5. training energy in kWh", result.get("training_kwh"), 1_000_000.0)
    check("5. inference energy per query in kWh",
          result.get("inference_kwh_per_query"), 0.001)
    check("5. break-even queries", result.get("break_even_queries"), 1_000_000_000)
    check("5. break-even days", result.get("break_even_days"), 1000.0)
    check("5. annual inference energy",
          round(result.get("inference_kwh_per_year", 0), 2), 365_000.0)

    heavy = energy_break_even(1000.0, 1.0, 100_000_000)
    check("5. at high volume, break-even arrives in days",
          round(heavy.get("break_even_days", 0), 1), 10.0)

    free = energy_break_even(1000.0, 0.0, 1_000_000)
    check("5. zero-energy inference never catches up",
          (free.get("break_even_queries"), free.get("break_even_days")),
          (None, None))

    # --- TASK 6 ---
    personal = impact_assessment({"autonomy": "personal_tool"})
    check("6. a personal tool is low risk", personal.get("risk_level"), "low")
    check("6. and needs nothing before building",
          personal.get("required_before_building"), [])

    hiring = impact_assessment({
        "autonomy": "automated_decision",
        "domain": "employment",
        "affects_people": True,
        "errors_visible_to_affected_person": False,
        "errors_recoverable": False,
        "human_review": False,
        "can_be_contested": False,
        "measures_fairness": False,
        "has_evaluation_set": False,
        "discloses_ai_use": False,
    })
    check("6. automated hiring decisions are very high risk",
          hiring.get("risk_level"), "very high")
    check("6. and raise several concerns",
          len(hiring.get("concerns", [])) >= 5, True)
    check("6. and require work before building",
          len(hiring.get("required_before_building", [])) >= 3, True)

    # Claiming review without time should score WORSE than admitting none.
    with_real_review = impact_assessment({
        "autonomy": "recommendation", "affects_people": True,
        "human_review": True, "human_review_has_time": True,
        "can_be_contested": True, "errors_visible_to_affected_person": True,
        "errors_recoverable": True,
    })
    with_rubber_stamp = impact_assessment({
        "autonomy": "recommendation", "affects_people": True,
        "human_review": True, "human_review_has_time": False,
        "can_be_contested": True, "errors_visible_to_affected_person": True,
        "errors_recoverable": True,
    })
    check("6. rubber-stamp review scores worse than real review",
          with_rubber_stamp.get("score", 0) > with_real_review.get("score", 0),
          True)

    invisible = impact_assessment({
        "autonomy": "recommendation", "affects_people": True,
        "errors_visible_to_affected_person": False, "errors_recoverable": False,
        "human_review": True, "human_review_has_time": True,
        "can_be_contested": True,
    })
    visible = impact_assessment({
        "autonomy": "recommendation", "affects_people": True,
        "errors_visible_to_affected_person": True, "errors_recoverable": True,
        "human_review": True, "human_review_has_time": True,
        "can_be_contested": True,
    })
    check("6. invisible, unrecoverable errors score worse",
          invisible.get("score", 0) - visible.get("score", 0), 4)

    check("6. a higher-bar domain raises the score by 3",
          impact_assessment({"autonomy": "informational", "domain": "health"}
                            ).get("score", 0)
          - impact_assessment({"autonomy": "informational", "domain": "retail"}
                              ).get("score", 0), 3)

    # --- report ---
    print()
    print("=" * 76)
    print("  LAB 14 SELF-TEST - measure it, then decide")
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
        print("  You can now measure fairness rather than assert it - and you")
        print("  have shown yourself that the metrics disagree.")
    else:
        print(f"  {failures} of {len(checks)} failing.")
        print("  Order: 1, 2, 3, 4 (uses all three), 5, 6.")
    print("-" * 76)
    print()
    return failures


# ======================================================================
# EXPERIMENTS
# ======================================================================

def experiment_metrics_disagree():
    print("=" * 76)
    print("  EXPERIMENT 1: the fairness metrics disagree")
    print("=" * 76)
    print()
    print("  A screening system. Two groups, 100 people each, and in each")
    print("  group exactly 50 are genuinely qualified.")
    print()

    scenarios = {
        "obviously unequal": {
            # A: 50 selected of 100. B: 30 selected of 100.
            "predictions": ([1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
                            + [1] * 25 + [0] * 25 + [1] * 5 + [0] * 45),
            "actuals": [1] * 50 + [0] * 50 + [1] * 50 + [0] * 50,
        },
        "equal rates, unequal accuracy": {
            # Both select 50. A picks 45 qualified; B picks 25.
            "predictions": ([1] * 45 + [0] * 5 + [1] * 5 + [0] * 45
                            + [1] * 25 + [0] * 25 + [1] * 25 + [0] * 25),
            "actuals": [1] * 50 + [0] * 50 + [1] * 50 + [0] * 50,
        },
        "genuinely fair": {
            "predictions": ([1] * 45 + [0] * 5 + [1] * 5 + [0] * 45
                            + [1] * 45 + [0] * 5 + [1] * 5 + [0] * 45),
            "actuals": [1] * 50 + [0] * 50 + [1] * 50 + [0] * 50,
        },
    }

    groups = ["A"] * 100 + ["B"] * 100

    print(f"  {'scenario':<32}{'DI ratio':>10}{'4/5?':>7}{'EO gap':>9}{'verdict':>12}")
    print("  " + "-" * 70)

    for label, data in scenarios.items():
        report = fairness_report(data["predictions"], groups, data["actuals"])
        if not report:
            print("  fairness_report not implemented yet.")
            print()
            return
        verdict = "DISAGREE" if report["metrics_disagree"] else (
            "unfair" if not report["passes_four_fifths"] else "fair")
        print(f"  {label:<32}{report['disparate_impact_ratio']:>10.2f}"
              f"{('yes' if report['passes_four_fifths'] else 'NO'):>7}"
              f"{report['equal_opportunity_difference']:>9.2f}{verdict:>12}")

    print()
    print("  Row 2 is the whole point of this experiment.")
    print()
    report = fairness_report(scenarios["equal rates, unequal accuracy"]["predictions"],
                             groups,
                             scenarios["equal rates, unequal accuracy"]["actuals"])
    print(f"    selection rates    : {report['selection_rates']}")
    print(f"    disparate impact   : {report['disparate_impact_ratio']:.2f}"
          f"  <- looks PERFECT")
    print(f"    true positive rates: {report['true_positive_rates']}")
    print(f"    opportunity gap    : {report['equal_opportunity_difference']:.2f}"
          f"  <- badly unfair")
    print()
    print("  Identical selection rates. It passes the two most commonly cited")
    print("  metrics with a perfect score.")
    print()
    print("  And it is far worse at finding qualified people in group B - it")
    print("  reaches the same headline number by selecting 25 candidates it")
    print("  should not have. A qualified person in group B is much less")
    print("  likely to be picked than a qualified person in group A.")
    print()
    print("  It is mathematically impossible to satisfy every fairness")
    print("  definition at once. So WHICH ONE you optimise is a values")
    print("  decision, and it should be written down and defended - not left")
    print("  to whichever metric your library happens to compute.")
    print()


def experiment_energy():
    print("=" * 76)
    print("  EXPERIMENT 2: training versus inference energy")
    print("=" * 76)
    print()
    print("  ILLUSTRATIVE figures. Published estimates vary by orders of")
    print("  magnitude and go stale quickly - the SHAPE is the robust part,")
    print("  not the numbers.")
    print()

    TRAINING_MWH = 1000.0
    INFERENCE_WH = 1.0

    print(f"  Assume: {TRAINING_MWH:,.0f} MWh to train, "
          f"{INFERENCE_WH} Wh per query.")
    print()
    print(f"  {'queries/day':>16}{'break-even':>16}{'annual inference':>20}")
    print("  " + "-" * 52)

    for per_day in [10_000, 1_000_000, 100_000_000, 1_000_000_000]:
        result = energy_break_even(TRAINING_MWH, INFERENCE_WH, per_day)
        if result.get("break_even_days") is None:
            print("  energy_break_even not implemented yet.")
            print()
            return
        days = result["break_even_days"]
        when = f"{days:,.0f} days" if days >= 1 else f"{days*24:,.1f} hours"
        annual = result["inference_kwh_per_year"]
        print(f"  {per_day:>16,}{when:>16}{f'{annual:,.0f} kWh':>20}")

    print()
    print("  At a million queries a day, cumulative inference overtakes")
    print("  training in under three years. At a hundred million a day it")
    print("  takes ten.")
    print()
    print("  For any widely deployed model, INFERENCE dominates its lifetime")
    print("  footprint - which is the opposite of how this is usually stated.")
    print()
    print("  That is also the good news, because inference is the part YOU")
    print("  control. Every optimisation from Module 13 - caching, smaller")
    print("  models, shorter prompts, fewer retrieved chunks - is an")
    print("  environmental lever as well as a cost one.")
    print()


def experiment_impact_assessment():
    print("=" * 76)
    print("  EXPERIMENT 3: what needs more care before you build it")
    print("=" * 76)
    print()

    systems = {
        "Personal note summariser": {
            "autonomy": "personal_tool",
        },
        "Docs Q&A bot (your Module 8 build)": {
            "autonomy": "informational",
            "affects_people": True,
            "errors_visible_to_affected_person": True,
            "errors_recoverable": True,
            "discloses_ai_use": True,
            "has_evaluation_set": True,
        },
        "Support ticket router": {
            "autonomy": "recommendation",
            "affects_people": True,
            "errors_visible_to_affected_person": False,
            "errors_recoverable": True,
            "human_review": True,
            "human_review_has_time": False,
            "can_be_contested": False,
        },
        "CV screening for hiring": {
            "autonomy": "automated_decision",
            "domain": "employment",
            "affects_people": True,
            "errors_visible_to_affected_person": False,
            "errors_recoverable": False,
            "human_review": False,
            "can_be_contested": False,
            "measures_fairness": False,
            "has_evaluation_set": False,
            "discloses_ai_use": False,
        },
    }

    for label, system in systems.items():
        result = impact_assessment(system)
        if not result:
            print("  impact_assessment not implemented yet.")
            print()
            return
        print(f"  {label}")
        print(f"    risk: {result['risk_level'].upper()} (score {result['score']})")
        for concern in result["concerns"]:
            print(f"      - {concern}")
        if result["required_before_building"]:
            print("    REQUIRED BEFORE BUILDING:")
            for requirement in result["required_before_building"]:
                print(f"      * {requirement}")
        print()

    print("  The support router is the interesting one. It scores worse than")
    print("  it would if it had NO human review at all - because it claims")
    print("  review that has no time to actually review.")
    print()
    print("  That is deliberate. A reviewer approving a hundred confident")
    print("  recommendations an hour is not reviewing; they are providing")
    print("  false assurance, and everyone downstream believes a human")
    print("  checked it.")
    print()
    print("  Honest 'no review' is safer than dishonest 'review', because at")
    print("  least the risk is visible.")
    print()


def experiment_your_project():
    print("=" * 76)
    print("  EXPERIMENT 4: assess something you actually want to build")
    print("=" * 76)
    print()
    print("  Edit MY_PROJECT below to describe a system you are genuinely")
    print("  considering, then re-run this file.")
    print()

    MY_PROJECT = {
        # --- edit these ---
        "autonomy": "informational",
        "domain": "general",
        "affects_people": True,
        "errors_visible_to_affected_person": True,
        "errors_recoverable": True,
        "human_review": False,
        "human_review_has_time": False,
        "can_be_contested": False,
        "measures_fairness": False,
        "has_evaluation_set": False,
        "discloses_ai_use": False,
    }

    result = impact_assessment(MY_PROJECT)
    if not result:
        print("  impact_assessment not implemented yet.")
        print()
        return

    print(f"  risk level: {result['risk_level'].upper()} (score {result['score']})")
    print()
    if result["concerns"]:
        print("  concerns:")
        for concern in result["concerns"]:
            print(f"    - {concern}")
    else:
        print("  no concerns raised.")
    print()
    if result["required_before_building"]:
        print("  do these first:")
        for requirement in result["required_before_building"]:
            print(f"    * {requirement}")
        print()

    print("  Answer honestly. A checklist you flatter yourself through is")
    print("  worse than none, because it produces a defensible-looking")
    print("  document and changes nothing.")
    print()
    print("  And note that 'do not build it' is a legitimate outcome. It is")
    print("  an engineering decision like any other, and often the right one.")
    print()


if __name__ == "__main__":
    failures = _run_self_test()
    if failures == 0:
        experiment_metrics_disagree()
        experiment_energy()
        experiment_impact_assessment()
        experiment_your_project()
    else:
        print("  Fix the self-test first, then the experiments will run.")
        print()
