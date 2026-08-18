"""
solution.py - Lab 14 reference solution.

Attempt starter.py first. See SOLUTION.md for the reasoning.

    python solution.py
"""

import math
from collections import defaultdict


# ======================================================================
# TASK 1 - selection_rates
# ======================================================================

def selection_rates(predictions: list, groups: list) -> dict:
    """Fraction of each group receiving a positive outcome."""
    # zip() silently truncates to the shorter list, which would give a wrong
    # answer with no error. Check explicitly.
    if len(predictions) != len(groups):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions, "
            f"{len(groups)} groups")

    totals = defaultdict(int)
    positives = defaultdict(int)

    for prediction, group in zip(predictions, groups):
        totals[group] += 1
        if prediction:
            positives[group] += 1

    return {group: positives[group] / totals[group]
            for group in totals if totals[group] > 0}


# ======================================================================
# TASK 2 - disparate impact and demographic parity
# ======================================================================

FOUR_FIFTHS = 0.8


def disparate_impact_ratio(predictions: list, groups: list) -> float:
    """Ratio of the lowest selection rate to the highest."""
    rates = selection_rates(predictions, groups)

    if not rates:
        return 1.0

    lowest = min(rates.values())
    highest = max(rates.values())

    # Nobody selected anywhere. Equally bad for everyone, and it avoids a
    # division by zero that would otherwise crash on a perfectly ordinary
    # input (a filter that matched nothing).
    if highest == 0:
        return 1.0

    return lowest / highest


def demographic_parity_difference(predictions: list, groups: list) -> float:
    """Largest gap between any two groups' selection rates."""
    rates = selection_rates(predictions, groups)
    if not rates:
        return 0.0
    return max(rates.values()) - min(rates.values())


# ======================================================================
# TASK 3 - true positive rates and equal opportunity
# ======================================================================

def true_positive_rates(predictions: list, actuals: list, groups: list) -> dict:
    """Of those who SHOULD be positive, what fraction were?"""
    if not (len(predictions) == len(actuals) == len(groups)):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions, "
            f"{len(actuals)} actuals, {len(groups)} groups")

    should_be_positive = defaultdict(int)
    correctly_positive = defaultdict(int)

    for prediction, actual, group in zip(predictions, actuals, groups):
        # Only rows that SHOULD be positive contribute. Negatives are
        # irrelevant to this metric - that is what makes it "equal
        # opportunity" rather than "equal outcome".
        if actual:
            should_be_positive[group] += 1
            if prediction:
                correctly_positive[group] += 1

    # OMIT groups with no positives rather than recording 0.0. There is no
    # rate to compute, and a 0.0 would read as total failure instead of
    # no data - which would then poison the max/min in the gap calculation.
    return {group: correctly_positive[group] / should_be_positive[group]
            for group in should_be_positive if should_be_positive[group] > 0}


def equal_opportunity_difference(predictions: list, actuals: list,
                                 groups: list) -> float:
    """Largest gap between any two groups' true positive rates."""
    rates = true_positive_rates(predictions, actuals, groups)
    if not rates:
        return 0.0
    return max(rates.values()) - min(rates.values())


# ======================================================================
# TASK 4 - fairness_report
# ======================================================================

DISAGREEMENT_THRESHOLD = 0.1


def fairness_report(predictions: list, groups: list,
                    actuals: list = None) -> dict:
    """Compute several fairness metrics together, and flag disagreement."""
    rates = selection_rates(predictions, groups)
    ratio = disparate_impact_ratio(predictions, groups)
    parity_gap = demographic_parity_difference(predictions, groups)
    passes = ratio >= FOUR_FIFTHS

    report = {
        "selection_rates": rates,
        "disparate_impact_ratio": ratio,
        "passes_four_fifths": passes,
        "demographic_parity_difference": parity_gap,
        "true_positive_rates": {},
        "equal_opportunity_difference": None,
        "metrics_disagree": False,
    }

    if actuals is None:
        # You cannot claim disagreement you have not measured. Without ground
        # truth, only the selection-rate view exists.
        return report

    tprs = true_positive_rates(predictions, actuals, groups)
    opportunity_gap = equal_opportunity_difference(predictions, actuals, groups)

    report["true_positive_rates"] = tprs
    report["equal_opportunity_difference"] = opportunity_gap

    # THE POINT of this function: the selection-rate view can look perfect
    # while the opportunity view is badly unfair.
    report["metrics_disagree"] = (passes
                                  and opportunity_gap > DISAGREEMENT_THRESHOLD)

    return report


# ======================================================================
# TASK 5 - energy_break_even
# ======================================================================

def energy_break_even(training_mwh: float, inference_wh_per_query: float,
                      queries_per_day: int) -> dict:
    """When does cumulative inference energy overtake training energy?"""
    training_kwh = training_mwh * 1000.0
    inference_kwh_per_query = inference_wh_per_query / 1000.0

    if inference_kwh_per_query <= 0:
        # Free inference never catches up, however long you wait.
        break_even_queries = None
        break_even_days = None
    else:
        break_even_queries = int(math.ceil(training_kwh / inference_kwh_per_query))
        break_even_days = (break_even_queries / queries_per_day
                           if queries_per_day > 0 else None)

    return {
        "training_kwh": training_kwh,
        "inference_kwh_per_query": inference_kwh_per_query,
        "break_even_queries": break_even_queries,
        "break_even_days": break_even_days,
        "inference_kwh_per_year": (inference_kwh_per_query
                                   * max(queries_per_day, 0) * 365),
    }


# ======================================================================
# TASK 6 - impact_assessment
# ======================================================================

AUTONOMY_LEVELS = {
    "personal_tool": 0,
    "informational": 1,
    "recommendation": 2,
    "automated_decision": 3,
}

HIGHER_BAR_DOMAINS = {
    "health", "legal", "financial", "employment",
    "education", "criminal_justice", "children",
}


def impact_assessment(system: dict) -> dict:
    """Assess how much care a proposed system needs before building it."""
    concerns = []
    score = 0

    autonomy = system.get("autonomy", "personal_tool")
    autonomy_score = AUTONOMY_LEVELS.get(autonomy, 0)
    score += autonomy_score

    if autonomy_score >= 3:
        concerns.append(
            "the system decides automatically - nobody sees the decision "
            "before it takes effect")
    elif autonomy_score == 2:
        concerns.append(
            "someone acts on its recommendations, so its errors become their "
            "actions")

    domain = system.get("domain", "general")
    if domain in HIGHER_BAR_DOMAINS:
        score += 3
        concerns.append(
            f"{domain} is a higher-bar domain: consequences are serious and "
            f"regulation likely applies")

    affects_people = bool(system.get("affects_people"))
    if affects_people:
        score += 1

    if affects_people and not system.get("errors_visible_to_affected_person"):
        score += 2
        concerns.append(
            "errors are invisible to the person affected, so they cannot "
            "notice or challenge them")

    if affects_people and not system.get("errors_recoverable"):
        score += 2
        concerns.append("errors are not recoverable")

    has_review = bool(system.get("human_review"))

    if autonomy_score >= 2 and not has_review:
        score += 2
        concerns.append("no human review on a system that drives real actions")

    if has_review and not system.get("human_review_has_time"):
        score += 1
        concerns.append(
            "human review exists but has no time to be real - a rubber stamp "
            "provides false assurance to everyone downstream")

    if affects_people and not system.get("can_be_contested"):
        score += 2
        concerns.append(
            "the affected person has no route to contest the outcome")

    if score <= 2:
        risk_level = "low"
    elif score <= 5:
        risk_level = "medium"
    elif score <= 9:
        risk_level = "high"
    else:
        risk_level = "very high"

    required = []
    if risk_level in ("high", "very high"):
        if not system.get("has_evaluation_set"):
            required.append(
                "build an evaluation set with a baseline number "
                "(Module 11, section 11.7)")
        if not system.get("measures_fairness"):
            required.append(
                "measure outcomes per group, and decide WHICH fairness "
                "definition you are optimising")
        if not system.get("discloses_ai_use"):
            required.append("disclose that AI is involved")
        if affects_people and not system.get("can_be_contested"):
            required.append("provide a route to contest the outcome")
        if autonomy_score >= 2 and not (has_review
                                        and system.get("human_review_has_time")):
            required.append(
                "provide human review with the time and information to "
                "genuinely disagree")

    return {"risk_level": risk_level, "score": score, "concerns": concerns,
            "required_before_building": required}


# ======================================================================
# BONUS - things the tasks do not cover
# ======================================================================

def false_positive_rates(predictions: list, actuals: list, groups: list) -> dict:
    """Of those who should be NEGATIVE, what fraction were flagged anyway?

    The mirror of equal opportunity, and the metric that matters when the
    harm is being wrongly flagged rather than wrongly excluded - fraud
    detection, content moderation, risk scoring.
    """
    if not (len(predictions) == len(actuals) == len(groups)):
        raise ValueError("length mismatch")

    should_be_negative = defaultdict(int)
    wrongly_positive = defaultdict(int)

    for prediction, actual, group in zip(predictions, actuals, groups):
        if not actual:
            should_be_negative[group] += 1
            if prediction:
                wrongly_positive[group] += 1

    return {group: wrongly_positive[group] / should_be_negative[group]
            for group in should_be_negative if should_be_negative[group] > 0}


def intersectional_rates(predictions: list, group_lists: dict) -> dict:
    """Selection rates across COMBINATIONS of attributes.

    A system can look fair on each attribute separately and be badly unfair
    on a combination. Checking attributes one at a time misses it entirely.

    Args:
        predictions: What the system decided.
        group_lists: {attribute_name: [value per prediction]}

    Returns:
        {(value1, value2, ...): rate}
    """
    names = sorted(group_lists)
    combined = [tuple(group_lists[name][i] for name in names)
                for i in range(len(predictions))]
    return selection_rates(predictions, combined)


def carbon_from_energy(kwh: float, grid_gco2_per_kwh: float = 400.0) -> dict:
    """Convert energy to emissions.

    Args:
        kwh:                Energy consumed.
        grid_gco2_per_kwh:  Grid carbon intensity. This varies by more than
                            10x between regions - which is why an energy
                            figure without a grid figure tells you very little.

    Returns:
        {"kg_co2", "grid_gco2_per_kwh"}
    """
    return {"kg_co2": kwh * grid_gco2_per_kwh / 1000.0,
            "grid_gco2_per_kwh": grid_gco2_per_kwh}


# ======================================================================
# Demonstrations
# ======================================================================

def demo_removing_the_attribute_does_not_help():
    print("=" * 76)
    print("  'WE REMOVED THE PROTECTED ATTRIBUTE' DOES NOT WORK")
    print("=" * 76)
    print()

    import random
    rng = random.Random(42)

    # A population where a PROXY feature correlates with group membership.
    # Postcode, school, employment gaps - all of these do this in practice.
    people = []
    for _ in range(1000):
        group = "A" if rng.random() < 0.5 else "B"
        # The proxy is higher for group A, on average. Nobody designed this;
        # it reflects a real-world correlation.
        proxy = rng.gauss(70 if group == "A" else 55, 12)
        # Genuine qualification is INDEPENDENT of group.
        qualified = rng.random() < 0.5
        people.append({"group": group, "proxy": proxy, "qualified": qualified})

    # A model that never sees `group` - it only uses the proxy.
    predictions = [1 if person["proxy"] > 62 else 0 for person in people]
    groups = [person["group"] for person in people]
    actuals = [1 if person["qualified"] else 0 for person in people]

    report = fairness_report(predictions, groups, actuals)

    print("  The model never sees group membership. It only uses one")
    print("  seemingly neutral feature that happens to correlate with it.")
    print()
    print(f"    selection rates       : "
          f"{ {k: round(v, 3) for k, v in report['selection_rates'].items()} }")
    print(f"    disparate impact ratio: {report['disparate_impact_ratio']:.2f}")
    print(f"    passes four-fifths    : "
          f"{'yes' if report['passes_four_fifths'] else 'NO'}")
    print()
    print("  Removing the protected attribute removed your ability to MEASURE")
    print("  the disparity. It did not remove the disparity.")
    print()
    print("  Worse: had we not recorded group membership for this audit, we")
    print("  could not have produced these numbers at all - and the system")
    print("  would look clean because nothing was being checked.")
    print()


def demo_intersectionality():
    print("=" * 76)
    print("  FAIR ON EACH ATTRIBUTE, UNFAIR ON THE COMBINATION")
    print("=" * 76)
    print()

    # Four intersectional cells of 100 people each, chosen so that EACH
    # attribute passes the four-fifths rule on its own while one cell is
    # selected at half the rate of another.
    #
    #   attribute 1:  X = (80+40)/200 = 0.60   Y = (40+70)/200 = 0.55  -> 0.92
    #   attribute 2:  P = (80+40)/200 = 0.60   Q = (40+70)/200 = 0.55  -> 0.92
    #   cells:        0.80, 0.40, 0.40, 0.70                           -> 0.50
    cells = {("X", "P"): 80, ("X", "Q"): 40, ("Y", "P"): 40, ("Y", "Q"): 70}

    predictions, attribute_1, attribute_2 = [], [], []
    for (a1, a2), selected in cells.items():
        predictions.extend([1] * selected + [0] * (100 - selected))
        attribute_1.extend([a1] * 100)
        attribute_2.extend([a2] * 100)

    print("  Checking ONE attribute at a time:")
    for name, values in [("attribute 1", attribute_1), ("attribute 2", attribute_2)]:
        ratio = disparate_impact_ratio(predictions, values)
        rates = {k: round(v, 3) for k, v in selection_rates(predictions, values).items()}
        verdict = "passes" if ratio >= FOUR_FIFTHS else "FAILS"
        print(f"    {name}: {rates}  ratio {ratio:.2f}  {verdict}")
    print()

    print("  Checking the COMBINATION:")
    combined = intersectional_rates(predictions,
                                    {"a1": attribute_1, "a2": attribute_2})
    for key in sorted(combined):
        print(f"    {key}: {combined[key]:.2f}")

    ratio = min(combined.values()) / max(combined.values())
    print(f"    ratio {ratio:.2f}  "
          f"{'passes' if ratio >= FOUR_FIFTHS else 'FAILS'}")
    print()
    print("  Both attributes pass the four-fifths rule on their own - 0.92")
    print("  each. Yet (X, Q) and (Y, P) are selected at HALF the rate of")
    print("  (X, P), and the combined ratio of 0.50 fails badly.")
    print()
    print("  Auditing attributes one at a time misses this entirely, and one")
    print("  at a time is how it is usually done.")
    print()


def demo_carbon_depends_on_the_grid():
    print("=" * 76)
    print("  WHY AN ENERGY FIGURE WITHOUT A GRID FIGURE MEANS LITTLE")
    print("=" * 76)
    print()

    energy = energy_break_even(1000.0, 1.0, 10_000_000)
    annual_kwh = energy["inference_kwh_per_year"]

    # Illustrative grid intensities. Real values vary by region, season and
    # time of day.
    grids = {
        "very low carbon (hydro/nuclear)": 30.0,
        "low carbon (heavy renewables)": 150.0,
        "world average, roughly": 400.0,
        "coal-heavy": 800.0,
    }

    print(f"  Same workload throughout: {annual_kwh:,.0f} kWh of inference")
    print(f"  per year (10M queries/day at 1 Wh each).")
    print()
    print(f"  {'grid':<34}{'gCO2/kWh':>11}{'tonnes CO2/yr':>16}")
    print("  " + "-" * 61)

    for label, intensity in grids.items():
        result = carbon_from_energy(annual_kwh, intensity)
        print(f"  {label:<34}{intensity:>11.0f}{result['kg_co2']/1000:>16,.0f}")

    print()
    print("  Identical energy. A 26x spread in emissions.")
    print()
    print("  So 'this model emits X tonnes' is not a property of the model.")
    print("  It is a property of the model AND where it runs AND when.")
    print()
    print("  That is why published figures vary so much, and why quoting one")
    print("  without its assumptions is close to meaningless.")
    print()


if __name__ == "__main__":
    demo_removing_the_attribute_does_not_help()
    demo_intersectionality()
    demo_carbon_depends_on_the_grid()
