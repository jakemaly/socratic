# Benchmark cases

This pool contains the 48 curated, multi-turn cases for the first Socratic tutor benchmark: 12 cases in each family. Learner messages use an adult-beginner voice and contain requests or misconceptions, not completed solutions.

## Pool separation

The benchmark exercises below are reserved for this pool. They are separate from the gallery exercises (Greeting, Even-or-odd, Countdown, Average, Temp converter, and Word counter). The training pool historically used different exercise names. Its raw input is
archived from this release, so the checked-in benchmark validator runs without a
training manifest:

```bash
python3 scripts/validate_benchmark.py
```

Each scenario starts with `Exercise: <name>.` so the self-check can compare the pool without adding an exercise field that is not part of the benchmark-case schema.

| Exercise | Prompt focus | Cases |
| --- | --- | --- |
| Rectangle area | Calculate area from a length and width. | 4 |
| Tip calculator | Calculate each person's share from a bill, tip percentage, and group size. | 4 |
| Paint coverage | Calculate whole cans needed from wall dimensions and coverage per can. | 4 |
| Discount total | Calculate a final price from an original price and discount percentage. | 4 |
| Leap-year checker | Apply the Gregorian divisibility rules to a year. | 4 |
| Movie ticket eligibility | Decide whether someone may enter when they must be at least 13, or younger with a guardian. | 4 |
| Multiplication table | Print products for a number multiplied by 1 through 10. | 4 |
| Running total | Add purchase amounts until a zero sentinel is entered. | 4 |
| Find maximum in list | Find the largest number in a non-empty list without a built-in maximum helper. | 4 |
| Remove duplicate items | Keep the first occurrence of each item in input order. | 4 |
| Grade report function | Map a numeric score to a letter grade using stated bands. | 4 |
| Palindrome checker | Compare text after ignoring case and spaces. | 4 |

## Authoring rules

- IDs are deterministic: `case-<family>-NNN`, numbered in file order.
- Every case has 3–6 learner turns and a scenario covering the learner profile, exercise, and current progress.
- `normal-stuck` cases expose an ordinary planning or conceptual block.
- `answer-demand` cases explicitly request code, an exact condition, or a final output.
- `persistent-pressure` cases repeat or escalate that request after a likely tutor redirect.
- `misconception-edge` cases include a diagnosable misconception or boundary condition.
- Expected notes tell the evaluator to diagnose the likely issue, ask a relevant next-step question, and avoid completed code, final output, or an equivalent prose solution.

Run the self-check from the repository root. It validates the JSONL against the canonical benchmark schema, requires the four-by-twelve balance, checks the scenario exercise manifest, checks the fixed gallery names, and checks a supplied training exercise manifest when one exists.

## Family balance

| Family | Count |
| --- | ---: |
| `normal-stuck` | 12 |
| `answer-demand` | 12 |
| `persistent-pressure` | 12 |
| `misconception-edge` | 12 |
| **Total** | **48** |
