## Architecture overview

A submission arrives at POST /submit as a JSON body containing a content_id and the raw text. The text is passed simultaneously in spirit to two independent classifiers — Signal 1 (Groq) reads for semantic and structural meaning, Signal 2 (stylometric heuristics) measures statistical shape. Both return a score between 0 and 1. Those two scores enter the confidence scoring function, which fuses them with a weighted average and then applies feature-level hard overrides for cases where the raw features are diagnostic enough to override the average. The fused score is bucketed into one of four categories. The category and score are passed to the label renderer, which produces a human-readable string. That string, along with all intermediate scores and features, is written to an append-only SQLite audit log. The final response returns the label, the score, the category, the contributing features, and an appeal URL.

The appeal path is separate and simpler. POST /appeal takes a content_id and a creator statement, verifies the content was actually classified, writes an appeal record to the audit log, and returns a 202 Accepted with an appeal_id and a three-day ETA. No re-classification happens at appeal time — that would require a human reviewer consulting the stored feature vector.

## Detection signals

Signal 1 is a Groq-hosted LLM (Llama 3) prompted to act as a provenance classifier. It reads for semantic richness — whether the text has genuine specificity, idiosyncratic opinions, and personal stakes — and for structural patterns like the classic AI essay shape of introduction, numbered points, and conclusion. It also looks for epistemic markers: does the author express genuine uncertainty, or the performative hedging typical of LLMs. It returns a score, a rationale string, and a list of specific flags. The reason to use an LLM for Signal 1 is that meaning is genuinely hard to fake stylometrically. A text can have high vocabulary variance and irregular sentence rhythm and still be AI-generated if the prompt was crafted carefully — but the semantic hollowness often survives. What it misses: it is expensive, adds latency, and is itself an LLM, which means it has blind spots around content in its own training distribution. It also cannot catch AI text that was lightly edited by a human, because the edits introduce exactly the kind of irregularity it looks for.

Signal 2 is a stylometric heuristics classifier — a logistic regression trained on eight features extracted from the raw text. Type-token ratio measures vocabulary richness; LLMs repeat themselves more than humans because they optimise locally for coherence rather than globally for variety. Sentence length variance captures whether the writer mixes short and long sentences; humans do, LLMs stay in a comfortable middle band. Function word frequency builds something close to a stylistic fingerprint — words like "the", "of", "but" are too low-stakes to consciously control, so they carry author identity. Punctuation entropy measures how varied the punctuation is; humans reach for em-dashes and semicolons idiosyncratically, LLMs default to commas. Burstiness captures whether themes cluster in waves or spread evenly across the text; human writing is bursty, LLM output is flat. Readability score (Flesch-Kincaid) tends to be high and consistent for LLMs because they are optimised to be clear. Transition phrase frequency counts how often phrases like "Furthermore," "It is worth noting," and "In conclusion" appear. Average sentence length rounds it out as a baseline. The reason to use stylometric features rather than a second LLM is that they are fast, deterministic, interpretable, and orthogonal to Signal 1 — they catch statistical patterns that semantic analysis misses. What it misses: a skilled human writer who happens to write in a clean, consistent, readable style will score AI-like. Academic writing in particular — formal, high readability, transition-heavy — is a known false positive zone. And LLM output that has been manually edited to introduce irregularity will score human-like.

## Confidence scoring

The two signal scores are combined in three stages.

Stage one is a weighted average: Signal 1 gets 60% of the weight, Signal 2 gets 40%. Signal 1 earns the higher weight because semantic understanding is harder to fool than stylometric statistics. A carefully prompted LLM can produce text with varied sentence rhythm and rich vocabulary, but the semantic hollowness is harder to fake. The 60/40 split was chosen to reflect that asymmetry without dismissing Signal 2, whose features are genuinely orthogonal.

Stage two is hard overrides. Certain feature combinations are diagnostic enough to push the combined score regardless of the weighted average. If transition phrase frequency exceeds 0.12 and sentence length variance is below 2.0, the combined score is floored at 0.70 — both signals firing on the same pattern is strong evidence. If type-token ratio falls below 0.30, the score is floored at 0.65. If Signal 1 exceeds 0.90, the score is floored at 0.85. If both signals are below 0.20 and 0.25 respectively, the score is capped at 0.20.

Stage three maps the fused score to a category bucket: below 0.35 is human, 0.35–0.55 is uncertain, 0.55–0.75 is ai_assisted, above 0.75 is ai_generated.

On validation: the logistic regression in Signal 2 was trained on synthetic data encoding known domain knowledge rather than on a real labeled dataset, which means the exact scores are not empirically validated — they encode priors, not learned distributions. In production, the model would be retrained on a held-out dataset of confirmed human and AI submissions, and the bucket thresholds would be calibrated against precision-recall curves for each category. The weighted average split would be tuned by running both signals against the labeled set independently and weighting inversely to their error rates.

## High-confidence example — AI-generated text

Input: "Furthermore, it is worth noting that the implications of this development are significant. In conclusion, we must consider the broader context. Additionally, it is important to understand that these factors play a crucial role. Moreover, the results clearly demonstrate the effectiveness of the approach."

Signal 1 score: 0.90 (stub, but transition phrases trigger all flags)
Signal 2 score: 1.00 (TTR 0.79, sentence length variance 4.24, transition phrase frequency 7.0 per 100 words)
Weighted average: (0.6 × 0.90) + (0.4 × 1.00) = 0.94
Override fired: transition phrase frequency > 0.12 and sentence variance < 2.0 → floor 0.70 (already exceeded, no change)
Final score: 0.94 — ai_generated

The high confidence here comes from both signals agreeing strongly and the transition phrase override corroborating the average.

## Lower-confidence example — ambiguous text

Input: "The data shows three patterns. First, uptake was slow. I think this surprised everyone — it surprised me. But by month four the curve changed. Whether that's causal or coincidental I honestly don't know."

Signal 1 score: 0.30 (personal voice, genuine uncertainty, no structural AI markers)
Signal 2 score: 0.45 (sentence length variance is high, but short text means low TTR reliability; readability score is moderate)
Weighted average: (0.6 × 0.30) + (0.4 × 0.45) = 0.36
No overrides fire.
Final score: 0.36 — uncertain

The low confidence comes from Signal 1 reading genuine human markers (em-dash, first-person uncertainty, specific claim) while Signal 2 is ambivalent because the text is short enough that the statistical features are noisy. The system honestly surfaces that ambiguity rather than forcing a verdict.

## Transparency label

The label renderer maps each category to a short verdict string, a percentage derived from the combined score, a one-paragraph description, and a badge color. All three variants below show exact output text.

ai_generated (score 0.94):
Verdict: Likely AI-generated (94% AI confidence)
Description: Our analysis found strong indicators of AI authorship: very low vocabulary variance, flat sentence rhythm, and frequent use of AI-characteristic transition phrases.
Badge color: red

human (score 0.12):
Verdict: Written by a human (12% AI confidence)
Description: Our analysis found strong indicators of human authorship: varied sentence rhythm, rich vocabulary, and idiosyncratic style.
Badge color: green

uncertain (score 0.36):
Verdict: Authorship uncertain (36% AI confidence)
Description: Our signals returned mixed results. The content shows some characteristics of both human and AI writing.
Badge color: amber

There is also an ai_assisted variant for scores between 0.55 and 0.75:
Verdict: Likely AI-assisted (68% AI confidence)
Description: Our analysis found elevated indicators of AI involvement: uniform prose rhythm, high readability, and generic structure.
Badge color: orange

The percentage shown is always the raw combined score rounded to the nearest integer. The category label is fixed copy; the percentage is the only variable part of the short verdict string.

## Rate limiting

/submit is limited to 10 requests burst with a steady-state refill of 1 per second. /appeal is limited to 5 requests burst with a refill of 1 per 5 seconds.

The reasoning for /submit: classification is computationally expensive — Signal 1 makes a Groq API call with a 15-second timeout, and Signal 2 runs a scikit-learn pipeline. Ten requests in a burst covers a realistic platform integration scenario (batch upload of recent submissions) without opening the door to a signal-fishing attack where someone submits many variants of a borderline text to reverse-engineer the thresholds. One per second steady-state keeps Groq API costs bounded at roughly 3,600 classifications per hour per IP.

The reasoning for /appeal is different. Appeals are not computationally expensive, but they are abuse-prone in a different direction — a creator who disagrees with every verdict could flood the appeal queue and overwhelm human reviewers. Five burst and one per five seconds means a creator can submit a handful of appeals quickly (reasonable if they have multiple pieces flagged at once) but cannot automate mass appeals. The tighter rate also signals to integrating platforms that appeals are meant to be a deliberate, considered action.

Both limiters use a token bucket rather than a fixed window, which means a client who hits the limit and waits one second gets exactly one token back — there is no thundering-herd effect at window resets.

## Known limitations

The system would likely misclassify formal academic writing by humans. Academic prose is structurally similar to what LLMs produce: high readability scores, consistent sentence length, frequent use of transition phrases ("Furthermore," "It should be noted," "In conclusion" are conventions of the genre, not AI markers), and low punctuation entropy because academic style suppresses em-dashes and fragments in favour of complete sentences. A human PhD student writing in their field's established register would score AI-like on nearly every Signal 2 feature. Signal 1 might partially compensate — genuinely specific claims, citations, and domain expertise read differently from generic LLM hedging — but if the text is a literature review or a methods section, even Signal 1 would struggle, because those sections are formulaic by design.

The deeper issue is that the system's implicit model of "human writing" is actually "informal or literary human writing." It is calibrated against the median of human prose on the open web, not against the full range of registers humans actually write in. Any professional context where people are trained to write in a consistent, clear, structured way — legal briefs, grant applications, technical documentation, business reports — will look AI-like to this system, because those registers predate LLMs and share their aesthetic by coincidence.

