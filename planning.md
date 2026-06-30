## Planning

**Detection Signals:**
Detection signals
* Signal 1 — Groq

What it measures: the statistical predictability of each token in the submission given its context, using a reference language model. Human writing tends to alternate between predictable and surprising word choices — some sentences are highly foreseeable, others deviate sharply from what the model expects. This variation is called burstiness. AI-generated text, by contrast, tends to occupy a narrow, medium-entropy band throughout: rarely very surprising, rarely very predictable, just consistently smooth. The perplexity signal captures this by computing the average log-probability of each token and the variance of that distribution across the full submission.
What the output looks like: a continuous float between 0 and 1, where values closer to 1 indicate low-entropy, AI-like smoothness and values closer to 0 indicate high-entropy, human-like burstiness. The raw perplexity and burstiness values are computed separately and then linearly scaled into this 0–1 range before being passed downstream.
Strengths: fast, cheap, requires no fine-tuning, and is interpretable — a reviewer can understand what "low burstiness" means without a statistics background.
Known weaknesses: heavily penalizes short submissions, where there is not enough text for the entropy distribution to stabilize. Also penalizes non-native English speakers and writers who deliberately use plain, repetitive style — both of whom produce low-burstiness text for reasons entirely unrelated to AI use.
Minimum viable input length: the signal should be suppressed and logged as null for submissions under approximately 150 tokens, rather than producing a score that will be noisy and unreliable.

* Signal 2 — Stylometric heuristics classifier

It measures the statistical shape of the writing. Type-token ratio captures vocabulary richness (unique words divided by total words); LLMs repeat themselves more than humans do. Sentence length variance captures whether the writer mixes short punchy sentences with long ones — humans do, LLMs stay in a comfortable middle band. Function word frequency (the, of, and, but) builds something close to a stylistic fingerprint; it's stable across a person's writing and very hard to fake. Punctuation entropy measures how varied and idiosyncratic the punctuation is — humans reach for em-dashes and semicolons in personal patterns, LLMs default to commas. Burstiness captures whether themes and words cluster in waves or spread evenly; human writing is bursty, LLM output is flat. Readability score (Flesch-Kincaid) tends to be consistently high for LLMs because they're optimised to be clear. Transition phrase frequency counts how often "Furthermore," "It is worth noting," and "In conclusion" appear — LLMs lean on these heavily. Average sentence length rounds it out as a basic baseline.
What the output looks like
Signal 2 returns a continuous float between 0 and 1 — not a binary flag. The score comes from a logistic regression or gradient-boosted tree trained on the feature vector. A convenience label ("human" or "ai_generated") is derived from it by thresholding at 0.5, but the score itself is what gets passed forward, because you need the granularity for fusion. Alongside the score, the full feature dict is returned — those raw values are what get stored in the audit log and surfaced to human appeal reviewers as evidence.
How the two scores combine
Three stages. First, a weighted average: Signal 1 gets 60% of the weight, Signal 2 gets 40%. Signal 1 earns the higher weight because it understands meaning — it can catch a text that has human-looking stylometric patterns but was clearly LLM-generated through careful prompting. Signal 2 is the corroborating witness, not the lead.
Second, hard overrides. Certain feature values are diagnostic enough to push the combined score regardless of the average. Very high transition phrase frequency paired with very low sentence length variance pushes the score up to at least 0.70. A very low type-token ratio pushes it to at least 0.65. If Signal 1 is extremely confident (above 0.90), the combined score is pushed to at least 0.85. If both signals are very low, the score is capped at 0.20.
Third, bucket assignment. Below 0.35 is classified as human. Between 0.35 and 0.55 is uncertain. Between 0.55 and 0.75 is ai-assisted. Above 0.75 is ai-generated. The uncertain bucket is deliberate — it's more honest than forcing a binary verdict on ambiguous content, and it's what triggers a softer transparency label to the user rather than a firm one.



* How the two signals are combined


**Uncertainty representation:**
Uncertainty representation
What a score of 0.6 actually means
A combined score of 0.6 does not mean the system is 60% confident the content is AI-generated. That interpretation would imply the score is a calibrated probability, and at v1 it is not — it is a weighted linear combination of two model outputs that have not been passed through any calibration step. What 0.6 means to the system is narrower and more honest: the combined signal output landed in the uncertain band, the two signals did not agree strongly enough in either direction to produce a high-confidence classification, and the case should be routed to human review. Treating the raw score as a probability and displaying it to users or reviewers as one would be a mistake, because it implies a precision the system does not yet have.
Raw signal outputs and what they look like before combination
The perplexity signal produces a score between 0 and 1, where values near 1 indicate low-entropy, AI-like text and values near 0 indicate high-entropy, human-like burstiness. The trained classifier also produces a score between 0 and 1, representing its estimated likelihood that the content is AI-generated. Both of these raw outputs are themselves uncalibrated in the statistical sense — a classifier that outputs 0.7 has not necessarily been verified to be correct 70% of the time on held-out data. Before the weighted average is computed, neither signal should be assumed to be a true probability. They are scores, not probabilities, and the system should treat them accordingly in all internal documentation, logging, and reviewer-facing displays.
Calibration: the gap between scores and probabilities
Calibration is the process of adjusting a model's raw output scores so that a score of 0.7 actually corresponds to the model being correct approximately 70% of the time across a large sample. Uncalibrated classifiers are very common — they tend to be overconfident (clustering outputs near 0 and 1) or underconfident (clustering near 0.5) depending on their architecture and training procedure. For v1, full Platt scaling or isotonic regression calibration is not required, but two things should be done from the start to prepare for it. First, every raw signal score — the perplexity score, the classifier score, and the combined weighted score — should be logged to the audit table on every submission, even when the classification is never appealed. This creates the dataset needed to measure calibration later. Second, the system should never display the raw combined score to users or creators in the transparency label. The label should show the classification (HI-CONF-AI, UNCERTAIN, HI-CONF-HU) and nothing more. Numeric scores are for internal logging and reviewer tooling only, where the audience understands their limitations.
Threshold placement and the logic behind it
The thresholds that convert the combined score into a classification are: 0.80 and above maps to HI-CONF-AI, 0.20 and below maps to HI-CONF-HU, and everything between 0.21 and 0.79 maps to UNCERTAIN. These numbers are not derived from a statistical analysis of the signal distributions — they are a conservative starting point chosen to minimize the cost of early errors. The asymmetry worth noting is that the uncertain band is intentionally wide: 58 percentage points of the 0–1 range route to human review rather than to an automated label. This is the right tradeoff at v1 because the cost of a false positive — wrongly labeling a human creator's work as AI-generated — is significantly higher than the cost of routing a case to human review. As real traffic accumulates and the distribution of scores on genuine human content and genuine AI content becomes visible in the audit log, the thresholds should be revisited empirically. If the score distributions are well-separated — human content clustering near 0.1 and AI content clustering near 0.9 — the uncertain band can be narrowed. If they overlap significantly in the 0.3–0.7 range, the current thresholds are probably already too aggressive and should be widened further.
How the thresholds interact with the appeal flow
The threshold placement has a direct consequence for appeal volume that should be planned for explicitly. A wide uncertain band means a large fraction of submissions will be routed to human review via the UNCERTAIN classification, which creates reviewer load even before any appeals are filed. If early traffic analysis shows that more than roughly 15–20% of submissions are landing in the uncertain band, that is a signal either that the signal weights need tuning or that the thresholds are too conservative — not necessarily that the system is broken. Tracking the proportion of submissions in each band as a live metric from day one will make this visible early enough to act on it before reviewer queue depth becomes a problem.

**Transparency label design:**
* High-confidence AI (HI-CONF-AI, combined score ≥ 0.80)
Text: "This content was likely generated by an AI tool."

* High-confidence human (HI-CONF-HU, combined score ≤ 0.20)
Text: "This content appears to have been written by a person."

* Uncertain (UNCERTAIN, combined score 0.21–0.79)
Text: "The origin of this content could not be determined."


**Appeals workflow:**
Any creator can appeal, but only if their classification is HI-CONF-AI or UNCERTAIN. HI-CONF-HU labels do not open an appeal window.
Reviewer view (in decision order): Creator's appeal statement → per-signal score breakdown → full submission text → creator's appeal history.

**Anticipated edge cases:**
- Non-native English speakers writing in simple, grammatically regular English — short sentences and common vocabulary push the perplexity signal toward AI-like scores even when the content is clearly personal and human. Highest-risk failure mode for discriminatory outcomes.
- Intentionally stylized human writing — minimalist prose, list-based essays, templated journalism (match recaps, earnings reports). Structurally similar to AI output by design; both signals will score them poorly.
- (Future) AI-assisted writing — human draft polished by AI. Neither label is accurate. Reserve a HYBRID value in the classification field schema now to avoid a migration later.

## Architecture

### Flow 1 — submission  (POST /submit)


    POST /submit
    │  raw text
    ▼
    Signal 1 — Groq (LLM semantic classifier)
    │  signal 1 score + rationale
    ▼
    Signal 2 — Stylometric heuristics classifier
    │  signal 2 score + feature vector
    ▼
    Confidence scoring (weighted fusion, threshold)
    │  combined score + category
    ▼
    Transparency label (human-readable verdict)
    │  label text + score + content_id
    ▼
    Audit log (immutable event record)
    │  log confirmation
    ▼
    Response → { label, score, appeal_url }

Submission flow: When a creator posts content, it passes through two classifiers in sequence — first Groq's LLM produces a zero-shot label and confidence score, then a trained binary classifier adds a probability score. A confidence scorer blends the two signals into a combined verdict, which a label generator turns into human-readable text (e.g. "Likely AI-generated, 87% confidence"). That label, along with the full scoring breakdown, is written to an append-only audit log and returned to the client in a single response.

### Flow 2 — appeal  (POST /appeal)

    POST /appeal
    │  content_id + creator statement
    ▼
    Status update (mark content as pending_review)
    │  appeal record + timestamp
    ▼
    Audit log (immutable event record)
    │  log confirmation
    ▼
    Response → { appeal_id, status, eta }

Appeal flow: When a creator disputes their classification, the appeal handler validates the content ID and checks eligibility, then hands a status update to a status updater that marks the record as "under review" and queues it for human evaluation — no classifiers are re-run at this stage. The appeal event is written to the audit log and the client receives an appeal ID plus an estimated review timeline, keeping the response fast while the slower human review happens out-of-band.

