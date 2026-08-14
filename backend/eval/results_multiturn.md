# DocuMind — Multi-turn Retrieval Eval

10 conversations · top-k = 5 · full hybrid+rerank pipeline.
Raw = retrieving on the literal follow-up; Condensed = retrieving on the
standalone query produced by the follow-up rewriting step.

| Query used | Hit@5 | MRR |
|------------|------:|----:|
| Raw follow-up | 80.0% | 0.700 |
| Condensed (DocuMind) | 90.0% | 0.900 |

**Lift from condensing:** Hit@5 +10.0 pts, MRR +0.200.

## Example rewrites

| Follow-up | Condensed query |
|-----------|-----------------|
| How do I fix it? | How do I fix error code E-4021 on the Atlas X200? |
| What's its part number? | Atlas X200 air filter part number |
| And what if they don't resolve the ticket? | What happens if Priority support doesn't resolve the ticket? |
| What voids it? | What voids the standard warranty for the Atlas X200? |
| What about the SFP+ ones? | How many SFP+ ports does the Atlas X200 have? |
| And after that window? | What is AcmeNet's policy on hardware returns after the 30‑day window? |
| How often should I update it? | How often should I update the firmware on the Atlas X200? |
| What's the default admin username after that? | default admin username for Atlas X200 after factory reset |
| How often should I inspect them to avoid that? | How often should I inspect fans to avoid fan failure? |
| How does that compare to the Priority tier's response time? | How does Mission Critical support's 1 hour response time compare to Priority tier response time? |
