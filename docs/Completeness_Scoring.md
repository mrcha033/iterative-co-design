# Completeness Scoring

The completeness score is calculated as the weighted average of metric coverage:

```
score = 0.4 * latency + 0.3 * quality + 0.3 * energy
```

Where each component is measured as the fraction of runs meeting the configured gate.
