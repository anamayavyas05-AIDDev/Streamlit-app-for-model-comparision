**Online Shoppers Purchasing Intention Dataset** — UCI Machine Learning
Repository (CC BY 4.0).

Each row is one browsing session on an e-commerce site. The target `Revenue` is
`True` if that session ended in a purchase.

| | |
|---|---|
| Instances | 12,330 sessions (9,864 train / 2,466 test) |
| Features | 17 — 10 numeric, 6 categorical, 1 boolean |
| Class balance | 84.53% no purchase / 15.47% purchase |
| Missing values | None |

**Feature groups**

- **Behaviour** — page counts and time spent on Administrative, Informational
  and ProductRelated pages.
- **Engagement** — `BounceRates`, `ExitRates`, `PageValues`.
- **Timing** — `Month`, `SpecialDay`, `Weekend`.
- **Visitor** — `OperatingSystems`, `Browser`, `Region`, `TrafficType`,
  `VisitorType`.

`OperatingSystems`, `Browser`, `Region` and `TrafficType` are stored as integers
but are category codes, not quantities, so they are one-hot encoded rather than
scaled.

**Why the metrics matter here.** With 84.53% of sessions ending without a
purchase, a model that always predicts "no purchase" already scores 84.51%
accuracy while being useless. AUC, F1 and especially MCC are the metrics that
expose that, which is why all six are reported side by side — and why a
`DummyClassifier` is included in the dropdown as the floor every real model must
clear.

**Uploaded files** are scored with the exact 17 feature columns the pipeline was
trained on. Include a `Revenue` column to see metrics; without it the app shows
predictions only.
