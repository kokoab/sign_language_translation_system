# Training-only diagnosis of blank collapse

No held-out labels or predictions were used for these measurements.

1. On the previous final joint checkpoint, pooled isolated/core CE has zero gradient
   route to the CTC head. Positive single-token CTC yields blank-bias gradient+0.05463;
   background CE yields-0.01799. Gradient descent moves these in opposite directions.
2. The full CTC train population contains1,291known and6,089OTHER targets over236,988
   frame tokens. Direct isolated/core supervision was absent from that output.
3. CTC likelihood and greedy-path recognition are different. A controlled32-frame,
   two-class example with constant per-frame known probability0.03 yields CTC loss
   0.95464 while every frame's argmax is blank. Raising known probability uniformly to
   0.10 worsens loss to1.98916; p=0.50 gives15.91161; p=0.99 gives0.30131. This exhibits
   a blank-dominated local basin in the simplified stationary-frame objective.
4. A measured16-clip **training-only** probe of repair epoch3 has mean blank probability
   0.9660 and correct-label probability0.02745. Correct sequence marginal likelihood
   exceeds empty in8/16, but greedy returns empty16/16. Positive supervision must learn
   localized, decisive nonblank evidence; a reduced marginal loss alone does not prove it.

The fixed first repair adds direct single-token CTC but retains greedy decoding and
the prior sequence recipe, so its full run can test whether this suffices. Epoch4 has
zero known connected training matches and only9/1,901isolated training exact results.
Do not tune a decoder from held-out errors or treat this early loss decrease as a fix.

Evidence: `gradient_diagnosis.json`, `constant_frame_ctc_diagnostic.json`,
`training_probability_diagnostic.json`, and `training_summary.json`.
