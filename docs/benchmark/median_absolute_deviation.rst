MedianAbsoluteDeviation on Twitter volume CRM
---------------------------------------------

Commit ``72b8ef3`` changed the denominator from ``b * MAD + 1e-10`` to ``b * MAD`` when MAD is positive. This follows the documented score formula. The fallback of ``1e-10`` remains when MAD is zero.

On ``nab/realTweets/Twitter_volume_CRM.csv``, the published earlier AUROC is 0.5542. A full-stream comparison in one environment gave 0.5534 with the earlier denominator and 0.5396 with the current denominator. The earlier constant changed scores by different amounts as the window MAD changed. This separated some scores that the exact formula and benchmark rounding treat as ties, which changed their contribution to AUROC. The lower score is an intended consequence of the formula correction, not a reason to restore the constant.

The short CI run uses only the first 2,000 points. These points do not contain both classes after warm-up, so their AUROC is undefined.
