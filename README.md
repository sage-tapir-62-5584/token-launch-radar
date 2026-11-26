# token-launch-radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/sage-tapir-62-5584/token-launch-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/sage-tapir-62-5584/token-launch-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

First Light is a token launch radar for make.fun's "Build the best token launch radar" bounty. It ranks 200 pre-token DeFi protocols by implied value, fitted from the snapshot's own figures: a ridge regression of log market cap on TVL, age, 7-day growth and chain count, trained on the protocols whose token already trades, multiplied by launch odds estimated from the peers' record (the share of comparable protocols that already record a token). Every row carries a concrete six-step farm plan with costs and durations, and the dashboard recomputes every figure live in the browser.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input ../input/snapshot.json
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=75k23xv8&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
