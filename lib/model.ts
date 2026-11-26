// First Light — the token launch radar's model.
//
// The board ranks pre-token protocols by an implied value that is fitted, not asserted:
// a ridge-regularised regression of log market cap on the snapshot's own figures, trained on the
// protocols whose token already trades, multiplied by launch odds estimated from the peers' record.
// Nothing here names a protocol: the same fitted function scores every row.
//
//   fitted log mcap = b0 + b1*log(TVL) + b2*log(age_days) + b3*growth + b4*log(chains)
//                     + category effects (Dexs, Lending, Liquid Staking, Yield, Derivatives,
//                       RWA, CDP, Bridge; every other category takes 0)
//   growth is the 7-day TVL change in per cent, clamped to [-30, 60]; age unknown counts as 365 days.
//   odds = the smoothed share of comparable protocols that already record a token, blending the
//          protocol's age bucket and its category (each shrunk toward the global share, alpha = 10).
//   implied_value_usd = round(exp(fitted log mcap) * odds), at least $1.
//
// The fit is done at run time, inside this file, from the snapshot main.ts was handed: the
// coefficients, the leave-one-out test and the walk-forward test (train on protocols listed before
// 2024-06-01, test on the ones listed after) are all computed here and carried into data.js, where
// the dashboard shows them and recomputes every row's figures from them.

/** One protocol of the sealed snapshot: exactly the ten fields the input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole sealed snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The regression's feature list, in coefficient order. */
export const FEATURES = [
  'intercept',
  'log_tvl',
  'log_age_days',
  'growth_pct_over_100',
  'log_chain_count',
  'cat_Dexs',
  'cat_Lending',
  'cat_Liquid_Staking',
  'cat_Yield',
  'cat_Derivatives',
  'cat_RWA',
  'cat_CDP',
  'cat_Bridge',
] as const;

/** Categories that get their own fitted effect; every other category takes 0. */
const EFFECT_CATEGORIES = ['Dexs', 'Lending', 'Liquid Staking', 'Yield', 'Derivatives', 'RWA', 'CDP', 'Bridge'];

/** Ridge penalty; the intercept is never penalised. */
export const RIDGE_LAMBDA = 5;

/** Growth is clamped to this window before it enters the regression (per cent). */
export const GROWTH_MIN = -30;
export const GROWTH_MAX = 60;

/** Age in days assumed when DefiLlama reports no listed_at date. */
export const UNKNOWN_AGE_DAYS = 365;

/** Age buckets used for the launch-odds estimate, in days. */
export const AGE_BUCKETS: Array<[number, number]> = [
  [0, 180],
  [180, 365],
  [365, 730],
  [730, 1460],
  [1460, Infinity],
];

/** Shrinkage strength for the odds estimate (pseudo-counts toward the global share). */
export const ODDS_ALPHA = 10;

/** Walk-forward split: train on protocols listed before this day, test on the rest. */
export const WALK_FORWARD_CUT = '2024-06-01';

/** How many pre-token protocols below the watchlist floor the board carries, by TVL. */
export const EARLY_ROWS = 160;

/** How many already-trading protocols the dashboard's Live view carries, by market cap. */
export const LIVE_ROWS = 24;

/** The model's version label, shown on the dashboard. */
export const MODEL_VERSION = 'first-light-1';

const DAY_MS = 86_400_000;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** Whole days from `listedAt` to `asOf`, at least 0; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** A protocol records no token when all three token fields are null. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** A name ending in a version marker is a pool of another protocol, not a launch of its own. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** Whether a protocol is one the board may rank: pre-token, not an exchange, not a versioned pool. */
export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && p.category !== 'CEX' && !VERSIONED.test(p.name);
}

/** The regression's feature vector for one protocol. */
export function features(p: SnapshotProtocol, asOf: string): number[] {
  const age = ageDays(asOf, p.listed_at) ?? UNKNOWN_AGE_DAYS;
  const rawGrowth = finite(p.change_7d, 0);
  const growth = Math.min(GROWTH_MAX, Math.max(GROWTH_MIN, rawGrowth)) / 100;
  const row: number[] = [1, Math.log(Math.max(p.tvl_usd, 1)), Math.log(Math.max(age, 1)), growth, Math.log(Math.max(p.chains.length, 1))];
  for (const c of EFFECT_CATEGORIES) row.push(p.category === c ? 1 : 0);
  return row;
}

/** Solve A x = b by Gaussian elimination with partial pivoting (A square, small). */
export function solveLinear(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r]![col]!) > Math.abs(m[pivot]![col]!)) pivot = r;
    const tmp = m[col]!;
    m[col] = m[pivot]!;
    m[pivot] = tmp;
    const diag = m[col]![col]!;
    if (diag === 0) throw new Error('solveLinear: singular matrix');
    for (let r = col + 1; r < n; r++) {
      const factor = m[r]![col]! / diag;
      for (let c = col; c <= n; c++) m[r]![c] = m[r]![c]! - factor * m[col]![c]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = m[i]![n]!;
    for (let j = i + 1; j < n; j++) sum -= m[i]![j]! * x[j]!;
    x[i] = sum / m[i]![i]!;
  }
  return x;
}

/** Ridge-regularised least squares: beta = (X'X + lambda*I)' X'y, intercept unpenalised. */
export function ridgeFit(xs: number[][], ys: number[], lambda: number): number[] {
  const k = xs[0]!.length;
  const xtx: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const xty: number[] = new Array<number>(k).fill(0);
  for (let i = 0; i < xs.length; i++) {
    const row = xs[i]!;
    for (let a = 0; a < k; a++) {
      xty[a]! += row[a]! * ys[i]!;
      for (let b = 0; b < k; b++) xtx[a]![b]! += row[a]! * row[b]!;
    }
  }
  for (let a = 1; a < k; a++) xtx[a]![a]! += lambda;
  return solveLinear(xtx, xty);
}

/** Dot product of two equal-length vectors. */
export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

/** R-squared of predictions against observed values. */
function rSquared(observed: number[], predicted: number[]): number {
  const mean = observed.reduce((s, v) => s + v, 0) / observed.length;
  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < observed.length; i++) {
    ssTot += (observed[i]! - mean) ** 2;
    ssRes += (observed[i]! - predicted[i]!) ** 2;
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

function rmse(errors: number[]): number {
  return Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / errors.length);
}

function meanAbs(errors: number[]): number {
  return errors.reduce((s, e) => s + Math.abs(e), 0) / errors.length;
}

/** The fitted model and the held-out tests that check it, all computed from the snapshot. */
export interface ModelFit {
  coefficients: number[];
  features: string[];
  ridge_lambda: number;
  training_n: number;
  training_r2: number;
  residual_sd_log: number;
  loo: { n: number; rmse_log: number; mae_log: number; within_2x: number };
  walk_forward: { cut: string; train_n: number; test_n: number; rmse_log: number; mae_log: number; within_2x: number };
  odds: { global_share: number; alpha: number; buckets: Array<{ lo: number; hi: number | null; with_token: number; total: number; share: number }> };
}

/**
 * Fit log market cap on the protocols whose token already trades (positive market cap only),
 * then test the fit two ways that never let a protocol grade itself: leave-one-out, and a
 * walk-forward split by the day DefiLlama began tracking each protocol.
 */
export function fitModel(snapshot: Snapshot): ModelFit {
  const train = snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
  const xs = train.map((p) => features(p, snapshot.as_of));
  const ys = train.map((p) => Math.log(p.mcap_usd!));
  const beta = ridgeFit(xs, ys, RIDGE_LAMBDA);
  const predicted = xs.map((x) => dot(beta, x));
  const resid = ys.map((y, i) => y - predicted[i]!);
  const residualSd = Math.sqrt(resid.reduce((s, e) => s + e * e, 0) / Math.max(resid.length - xs[0]!.length, 1));

  // Leave-one-out: refit without each protocol in turn, predict the one left out.
  const looErrors: number[] = [];
  for (let i = 0; i < train.length; i++) {
    const restX = xs.filter((_, j) => j !== i);
    const restY = ys.filter((_, j) => j !== i);
    const b = ridgeFit(restX, restY, RIDGE_LAMBDA);
    looErrors.push(ys[i]! - dot(b, xs[i]!));
  }

  // Walk-forward: train on protocols listed before the cut, test on the ones listed after.
  const wfTrainIdx = train.map((p, i) => (p.listed_at !== null && p.listed_at < WALK_FORWARD_CUT ? i : -1)).filter((i) => i >= 0);
  const wfTestIdx = train.map((p, i) => (p.listed_at !== null && p.listed_at >= WALK_FORWARD_CUT ? i : -1)).filter((i) => i >= 0);
  const wfBeta = ridgeFit(wfTrainIdx.map((i) => xs[i]!), wfTrainIdx.map((i) => ys[i]!), RIDGE_LAMBDA);
  const wfErrors = wfTestIdx.map((i) => ys[i]! - dot(wfBeta, xs[i]!));

  // Launch odds: the share of protocols that already record a token, by age bucket.
  const withAge = snapshot.protocols.filter((p) => ageDays(snapshot.as_of, p.listed_at) !== null);
  const globalShare = withAge.filter((p) => !isPreToken(p)).length / Math.max(withAge.length, 1);
  const buckets = AGE_BUCKETS.map(([lo, hi]) => {
    const inBucket = withAge.filter((p) => {
      const a = ageDays(snapshot.as_of, p.listed_at)!;
      return a >= lo && a < hi;
    });
    const withToken = inBucket.filter((p) => !isPreToken(p)).length;
    return { lo, hi: Number.isFinite(hi) ? hi : null, with_token: withToken, total: inBucket.length, share: withToken / Math.max(inBucket.length, 1) };
  });

  return {
    coefficients: beta,
    features: [...FEATURES],
    ridge_lambda: RIDGE_LAMBDA,
    training_n: train.length,
    training_r2: rSquared(ys, predicted),
    residual_sd_log: residualSd,
    loo: {
      n: looErrors.length,
      rmse_log: rmse(looErrors),
      mae_log: meanAbs(looErrors),
      within_2x: looErrors.filter((e) => Math.abs(e) < Math.log(2)).length / looErrors.length,
    },
    walk_forward: {
      cut: WALK_FORWARD_CUT,
      train_n: wfTrainIdx.length,
      test_n: wfTestIdx.length,
      rmse_log: rmse(wfErrors),
      mae_log: meanAbs(wfErrors),
      within_2x: wfErrors.filter((e) => Math.abs(e) < Math.log(2)).length / Math.max(wfErrors.length, 1),
    },
    odds: { global_share: globalShare, alpha: ODDS_ALPHA, buckets },
  };
}

/** Smoothed share of protocols with a token among a peer group, shrunk toward the global share. */
function smoothedShare(withToken: number, total: number, globalShare: number): number {
  return (withToken + ODDS_ALPHA * globalShare) / (total + ODDS_ALPHA);
}

/**
 * A protocol's launch odds from the peers' record: the smoothed token share of its age bucket
 * blended with the smoothed token share of its category (each shrunk toward the global share).
 * A protocol with no listed date or no category takes the half it has; one with neither takes
 * the global share. The result is a probability, never a prediction of when a token comes.
 */
export function launchOdds(p: SnapshotProtocol, snapshot: Snapshot, fit: ModelFit): { odds: number; age_share: number | null; category_share: number | null } {
  const global = fit.odds.global_share;
  const age = ageDays(snapshot.as_of, p.listed_at);
  let ageShare: number | null = null;
  if (age !== null) {
    const bucket = fit.odds.buckets.find((b) => age >= b.lo && (b.hi === null || age < b.hi));
    if (bucket) ageShare = smoothedShare(bucket.with_token, bucket.total, global);
  }
  let catShare: number | null = null;
  if (p.category !== null) {
    const peers = snapshot.protocols.filter((q) => q.category === p.category);
    if (peers.length > 0) catShare = smoothedShare(peers.filter((q) => !isPreToken(q)).length, peers.length, global);
  }
  const parts = [ageShare, catShare].filter((v): v is number => v !== null);
  const odds = parts.length === 0 ? global : parts.reduce((s, v) => s + v, 0) / parts.length;
  return { odds, age_share: ageShare, category_share: catShare };
}

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
  token_status: 'pre-token';
  on_watchlist: boolean;
  implied_value_usd: number;
  model: {
    fitted_value_usd: number;
    launch_odds: number;
    age_share: number | null;
    category_share: number | null;
    fitted_log_mcap: number;
    growth_pct: number;
  };
  farm_steps: string[];
}

/** A live-token protocol: it already trades, so the board shows the market's value, never a model. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** Everything the dashboard needs: the board, the live set, and the fitted model. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  protocols_total: number;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  model: ModelFit;
  rows: BoardRow[];
  live: LiveRow[];
}

/** The snapshot's protocol by slug. */
function bySlug(protocols: SnapshotProtocol[]): Map<string, SnapshotProtocol> {
  return new Map(protocols.map((p) => [p.slug, p]));
}

/** Score one pre-token protocol: fitted value, launch odds, implied value. */
export function scoreProtocol(p: SnapshotProtocol, snapshot: Snapshot, fit: ModelFit): Omit<BoardRow, 'rank' | 'farm_steps' | 'on_watchlist'> {
  const x = features(p, snapshot.as_of);
  const fittedLog = dot(fit.coefficients, x);
  const fittedValue = Math.max(1, Math.round(Math.exp(fittedLog)));
  const { odds, age_share, category_share } = launchOdds(p, snapshot, fit);
  // Round the odds first, then multiply: the dashboard recomputes the implied value from the
  // stored odds, so the stored implied value must be the product of the stored figures.
  const oddsStored = Math.round(odds * 10000) / 10000;
  const implied = Math.max(1, Math.round(fittedValue * oddsStored));
  const rawGrowth = finite(p.change_7d, 0);
  return {
    slug: p.slug,
    name: p.name,
    category: p.category,
    chains: p.chains,
    tvl_usd: p.tvl_usd,
    change_7d: p.change_7d,
    listed_at: p.listed_at,
    age_days: ageDays(snapshot.as_of, p.listed_at),
    token_status: 'pre-token',
    implied_value_usd: implied,
    model: {
      fitted_value_usd: fittedValue,
      launch_odds: oddsStored,
      age_share: age_share === null ? null : Math.round(age_share * 10000) / 10000,
      category_share: catShareRounded(category_share),
      fitted_log_mcap: Math.round(fittedLog * 10000) / 10000,
      growth_pct: Math.min(GROWTH_MAX, Math.max(GROWTH_MIN, rawGrowth)),
    },
  };
}

function catShareRounded(v: number | null): number | null {
  return v === null ? null : Math.round(v * 10000) / 10000;
}

/** The protocols the board ranks, strongest first, with ranks 1..n. */
export function buildBoard(snapshot: Snapshot, fit: ModelFit): BoardRow[] {
  const index = bySlug(snapshot.protocols);
  const chosen = new Map<string, SnapshotProtocol>();
  // The watchlist is the floor: every slug on it, whatever its size.
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  // The early radar: pre-token protocols below the watchlist floor, by TVL, capped.
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p) && p.tvl_usd < snapshot.floor_usd)
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0))
    .slice(0, EARLY_ROWS);
  for (const p of early) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => ({ p, row: scoreProtocol(p, snapshot, fit) }));
  scored.sort((a, b) => b.row.implied_value_usd - a.row.implied_value_usd || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  return scored.map((s, i) => ({
    ...s.row,
    rank: i + 1,
    on_watchlist: snapshot.watchlist.includes(s.p.slug),
    farm_steps: farmSteps(s.p, s.row.age_days),
  }));
}

// ---------------------------------------------------------------- farm plans

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** The category's core action, in words a farmer can follow, with cost and time. */
function coreAction(p: SnapshotProtocol, capital: number, chain: string): { text: string; cost: string; time: string } {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  const extra = p.chains.length > 1 ? ` It also runs on ${p.chains.slice(1, 3).join(' and ')}, so repeat the position there if the gas is cheap.` : '';
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) {
    return { text: `Supply $${capital} of USDC to ${name} on ${chain} and keep the position open; borrow no more than 30% of it so a price move cannot liquidate you.${extra}`, cost: `$${capital} of capital and about $5 of gas`, time: '15 minutes to set up' };
  }
  if (c.includes('dex') || c.includes('leveraged farm')) {
    return { text: `Provide $${capital} of liquidity to ${name}'s main pool on ${chain} in a narrow range around the current price, and keep the position open.${extra}`, cost: `$${capital} of capital and about $6 of gas`, time: '20 minutes to set up' };
  }
  if (c.includes('yield') || c.includes('allocator') || c.includes('curator')) {
    return { text: `Deposit $${capital} into a ${name} vault on ${chain} and keep the shares; check the vault's strategy page first so you know what it holds.${extra}`, cost: `$${capital} of capital and about $5 of gas`, time: '15 minutes to set up' };
  }
  if (c.includes('bridge')) {
    return { text: `Bridge $${capital} of USDC to ${chain} through ${name}, then use the bridged funds in the destination app instead of bridging straight back.${extra}`, cost: `$${capital} of capital and about $4 of bridge fees`, time: '20 minutes, plus the bridge wait' };
  }
  if (c.includes('stak') || c.includes('restak')) {
    return { text: `Stake $${capital} of the chain's staking asset through ${name} on ${chain} and keep the receipt token in the same wallet.${extra}`, cost: `$${capital} of capital and about $4 of gas`, time: '15 minutes to set up' };
  }
  if (c.includes('perpetual') || c.includes('derivative')) {
    return { text: `Deposit $${capital} of USDC as margin on ${name} on ${chain}, open one small hedged position, and close it the same week.${extra}`, cost: `$${capital} of margin and about $6 of gas`, time: '20 minutes to set up' };
  }
  if (c.includes('prediction')) {
    return { text: `Fund $${capital} on ${name} and place one small position in a market that closes within the week, then let it settle.${extra}`, cost: `$${capital} of capital and about $3 of gas`, time: '10 minutes to set up' };
  }
  if (c.includes('stable')) {
    return { text: `Mint or deposit $${capital} of the stable asset through ${name} on ${chain} and hold it in the wallet you farm with.${extra}`, cost: `$${capital} of capital and about $4 of gas`, time: '15 minutes to set up' };
  }
  if (c.includes('rwa')) {
    return { text: `Mint $${capital} of the protocol's yield-bearing asset on ${chain} through ${name} and hold it; note any lock-up before you buy.${extra}`, cost: `$${capital} of capital and about $5 of gas`, time: '15 minutes to set up' };
  }
  if (c.includes('insurance')) {
    return { text: `Buy $${capital} of cover from ${name} on ${chain} for a protocol you already use, and keep the policy active.${extra}`, cost: `$${capital} of premium and about $4 of gas`, time: '15 minutes to set up' };
  }
  if (c.includes('basis')) {
    return { text: `Deposit $${capital} into ${name}'s basis position on ${chain} and hold it through one funding cycle before judging it.${extra}`, cost: `$${capital} of capital and about $5 of gas`, time: '15 minutes to set up' };
  }
  return { text: `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open; use the protocol's own app, not a third-party link.${extra}`, cost: `$${capital} of capital and about $4 of gas`, time: '15 minutes to set up' };
}

/** The weekly habit, worded for the category. */
function weeklyHabit(p: SnapshotProtocol): string {
  const c = (p.category ?? '').toLowerCase();
  if (c.includes('dex')) return 'Make one qualifying action each week: add or adjust liquidity, or swap through the pool.';
  if (c.includes('lend') || c.includes('cdp')) return 'Make one qualifying action each week: top up the supply, repay a little, or adjust the borrow.';
  if (c.includes('perpetual') || c.includes('derivative')) return 'Make one qualifying action each week: one small trade or a margin top-up, then close it.';
  if (c.includes('bridge')) return 'Make one qualifying action each week: one small bridge in either direction.';
  if (c.includes('prediction')) return 'Make one qualifying action each week: one small position in a fresh market.';
  return 'Make one qualifying action each week: a deposit, a swap, a vote, or a claim inside the protocol.';
}

/**
 * A protocol's farm plan: six steps, each an action with a cost and a time, derived from the
 * protocol's own record (category, chains, TVL, age), never from a hand-written list of names.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null): string[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const action = coreAction(p, capital, chain);
  const steps: Array<{ text: string; cost: string; time: string }> = [
    {
      text: `Use a fresh wallet that holds only the funds you plan to farm with ${p.name}, and keep its seed phrase offline on paper.`,
      cost: 'under $3 of gas for the first transaction',
      time: '5 minutes',
    },
    action,
    {
      text: weeklyHabit(p),
      cost: 'about $2 of gas a week',
      time: '10 minutes a week',
    },
    {
      text: `Hold the position for at least ${hold} days so the activity is counted in more than one snapshot, and do not withdraw and re-enter to chase a snapshot.`,
      cost: `the $${capital} stays in the position`,
      time: `${hold} days`,
    },
    {
      text: `Before you withdraw, open ${p.name}'s own points or rewards page from the link the protocol itself publishes, and claim only there.`,
      cost: 'no fee to check, and never pay anyone to claim for you',
      time: '10 minutes',
    },
    {
      text: `Write the wallet, the dates and every transaction hash in a farm log, so you can repeat the position on ${chain} or unwind it cleanly.`,
      cost: 'nothing but the log',
      time: '5 minutes',
    },
  ];
  return steps.map((s) => {
    let out = `${s.text} Cost: ${s.cost}. Time: ${s.time}.`;
    if (out.length > 600) out = `${out.slice(0, 596)}...`;
    if (out.length < 20) out = `${out} Keep the position open and check the protocol's own page.`;
    return out;
  });
}

/** The live-token protocols, by market cap, for the dashboard's Live view. */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0))
    .slice(0, LIVE_ROWS)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
    }));
}

/** Everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const fit = fitModel(snapshot);
  const board = buildBoard(snapshot, fit);
  const live = liveRows(snapshot);
  const early = board.filter((r) => !snapshot.watchlist.includes(r.slug)).length;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    protocols_total: snapshot.protocols.length,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: early,
    live_size: live.length,
    model: fit,
    rows: board,
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new Error('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new Error('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new Error('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new Error('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new Error('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new Error('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new Error(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new Error(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, one shared value, farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  if (board.length < 100 || board.length > 4000) return `the board holds ${board.length} rows, outside 100..4000`;
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const byRank = [...board].sort((a, b) => a.rank - b.rank);
  for (let i = 0; i < byRank.length; i++) if (byRank[i]!.rank !== i + 1) return `rank ${i + 1} is ${byRank[i]!.rank}`;
  for (let i = 1; i < byRank.length; i++) if (byRank[i]!.implied_value_usd > byRank[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) {
      const t = s.trim();
      if (t.length < 20 || t.length > 600) return `${r.slug} has a farm step of ${t.length} characters`;
    }
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
    if (r.token_status !== 'pre-token') return `${r.slug} is not marked pre-token`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
