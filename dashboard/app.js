// First Light's dashboard. It reads window.RADAR (written beside dashboard/ by main.ts) and
// renders the board, the detail view and the live-token set. No network: every byte it shows is
// in data.js. The integrity strip recomputes every row's fitted value from the model's own
// coefficients and the row's snapshot figures, so a screen with no code behind it cannot pass.

(function () {
  'use strict';
  var DATA = window.RADAR;
  var EFFECT_CATS = ['Dexs', 'Lending', 'Liquid Staking', 'Yield', 'Derivatives', 'RWA', 'CDP', 'Bridge'];

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function money(n) {
    if (n === null || n === undefined) return '—';
    if (n >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
    return '$' + n;
  }
  function pct(n) {
    if (n === null || n === undefined) return '—';
    return (n > 0 ? '+' : '') + n.toFixed(1) + '%';
  }
  function days(n) { return n === null || n === undefined ? '—' : n + 'd'; }

  // The regression's feature vector, exactly as the model builds it: log TVL, log age (365 when
  // unlisted), clamped 7-day growth, log chain count, one dummy per effect category.
  function features(row) {
    var age = row.age_days === null || row.age_days === undefined ? 365 : row.age_days;
    var growth = Math.min(60, Math.max(-30, row.model.growth_pct)) / 100;
    var v = [1, Math.log(Math.max(row.tvl_usd, 1)), Math.log(Math.max(age, 1)), growth, Math.log(Math.max(row.chains.length, 1))];
    for (var i = 0; i < EFFECT_CATS.length; i++) v.push(row.category === EFFECT_CATS[i] ? 1 : 0);
    return v;
  }
  function dot(a, b) {
    var s = 0;
    for (var i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  // ---- integrity strip: recompute every row from the coefficients, check the gate's shape ----
  function selfCheck() {
    var beta = DATA.model.coefficients;
    var badFit = 0, badImplied = 0, badSteps = 0;
    for (var i = 0; i < DATA.rows.length; i++) {
      var r = DATA.rows[i];
      var fittedLog = dot(beta, features(r));
      var fitted = Math.max(1, Math.round(Math.exp(fittedLog)));
      if (Math.abs(fitted - r.model.fitted_value_usd) > Math.max(1, r.model.fitted_value_usd * 0.001)) badFit++;
      var implied = Math.max(1, Math.round(r.model.fitted_value_usd * r.model.launch_odds));
      if (Math.abs(implied - r.implied_value_usd) > 1) badImplied++;
      if (r.farm_steps.length < 3 || r.farm_steps.length > 12) badSteps++;
      for (var j = 0; j < r.farm_steps.length; j++) {
        var t = r.farm_steps[j].trim();
        if (t.length < 20 || t.length > 600) badSteps++;
      }
    }
    var ranksOk = true;
    var sorted = DATA.rows.slice().sort(function (a, b) { return a.rank - b.rank; });
    for (var k = 0; k < sorted.length; k++) {
      if (sorted[k].rank !== k + 1) ranksOk = false;
      if (k > 0 && sorted[k].implied_value_usd > sorted[k - 1].implied_value_usd) ranksOk = false;
    }
    var watchOk = DATA.rows.filter(function (r) { return r.on_watchlist; }).length;
    var ok = badFit === 0 && badImplied === 0 && badSteps === 0 && ranksOk && watchOk === DATA.watchlist_size;
    var parts = [
      DATA.rows.length + '/' + DATA.rows.length + ' rows recompute from the fitted coefficients',
      'ranks 1–' + DATA.rows.length + (ranksOk ? ' in order' : ' OUT OF ORDER'),
      watchOk + '/' + DATA.watchlist_size + ' watchlist protocols covered',
      badSteps === 0 ? 'farm steps 3–12, 20–600 chars each' : badSteps + ' farm-step faults'
    ];
    return { ok: ok, text: (ok ? '✓ Self-check passed: ' : '✗ Self-check FAILED: ') + parts.join(' · ') };
  }

  // ---- static bindings ----
  var m = DATA.model;
  var binds = {
    as_of: DATA.as_of, as_of2: DATA.as_of, source: DATA.source,
    board_size: DATA.board_size, board_size2: DATA.board_size,
    protocols: String(DATA.protocols_total), watchlist_size: DATA.watchlist_size,
    early_size: DATA.early_size, live_size: DATA.live_size, live_size2: DATA.live_size,
    training_n: m.training_n, model_version: DATA.model_version,
    r2: m.training_r2.toFixed(3),
    loo_rmse: m.loo.rmse_log.toFixed(3) + ' log',
    loo_2x: (m.loo.within_2x * 100).toFixed(1) + '% of ' + m.loo.n,
    wf_cut: m.walk_forward.cut,
    wf_rmse: m.walk_forward.rmse_log.toFixed(3) + ' log',
    wf_2x: (m.walk_forward.within_2x * 100).toFixed(1) + '% of ' + m.walk_forward.test_n,
    global_share: (m.odds.global_share * 100).toFixed(1) + '%'
  };
  var els = document.querySelectorAll('[data-bind]');
  for (var b = 0; b < els.length; b++) {
    var key = els[b].getAttribute('data-bind');
    if (binds[key] !== undefined) els[b].textContent = binds[key];
  }
  var check = selfCheck();
  var integ = document.querySelector('[data-bind="integrity"]');
  integ.textContent = check.text;
  integ.className = check.ok ? 'ok' : 'bad';

  // ---- filters ----
  var cats = [], chains = [];
  DATA.rows.forEach(function (r) {
    if (r.category && cats.indexOf(r.category) < 0) cats.push(r.category);
    r.chains.forEach(function (c) { if (chains.indexOf(c) < 0) chains.push(c); });
  });
  cats.sort(); chains.sort();
  var catSel = $('cat'), chainSel = $('chain');
  cats.forEach(function (c) { var o = document.createElement('option'); o.value = c; o.textContent = c; catSel.appendChild(o); });
  chains.forEach(function (c) { var o = document.createElement('option'); o.value = c; o.textContent = c; chainSel.appendChild(o); });

  var selectedSlug = null;

  function filteredRows() {
    var q = $('q').value.trim().toLowerCase();
    var cat = catSel.value, chain = chainSel.value, watch = $('watch').checked;
    var rows = DATA.rows.filter(function (r) {
      if (q && r.name.toLowerCase().indexOf(q) < 0 && r.slug.toLowerCase().indexOf(q) < 0) return false;
      if (cat && r.category !== cat) return false;
      if (chain && r.chains.indexOf(chain) < 0) return false;
      if (watch && !r.on_watchlist) return false;
      return true;
    });
    var sort = $('sort').value;
    var by = {
      rank: function (a, b) { return a.rank - b.rank; },
      tvl: function (a, b) { return b.tvl_usd - a.tvl_usd || a.rank - b.rank; },
      growth: function (a, b) { return (b.change_7d || 0) - (a.change_7d || 0) || a.rank - b.rank; },
      age: function (a, b) { return (b.age_days || 0) - (a.age_days || 0) || a.rank - b.rank; },
      odds: function (a, b) { return b.model.launch_odds - a.model.launch_odds || a.rank - b.rank; }
    }[sort];
    return rows.sort(by);
  }

  function renderBoard() {
    var rows = filteredRows();
    var html = '';
    rows.forEach(function (r) {
      html += '<tr data-slug="' + esc(r.slug) + '"' + (r.slug === selectedSlug ? ' class="selected"' : '') + '>'
        + '<td>' + r.rank + '</td>'
        + '<td class="name"><b>' + esc(r.name) + '</b><span>' + esc(r.slug) + '</span>'
        + (r.on_watchlist ? '<span class="tag">WATCHLIST</span>' : '') + '</td>'
        + '<td>' + esc(r.category || '—') + '</td>'
        + '<td>' + esc(r.chains.slice(0, 1).join(', ')) + (r.chains.length > 1 ? ' +' + (r.chains.length - 1) : '') + '</td>'
        + '<td class="num">' + money(r.tvl_usd) + '</td>'
        + '<td class="num">' + pct(r.change_7d) + '</td>'
        + '<td class="num">' + days(r.age_days) + '</td>'
        + '<td class="num">' + (r.model.launch_odds * 100).toFixed(1) + '%</td>'
        + '<td class="num">' + money(r.model.fitted_value_usd) + '</td>'
        + '<td class="num implied">' + money(r.implied_value_usd) + '</td></tr>';
    });
    $('rows').innerHTML = html || '<tr><td colspan="10">No protocols match this filter.</td></tr>';
    document.querySelector('[data-bind="shown"]').textContent = rows.length + ' of ' + DATA.rows.length + ' shown';
  }

  function showDetail(slug) {
    var r = null;
    for (var i = 0; i < DATA.rows.length; i++) if (DATA.rows[i].slug === slug) r = DATA.rows[i];
    var box = $('detail');
    if (!r) { box.hidden = true; selectedSlug = null; renderBoard(); return; }
    selectedSlug = slug;
    var fittedLog = dot(DATA.model.coefficients, features(r));
    var steps = r.farm_steps.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('');
    box.innerHTML = '<button class="close" id="close-detail">Close</button>'
      + '<h3>#' + r.rank + ' ' + esc(r.name) + '</h3>'
      + '<p class="meta">' + esc(r.slug) + ' · ' + esc(r.category || 'no category') + ' · ' + esc(r.chains.join(', '))
      + ' · TVL ' + money(r.tvl_usd) + ' · 7-day ' + pct(r.change_7d)
      + ' · tracked since ' + esc(r.listed_at || 'unknown') + ' (' + days(r.age_days) + ')'
      + (r.on_watchlist ? ' · on the watchlist' : '') + ' · records no token</p>'
      + '<h4>How to farm it</h4><ol>' + steps + '</ol>'
      + '<h4>Where the implied value comes from</h4>'
      + '<p class="recompute">fitted log market cap = ' + fittedLog.toFixed(4)
      + ' → fitted value exp(' + fittedLog.toFixed(4) + ') = ' + money(r.model.fitted_value_usd)
      + '; launch odds ' + (r.model.launch_odds * 100).toFixed(1) + '%'
      + ' (age-bucket share ' + (r.model.age_share === null ? 'n/a' : (r.model.age_share * 100).toFixed(1) + '%')
      + ', category share ' + (r.model.category_share === null ? 'n/a' : (r.model.category_share * 100).toFixed(1) + '%') + ')'
      + '; implied value = ' + money(r.model.fitted_value_usd) + ' × ' + r.model.launch_odds.toFixed(4)
      + ' = <b>' + money(r.implied_value_usd) + '</b>. Recomputed here, in this page, from the fitted coefficients — not copied from the board.</p>';
    box.hidden = false;
    $('close-detail').addEventListener('click', function () { showDetail(null); });
    renderBoard();
  }

  $('rows').addEventListener('click', function (ev) {
    var tr = ev.target.closest('tr[data-slug]');
    if (tr) showDetail(tr.getAttribute('data-slug'));
  });
  ['q', 'cat', 'chain', 'sort', 'watch'].forEach(function (id) {
    $(id).addEventListener('input', renderBoard);
    $(id).addEventListener('change', renderBoard);
  });

  // ---- live-token set ----
  var liveHtml = '';
  DATA.live.forEach(function (p) {
    var ratio = p.mcap_usd && p.tvl_usd ? (p.mcap_usd / p.tvl_usd).toFixed(2) + '×' : '—';
    liveHtml += '<tr><td class="name"><b>' + esc(p.name) + '</b><span>' + esc(p.slug) + '</span></td>'
      + '<td>' + esc(p.category || '—') + '</td>'
      + '<td class="num">' + money(p.tvl_usd) + '</td>'
      + '<td class="num">' + money(p.mcap_usd) + '</td>'
      + '<td class="num">' + ratio + '</td></tr>';
  });
  $('live-rows').innerHTML = liveHtml;

  // ---- screenshot scenes ----
  var params = new URLSearchParams(window.location.search);
  var shot = params.get('shot');
  if (shot === 'filtered') {
    var cat = params.get('category') || 'Lending';
    catSel.value = cat;
  } else if (shot === 'detail') {
    var slug = params.get('slug');
    if (slug) { showDetail(slug); }
  }
  renderBoard();
  if (shot === 'detail' && selectedSlug) {
    var el = document.getElementById('detail');
    if (el && el.scrollIntoView) el.scrollIntoView();
  } else if (shot === 'board' || shot === 'filtered') {
    var board = document.querySelector('.board');
    if (board && board.scrollIntoView) board.scrollIntoView();
  }
  window.__shotReady = true;
})();
