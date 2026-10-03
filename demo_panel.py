"""
demo_panel.py -- the on-page "inject a test flight" control (bottom centre of the map).

Kept as a plain string with __TOKENS__ (not an f-string) so the JavaScript braces
stay readable. map_plot.build_static_map() adds it to the same page.
"""

_TEMPLATE = r"""
<style>
  #tracils-demo {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); z-index: 1000;
    width: min(620px, calc(100vw - 600px)); min-width: 420px;
    background: var(--panel); border: 1px solid #3A2F55; border-radius: 14px;
    padding: 12px 14px 11px; backdrop-filter: blur(10px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    font-family: var(--font-sans); color: var(--text);
  }
  #tracils-demo .dm-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  #tracils-demo .dm-title { font-size:10.5px; letter-spacing:.09em; color:#B98CE0; font-weight:600; margin-right:2px; }
  #tracils-demo select {
    background:#0F141A; color:var(--text); border:1px solid var(--border); border-radius:8px;
    padding:6px 8px; font:12.5px var(--font-sans); outline:none; max-width:230px;
  }
  #tracils-demo select:focus { border-color:#B98CE0; }
  #tracils-demo button {
    font:600 12.5px var(--font-sans); border-radius:8px; padding:7px 13px; cursor:pointer;
    border:1px solid var(--border); background:#161E26; color:var(--text);
  }
  #tracils-demo button:hover { border-color:#5B6B79; }
  #tracils-demo button.primary { background:#B98CE0; border-color:#B98CE0; color:#14101C; }
  #tracils-demo button.primary:hover { background:#C9A2EA; }
  #tracils-demo button:disabled { opacity:.5; cursor:default; }
  #tracils-demo .dm-blurb { font-size:11.5px; color:var(--text-muted); margin-top:8px; line-height:1.4; }
  #tracils-demo .dm-live { margin-top:9px; padding-top:9px; border-top:1px solid var(--border); font-size:12px; }
  #tracils-demo .dm-row { display:flex; align-items:center; gap:8px; padding:2px 0; flex-wrap:wrap; }
  #tracils-demo .dm-chip { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
  #tracils-demo .dm-cs { font-weight:600; }
  #tracils-demo .dm-mono { font-family:var(--font-mono); color:#B8C6D1; font-size:11.5px; }
  #tracils-demo .dm-muted { color:var(--text-muted); }
  #tracils-demo .dm-note { font-size:10.5px; color:#5E6D7A; margin-top:6px; }
</style>

<div id="tracils-demo">
  <div class="dm-head">
    <span class="dm-title">&#129514; DEMO &middot; INJECT TEST FLIGHT</span>
    <select id="dm-scn" aria-label="Scenario"></select>
    <select id="dm-rwy" aria-label="Runway">
      <option value="RWY 32">Runway 32</option>
      <option value="RWY 14">Runway 14</option>
    </select>
    <button class="primary" id="dm-go">Inject flight</button>
    <button id="dm-clear">Clear</button>
  </div>
  <div class="dm-blurb" id="dm-blurb"></div>
  <div class="dm-live" id="dm-live" style="display:none;"></div>
  <div class="dm-note">Simulated flights run through the same ILS, ML and sequencing checks as real traffic, on a ~6x faster clock.
    They are never saved to the landing log or used for ML training.</div>
</div>

<script>
window.addEventListener('load', function () {
  var map = __MAP__;
  var scn = document.getElementById('dm-scn');
  var rwy = document.getElementById('dm-rwy');
  var go = document.getElementById('dm-go');
  var clr = document.getElementById('dm-clear');
  var blurb = document.getElementById('dm-blurb');
  var live = document.getElementById('dm-live');
  var scenarios = {};

  function esc(t) {
    return String(t).replace(/[&<>"']/g, function (c) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
    });
  }

  fetch('/api/scenarios').then(function (r) { return r.json(); }).then(function (d) {
    scenarios = d;
    Object.keys(d).forEach(function (k) {
      var o = document.createElement('option');
      o.value = k; o.textContent = d[k].label;
      scn.appendChild(o);
    });
    scn.value = 'drift';
    showBlurb();
  });

  function showBlurb() {
    var s = scenarios[scn.value];
    blurb.textContent = s ? s.blurb : '';
  }
  scn.addEventListener('change', showBlurb);

  go.addEventListener('click', function () {
    go.disabled = true;
    fetch('/api/inject', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({scenario: scn.value, runway: rwy.value})
    }).then(function (r) { return r.json(); }).then(function (d) {
      go.disabled = false;
      if (!d.ok) { alert('Could not inject: ' + d.error); return; }
      // bring the new flight(s) and the airport into view
      var pts = [[__LAT__, __LON__]];
      d.flights.forEach(function (f) { pts.push([f.lat, f.lon]); });
      map.flyToBounds(pts, {maxZoom: 11, padding: [90, 90], duration: 1.2});
      poll();
    }).catch(function () { go.disabled = false; alert('Server not reachable'); });
  });

  clr.addEventListener('click', function () {
    fetch('/api/clear', {method: 'POST'}).then(function () { lastResult = null; poll(); });
  });

  var lastResult = null;   // most recent simulated landing, kept on screen after touchdown

  function render(data) {
    var demo = (data.demo && data.demo.active) || [];
    var byCs = {};
    (data.aircraft || []).forEach(function (a) { byCs[a.callsign] = a; });

    // remember the newest simulated landing so the verdict stays visible
    var simLanded = (data.landed || []).filter(function (e) { return e.synthetic; });
    if (simLanded.length) lastResult = simLanded[0];
    else if (!demo.length) lastResult = null;

    var html = '';
    demo.forEach(function (f) {
      var a = byCs[f.callsign];
      var atc = a && a.atc;
      var col = atc ? atc.color : '#5B9BD9';
      var state = 'Not yet on final';
      if (atc) {
        state = atc.status === 'stable' ? 'STABLE' : (atc.status === 'watch' ? 'WATCH' : 'ACT NOW');
        if (atc.headline) state += ' &middot; ' + esc(atc.headline);
      }
      var ml = a && a.ml;
      var mlTxt = '';
      if (ml) {
        if (ml.verdict === 'COLLECTING') mlTxt = 'ML collecting ' + ml.samples + '/' + ml.needed + ' fixes';
        else if (ml.verdict === 'ABNORMAL') mlTxt = 'ML: ' + (ml.baseline === 'sim' ? 'unusual vs simulated baseline' : 'ABNORMAL');
        else if (ml.verdict === 'NORMAL') mlTxt = 'ML: normal';
      }
      html += '<div class="dm-row"><span class="dm-chip" style="background:' + col + ';"></span>' +
        '<span class="dm-cs">' + esc(f.callsign) + '</span>' +
        '<span class="dm-mono">' + f.distance_nm + ' NM out</span>' +
        '<span>' + state + '</span>' +
        (mlTxt ? '<span class="dm-muted">&middot; ' + mlTxt + '</span>' : '') + '</div>';
    });

    if (!demo.length && lastResult) {
      var ev = lastResult, m = ev.ml;
      var verdict = !m ? 'no ML result' : (m.verdict === 'NOT_SCORED' ? 'not scored (' + esc(m.note || 'too few fixes') + ')' :
        (m.verdict === 'ABNORMAL' ? (m.baseline === 'sim' ? 'ML: UNUSUAL vs simulated baseline' : 'ML: ABNORMAL') : 'ML: NORMAL'));
      html += '<div class="dm-row"><span class="dm-chip" style="background:#B98CE0;"></span>' +
        '<span class="dm-cs">' + esc(ev.callsign) + '</span><span>landed on ' + esc(ev.runway) + '</span>' +
        '<span class="dm-muted">&middot; ' + verdict + ' &middot; click the purple marker for the full report</span></div>';
    }
    live.style.display = html ? 'block' : 'none';
    live.innerHTML = html;
  }

  function poll() {
    fetch('/api/data').then(function (r) { return r.json(); }).then(render).catch(function () {});
  }
  poll();
  setInterval(poll, 1500);
});
</script>
"""


def render(map_var, trv_lat, trv_lon):
    return (_TEMPLATE.replace("__MAP__", map_var)
                     .replace("__LAT__", str(trv_lat))
                     .replace("__LON__", str(trv_lon)))
