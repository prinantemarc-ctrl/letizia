/**
 * Letizia — ta guide corse (widget chat streaming).
 */
(function () {
  var API = window.VISIT_CORSA_CHAT_API || '';
  if (!API) console.warn('[visit-corsica-chat] Définissez window.VISIT_CORSA_CHAT_API');

  var N = 'vc-chat-';
  var blue = '#356eb5';
  var blueHi = '#2b5a96';
  var dark = '#111';
  var base = (API || '').replace(/\/$/, '');
  var avatarUrl = base + '/static/widget/letizia.png';
  var logoUrl = base + '/static/widget/logo.png';

  /* ── Markdown léger ────────────────────────────────────────────── */
  function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  function md(raw) {
    var t = esc(raw)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
      .replace(/^[-] (.+)/gm, '<li>$1</li>')
      .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>')
      .replace(/<\/ul>\s*<ul>/g, '')
      .replace(/\n{2,}/g, '</p><p>')
      .replace(/\n/g, '<br>');
    return '<p>' + t + '</p>';
  }
  function mdLive(raw) {
    var t = esc(raw)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*\*([^*]*$)/, '<strong>$1</strong>')
      .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
      .replace(/^[-] (.+)/gm, '<li>$1</li>')
      .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>')
      .replace(/<\/ul>\s*<ul>/g, '')
      .replace(/\n{2,}/g, '</p><p>')
      .replace(/\n/g, '<br>');
    return '<p>' + t + '</p>';
  }

  /* ── CSS ────────────────────────────────────────────────────────── */
  var css = [
    /* FAB — avatar rond */
    '#' + N + 'fab{position:fixed;right:16px;bottom:16px;z-index:99999;width:60px;height:60px;border-radius:50%;border:3px solid ' + blue + ';background:url(' + avatarUrl + ') center/cover;cursor:pointer;box-shadow:0 6px 22px rgba(53,110,181,.4);transition:transform .15s,box-shadow .15s}',
    '#' + N + 'fab:hover{transform:scale(1.06);box-shadow:0 8px 28px rgba(53,110,181,.5)}',

    /* Panel — desktop */
    '#' + N + 'panel{position:fixed;right:16px;bottom:92px;z-index:99999;width:min(420px,92vw);height:min(600px,76vh);max-height:76vh;background:#fff;border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.15),0 0 0 1px rgba(0,0,0,.06);display:none;flex-direction:column;overflow:hidden;font-family:"Segoe UI",system-ui,-apple-system,sans-serif}',
    '#' + N + 'panel.open{display:flex}',

    /* Panel — mobile plein écran */
    '@media(max-width:480px){' +
      '#' + N + 'panel{right:0;bottom:0;left:0;top:0;width:100%;height:100%;max-height:100%;border-radius:0;box-shadow:none}' +
      '#' + N + 'fab{right:12px;bottom:12px;width:54px;height:54px}' +
    '}',

    /* Header */
    '#' + N + 'head{padding:14px 16px;background:' + dark + ';color:#fff;display:flex;justify-content:space-between;align-items:center;flex-shrink:0}',
    '#' + N + 'head-left{display:flex;align-items:center;gap:10px}',
    '#' + N + 'avatar{width:36px;height:36px;border-radius:50%;object-fit:cover;border:2px solid rgba(255,255,255,.25)}',
    '#' + N + 'head-info{display:flex;flex-direction:column}',
    '#' + N + 'name{font-weight:700;font-size:.95rem;line-height:1.2}',
    '#' + N + 'status{font-size:.72rem;color:rgba(255,255,255,.55);font-weight:400}',
    '#' + N + 'close{background:none;border:none;color:rgba(255,255,255,.5);font-size:1.3rem;cursor:pointer;padding:4px 6px;border-radius:6px;line-height:1;transition:color .15s,background .15s}',
    '#' + N + 'close:hover{color:#fff;background:rgba(255,255,255,.1)}',

    /* Messages */
    '#' + N + 'msgs{flex:1;overflow-y:auto;padding:18px 14px 14px;background:#fafafa}',

    /* Dots */
    '@keyframes vcPulse{0%,80%,100%{opacity:.25}40%{opacity:1}}',
    '.' + N + 'dots{display:flex;gap:5px;padding:4px 0;align-items:center;white-space:nowrap}',
    '.' + N + 'dot{width:6px;height:6px;border-radius:50%;background:#999;animation:vcPulse 1.2s infinite;flex-shrink:0}',
    '.' + N + 'dot:nth-child(2){animation-delay:.15s}',
    '.' + N + 'dot:nth-child(3){animation-delay:.3s}',
    '.' + N + 'phase{font-size:.78rem;color:#999;margin-left:6px;font-style:italic;white-space:nowrap}',

    /* Bot bubble with avatar */
    '.' + N + 'bot-row{display:flex;gap:8px;align-items:flex-end;margin:8px 0}',
    '.' + N + 'bot-av{width:28px;height:28px;border-radius:50%;object-fit:cover;flex-shrink:0}',

    /* Bubbles */
    '.' + N + 'bubble{max-width:85%;padding:12px 16px;border-radius:16px;font-size:.9rem;line-height:1.6;word-break:break-word}',
    '.' + N + 'bubble.user{margin:8px 0 8px auto;background:' + blue + ';color:#fff;border-bottom-right-radius:4px}',
    '.' + N + 'bubble.bot{background:#fff;color:#1a1a1a;border:1px solid #e8e8e8;border-bottom-left-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.04)}',
    '.' + N + 'bubble.bot p{margin:0 0 .55em}',
    '.' + N + 'bubble.bot p:last-child{margin-bottom:0}',
    '.' + N + 'bubble.bot strong{font-weight:700;color:' + dark + '}',
    '.' + N + 'bubble.bot em{font-style:italic;color:#555}',
    '.' + N + 'bubble.bot ul{margin:.4em 0 .55em;padding-left:1.2em;list-style:disc}',
    '.' + N + 'bubble.bot li{margin-bottom:.3em}',

    /* Form */
    '#' + N + 'form{display:flex;gap:10px;padding:12px 14px;border-top:1px solid #eee;background:#fff;align-items:center;flex-shrink:0}',
    '@media(max-width:480px){#' + N + 'form{padding:10px 12px;padding-bottom:max(10px,env(safe-area-inset-bottom))}}',
    '#' + N + 'input{flex:1;border:1.5px solid #ddd;border-radius:12px;padding:11px 14px;font-size:16px;transition:border-color .15s,box-shadow .15s;background:#fafafa}',
    '#' + N + 'input:focus{outline:none;border-color:' + blue + ';box-shadow:0 0 0 3px rgba(53,110,181,.12);background:#fff}',
    '#' + N + 'send{border:none;border-radius:12px;padding:11px 18px;background:' + blue + ';color:#fff;font-weight:700;font-size:.9rem;cursor:pointer;transition:background .15s}',
    '#' + N + 'send:hover{background:' + blueHi + '}',
    '#' + N + 'send:disabled{opacity:.45;cursor:not-allowed}'
  ].join('\n');

  var s = document.createElement('style');
  s.textContent = css;
  document.head.appendChild(s);

  /* ── DOM ────────────────────────────────────────────────────────── */
  var fab = document.createElement('button');
  fab.id = N + 'fab';
  fab.type = 'button';
  fab.setAttribute('aria-label', 'Parler à Letizia');

  var panel = document.createElement('div');
  panel.id = N + 'panel';
  panel.setAttribute('role', 'dialog');
  panel.innerHTML =
    '<div id="' + N + 'head">' +
      '<div id="' + N + 'head-left">' +
        '<img id="' + N + 'avatar" src="' + avatarUrl + '" alt="Letizia">' +
        '<div id="' + N + 'head-info">' +
          '<span id="' + N + 'name">Letizia</span>' +
          '<span id="' + N + 'status">Ta guide corse</span>' +
        '</div>' +
      '</div>' +
      '<button type="button" id="' + N + 'close" aria-label="Fermer">&times;</button>' +
    '</div>' +
    '<div id="' + N + 'msgs"></div>' +
    '<form id="' + N + 'form">' +
      '<input id="' + N + 'input" type="text" autocomplete="off" placeholder="Dumanda\u2026 pose-moi ta question !">' +
      '<button id="' + N + 'send" type="submit">Envoyer</button>' +
    '</form>';

  document.body.appendChild(fab);
  document.body.appendChild(panel);

  var msgs = document.getElementById(N + 'msgs');
  var input = document.getElementById(N + 'input');
  var form = document.getElementById(N + 'form');
  var btn = document.getElementById(N + 'send');

  /* ── Helpers ────────────────────────────────────────────────────── */
  function addUserBubble(text) {
    var d = document.createElement('div');
    d.className = N + 'bubble user';
    d.textContent = text;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function addBotBubble(html) {
    var row = document.createElement('div');
    row.className = N + 'bot-row';
    var av = document.createElement('img');
    av.className = N + 'bot-av';
    av.src = avatarUrl;
    av.alt = 'Letizia';
    var bub = document.createElement('div');
    bub.className = N + 'bubble bot';
    bub.innerHTML = html;
    row.appendChild(av);
    row.appendChild(bub);
    msgs.appendChild(row);
    msgs.scrollTop = msgs.scrollHeight;
    return bub;
  }

  function dotsHtml(label) {
    return '<div class="' + N + 'dots">' +
      '<span class="' + N + 'dot"></span>' +
      '<span class="' + N + 'dot"></span>' +
      '<span class="' + N + 'dot"></span>' +
      (label ? '<span class="' + N + 'phase">' + label + '</span>' : '') +
      '</div>';
  }

  function lock(v) { btn.disabled = v; input.disabled = v; }
  function scrollDown() { msgs.scrollTop = msgs.scrollHeight; }

  /* ── Events ─────────────────────────────────────────────────────── */
  fab.addEventListener('click', function () {
    panel.classList.toggle('open');
    if (panel.classList.contains('open')) input.focus();
  });
  document.getElementById(N + 'close').addEventListener('click', function () {
    panel.classList.remove('open');
  });

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var q = (input.value || '').trim();
    if (!q || !API) return;
    addUserBubble(q);
    input.value = '';
    lock(true);
    var bub = addBotBubble(dotsHtml("Letizia t'h\u00E0 da risponde..."));
    var rawText = '';
    var streaming = false;

    fetch(base + '/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: q })
    })
      .then(function (res) {
        if (!res.ok) {
          return res.text().then(function (t) {
            try { var j = JSON.parse(t); throw new Error(j.detail || res.statusText); }
            catch (_) { throw new Error(t || res.statusText); }
          });
        }
        var reader = res.body.getReader();
        var decoder = new TextDecoder();
        var buf = '';

        function pump() {
          return reader.read().then(function (result) {
            if (result.done) return;
            buf += decoder.decode(result.value, { stream: true });
            var lines = buf.split('\n');
            buf = lines.pop() || '';
            for (var i = 0; i < lines.length; i++) {
              var line = lines[i];
              if (line.indexOf('data: ') !== 0) continue;
              var p;
              try { p = JSON.parse(line.slice(6)); } catch (_) { continue; }
              if (p.ping && !streaming) continue;
              if (p.t) {
                if (!streaming) { bub.innerHTML = ''; streaming = true; }
                rawText += p.t;
                bub.innerHTML = mdLive(rawText);
                scrollDown();
              }
              if (p.done && p.full) { bub.innerHTML = md(p.full); scrollDown(); }
              if (p.error) { bub.innerHTML = '<em>Erreur : ' + p.error.replace(/</g,'&lt;') + '</em>'; }
            }
            return pump();
          });
        }
        return pump();
      })
      .catch(function (err) {
        bub.innerHTML = '<em>Erreur : ' + (err.message || String(err)).replace(/</g,'&lt;') + '</em>';
      })
      .finally(function () { lock(false); input.focus(); });
  });
})();
