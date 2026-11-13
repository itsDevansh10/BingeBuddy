/**
 * BingeBuddy — single-page client.
 * Uses only the local Flask API. No remote images, fonts or scripts.
 *   GET /api/genres                    -> { genres: [str], total }
 *   GET /api/shows                     -> { results: [show], total, page, page_size }
 *   GET /api/shows/<id>                -> show
 *   GET /api/shows/<id>/recommendations-> { show_id, recommendations: [show], warning }
 * show: id, name, overview, genres ("A, B"), vote_average, vote_count, popularity,
 *       first_air_date, poster_path, number_of_seasons, number_of_episodes, recommendation_text
 * Dataset text is only ever inserted with textContent.
 */
(function () {
  'use strict';

  const PAGE_SIZE = 24;
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const STAR_PATH = 'M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z';

  // Quick views. "top" = rated above 6 with a minimum vote count, best rating first.
  const PRESETS = {
    shows: { sortBy: 'popularity', order: 'desc', minRating: null, minVotes: null },
    top: { sortBy: 'vote_average', order: 'desc', minRating: 6, minVotes: 50 },
  };

  const state = {
    query: '',
    genre: '',
    preset: 'shows',
    minRating: null,
    minVotes: null,
    sortBy: 'popularity',
    order: 'desc',
    page: 1,
    total: 0,
    totalPages: 1,
    listSeq: 0,
    drawerSeq: 0,
    lastFocus: null,
    searchTimer: null,
    featuredId: null,
    firstLoadDone: false,
  };

  const $ = (id) => document.getElementById(id);
  const dom = {
    header: $('site-header'), hero: $('hero'), catalog: $('catalog'), footer: $('site-footer'),
    brand: $('brand'), tabs: document.querySelectorAll('.nav-tab'),
    search: $('search-input'), searchClear: $('search-clear'),
    heroTitle: $('hero-title'), heroMeta: $('hero-meta'), heroRating: $('hero-rating'),
    heroYear: $('hero-year'), heroSeasons: $('hero-seasons'), heroGenres: $('hero-genres'),
    heroOverview: $('hero-overview'), heroArt: $('hero-art'),
    heroDetails: $('hero-details-btn'), heroSimilar: $('hero-similar-btn'),
    catalogTitle: $('catalog-title'), resultsLabel: $('results-label'),
    sortSelect: $('sort-select'), orderBtn: $('order-btn'), orderLabel: $('order-label'),
    orderIcon: $('order-icon'), resetBtn: $('reset-btn'),
    genreChips: $('genre-chips'), activeFilters: $('active-filters'),
    grid: $('shows-grid'), gridError: $('grid-error'), gridErrorMsg: $('grid-error-msg'),
    gridRetry: $('grid-retry'), gridEmpty: $('grid-empty'), gridEmptyReset: $('grid-empty-reset'),
    pager: $('pager'), pageFirst: $('page-first'), pagePrev: $('page-prev'),
    pageNext: $('page-next'), pageLast: $('page-last'), pageIndicator: $('page-indicator'),
    overlay: $('drawer-overlay'), drawer: $('drawer'), drawerClose: $('drawer-close'),
    drawerScroll: $('drawer-scroll'), drawerArt: $('drawer-art'), drawerTitle: $('drawer-title'),
    drawerRating: $('drawer-rating'), drawerVotes: $('drawer-votes'), drawerYear: $('drawer-year'),
    drawerGenres: $('drawer-genres'), drawerOverview: $('drawer-overview'),
    epSeasons: $('ep-seasons'), epEpisodes: $('ep-episodes'), epAvg: $('ep-avg'),
    epStrip: $('ep-strip'), epNote: $('ep-note'),
    recStatus: $('rec-status'), recGrid: $('rec-grid'),
  };

  // ---------------------------------------------------------------- helpers
  function el(tag, props, children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v === undefined || v === null) continue;
        if (k === 'className') node.className = v;
        else if (k === 'text') node.textContent = v;
        else node.setAttribute(k, v);
      }
    }
    if (children) for (const c of children) if (c) node.appendChild(c);
    return node;
  }

  function starIcon() {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    const p = document.createElementNS(SVG_NS, 'path');
    p.setAttribute('d', STAR_PATH);
    svg.appendChild(p);
    return svg;
  }

  function year(show) {
    const d = show && show.first_air_date;
    return typeof d === 'string' && /^\d{4}/.test(d) ? d.slice(0, 4) : '';
  }

  function genreList(show) {
    return (show && show.genres ? String(show.genres) : '')
      .split(',').map((g) => g.trim()).filter(Boolean);
  }

  function hasRating(show) {
    return show.vote_count > 0 && show.vote_average > 0;
  }

  function fillRating(node, show) {
    node.replaceChildren();
    node.classList.remove('muted');
    if (hasRating(show)) {
      node.appendChild(starIcon());
      node.appendChild(document.createTextNode(show.vote_average.toFixed(1)));
    } else {
      node.classList.add('muted');
      node.textContent = 'Not rated';
    }
  }

  function plural(n, word) {
    return `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`;
  }

  function seasonText(show) {
    return show.number_of_seasons > 0 ? plural(show.number_of_seasons, 'season') : '';
  }

  async function api(path) {
    let res;
    try {
      res = await fetch(path, { headers: { Accept: 'application/json' } });
    } catch (e) {
      throw new Error('Cannot reach the BingeBuddy server. Is it still running?');
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON body */ }
    if (!res.ok) throw new Error((data && data.error) || `Server returned HTTP ${res.status}.`);
    return data;
  }

  // ----------------------------------------- generated poster artwork (offline)
  const GENRE_HUES = {
    'Drama': [222, 255], 'Comedy': [34, 8], 'Action & Adventure': [6, 335],
    'Sci-Fi & Fantasy': [266, 214], 'Animation': [172, 205], 'Crime': [350, 285],
    'Documentary': [152, 190], 'Family': [26, 340], 'Kids': [192, 152],
    'Mystery': [258, 300], 'News': [205, 228], 'Reality': [322, 272],
    'Soap': [336, 12], 'Talk': [44, 24], 'War & Politics': [100, 20], 'Western': [30, 14],
  };

  function hashString(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return h;
  }

  function createTile(show) {
    const genres = genreList(show);
    const first = genres[0] || '';
    const pair = GENRE_HUES[first];
    const base = pair ? pair[0] : hashString(first || show.name || '') % 360;
    const nudge = (hashString(show.name || '') % 24) - 12;
    const h1 = (base + nudge + 360) % 360;
    const h2 = pair ? pair[1] : (h1 + 40) % 360;
    const tile = el('div', { className: 'poster-tile', 'aria-hidden': 'true' }, [
      el('span', { className: 'tile-genre', text: first || 'TV series' }),
      el('div', { className: 'tile-title', text: show.name || 'Untitled' }),
      el('span', { className: 'tile-year', text: year(show) }),
    ]);
    tile.style.setProperty('--c1', `hsl(${h1} 62% 36%)`);
    tile.style.setProperty('--c2', `hsl(${h2} 56% 13%)`);
    return tile;
  }

  function createCard(show, onOpen) {
    const art = el('div', { className: 'card-art' }, [createTile(show)]);
    const badge = el('span', { className: 'rating' });
    fillRating(badge, show);
    art.appendChild(badge);

    const subParts = [year(show), seasonText(show)].filter(Boolean);
    const body = el('div', { className: 'card-body' }, [
      el('div', { className: 'card-title', text: show.name || 'Untitled' }),
      el('div', { className: 'card-sub', text: subParts.join(' · ') || 'No release info' }),
    ]);
    const btn = el('button', {
      type: 'button', className: 'show-card',
      'aria-label': `Open details for ${show.name || 'show'}`,
    }, [art, body]);
    btn.addEventListener('click', () => onOpen(show.id, btn));
    return btn;
  }

  // ---------------------------------------------------------------- catalog
  function syncControls() {
    dom.sortSelect.value = state.sortBy;
    dom.orderLabel.textContent = state.order === 'desc' ? 'Descending' : 'Ascending';
    dom.orderIcon.classList.toggle('asc', state.order === 'asc');
    dom.tabs.forEach((t) => t.classList.toggle('active', t.dataset.preset === state.preset));
    dom.genreChips.querySelectorAll('.genre-chip').forEach((c) => {
      c.setAttribute('aria-pressed', String((c.dataset.genre || '') === state.genre));
    });
    dom.searchClear.hidden = !state.query;
    renderActiveFilters();
  }

  function renderActiveFilters() {
    dom.activeFilters.replaceChildren();
    const add = (label, onRemove, aria) => {
      const b = el('button', { type: 'button', 'aria-label': aria, text: '×' });
      b.addEventListener('click', onRemove);
      dom.activeFilters.appendChild(el('span', { className: 'filter-tag' }, [el('span', { text: label }), b]));
    };
    if (state.query) {
      add(`Search: “${state.query}”`, () => { setQuery(''); }, 'Remove search filter');
    }
    if (state.genre) {
      add(`Genre: ${state.genre}`, () => { state.genre = ''; state.page = 1; syncControls(); loadShows(); }, 'Remove genre filter');
    }
    if (state.minRating !== null) {
      add(`Rating above ${state.minRating}`, () => {
        state.minRating = null; state.minVotes = null; state.preset = null; state.page = 1; syncControls(); loadShows();
      }, 'Remove rating filter');
    }
    dom.activeFilters.hidden = !dom.activeFilters.children.length;
  }

  function showSkeleton() {
    dom.gridError.hidden = true;
    dom.gridEmpty.hidden = true;
    dom.grid.hidden = false;
    dom.grid.setAttribute('aria-busy', 'true');
    // Keep the current cards on screen (dimmed) while new results load, so the page
    // height does not collapse and the scroll position stays where the user is.
    if (state.firstLoadDone && dom.grid.children.length) {
      dom.grid.classList.add('is-loading');
      return;
    }
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 12; i++) {
      frag.appendChild(el('div', null, [el('div', { className: 'skeleton' }), el('div', { className: 'skeleton-line' })]));
    }
    dom.grid.replaceChildren(frag);
    dom.pager.hidden = true;
  }

  async function loadShows() {
    const seq = ++state.listSeq;
    showSkeleton();
    const params = new URLSearchParams({
      page: String(state.page), page_size: String(PAGE_SIZE),
      sort_by: state.sortBy, order: state.order,
    });
    if (state.query) params.set('query', state.query);
    if (state.genre) params.set('genre', state.genre);
    if (state.minRating !== null) params.set('min_rating', String(state.minRating));
    if (state.minVotes !== null) params.set('min_votes', String(state.minVotes));

    try {
      const data = await api(`/api/shows?${params}`);
      if (seq !== state.listSeq) return; // a newer request superseded this one
      state.total = data.total || 0;
      state.totalPages = Math.max(1, Math.ceil(state.total / PAGE_SIZE));
      renderShows(data.results || []);
      state.firstLoadDone = true;
    } catch (err) {
      if (seq !== state.listSeq) return;
      dom.grid.replaceChildren();
      dom.grid.classList.remove('is-loading');
      dom.grid.removeAttribute('aria-busy');
      dom.pager.hidden = true;
      dom.gridEmpty.hidden = true;
      dom.gridError.hidden = false;
      dom.gridErrorMsg.textContent = err.message;
      dom.resultsLabel.textContent = '';
    }
  }

  function renderShows(shows) {
    dom.grid.classList.remove('is-loading');
    dom.grid.removeAttribute('aria-busy');
    dom.gridError.hidden = true;

    const topView = state.minRating !== null;
    dom.catalogTitle.textContent = state.query
      ? `Results for “${state.query}”`
      : topView
        ? (state.genre ? `Most voted ${state.genre} shows` : 'Most voted shows')
        : state.genre ? `${state.genre} shows` : 'All TV shows';

    if (!shows.length) {
      dom.grid.replaceChildren();
      dom.gridEmpty.hidden = false;
      dom.pager.hidden = true;
      dom.resultsLabel.textContent = '0 shows found';
      return;
    }
    dom.gridEmpty.hidden = true;

    const frag = document.createDocumentFragment();
    for (const s of shows) frag.appendChild(createCard(s, openDrawer));
    dom.grid.replaceChildren(frag);

    const start = (state.page - 1) * PAGE_SIZE + 1;
    const end = start + shows.length - 1;
    dom.resultsLabel.textContent =
      `Showing ${start.toLocaleString()}–${end.toLocaleString()} of ${state.total.toLocaleString()} shows`;

    dom.pager.hidden = state.totalPages <= 1;
    dom.pageFirst.disabled = dom.pagePrev.disabled = state.page <= 1;
    dom.pageNext.disabled = dom.pageLast.disabled = state.page >= state.totalPages;
    dom.pageIndicator.textContent = `Page ${state.page.toLocaleString()} of ${state.totalPages.toLocaleString()}`;
  }

  // Scroll to the catalog only when it is still well below the top of the screen
  // (for example the user is at the hero). Once there, further typing never scrolls.
  function ensureCatalogVisible() {
    if (dom.catalog.getBoundingClientRect().top > 140) {
      dom.catalog.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function goToPage(n) {
    const target = Math.min(Math.max(1, n), state.totalPages);
    if (target === state.page) return;
    state.page = target;
    loadShows();
    dom.catalog.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function setQuery(q) {
    state.query = q;
    state.page = 1;
    dom.search.value = q;
    syncControls();
    loadShows();
  }

  function applyPreset(name) {
    const p = PRESETS[name];
    state.preset = name;
    state.sortBy = p.sortBy;
    state.order = p.order;
    state.minRating = p.minRating;
    state.minVotes = p.minVotes;
  }

  function resetAll() {
    state.query = '';
    state.genre = '';
    applyPreset('shows');
    state.page = 1;
    dom.search.value = '';
    syncControls();
    loadShows();
  }

  // ------------------------------------------------------------------ init
  async function initGenres() {
    try {
      const data = await api('/api/genres');
      const genres = data.genres || [];
      const mk = (label, value) => {
        const b = el('button', { type: 'button', className: 'genre-chip', 'aria-pressed': 'false', text: label });
        b.dataset.genre = value;
        b.addEventListener('click', () => {
          state.genre = state.genre === value ? '' : value;
          state.page = 1;
          syncControls();
          loadShows();
        });
        return b;
      };
      const frag = document.createDocumentFragment();
      frag.appendChild(mk('All genres', ''));
      for (const g of genres) frag.appendChild(mk(g, g));
      dom.genreChips.replaceChildren(frag);
      syncControls();
    } catch (e) {
      /* genre chips are optional; the catalog still works without them */
    }
  }

  async function initHero() {
    try {
      const data = await api('/api/shows?sort_by=popularity&order=desc&page=1&page_size=24');
      const list = data.results || [];
      const pick =
        list.find((s) => s.overview && s.overview.length > 80 && s.vote_count >= 50 && s.number_of_seasons > 0) ||
        list.find((s) => s.overview) || list[0];
      if (!pick) { dom.heroOverview.textContent = 'No shows are available yet.'; return; }

      state.featuredId = pick.id;
      dom.heroTitle.textContent = pick.name || 'Featured series';
      fillRating(dom.heroRating, pick);
      dom.heroYear.textContent = year(pick);
      dom.heroSeasons.textContent = seasonText(pick);
      dom.heroMeta.hidden = false;
      dom.heroGenres.textContent = genreList(pick).join(' · ');
      dom.heroOverview.textContent = pick.overview || 'No overview is available for this show.';
      dom.heroArt.replaceChildren(createTile(pick));
      dom.heroDetails.disabled = false;
      dom.heroSimilar.disabled = false;
    } catch (e) {
      dom.heroOverview.textContent = e.message;
    }
  }

  // ---------------------------------------------------------------- drawer
  function setDrawerLoading() {
    dom.drawerArt.replaceChildren(el('div', { className: 'skeleton' }));
    dom.drawerTitle.textContent = 'Loading…';
    for (const n of [dom.drawerRating, dom.drawerVotes, dom.drawerYear]) { n.replaceChildren(); }
    dom.drawerGenres.replaceChildren();
    dom.drawerOverview.textContent = '';
    dom.epSeasons.textContent = dom.epEpisodes.textContent = dom.epAvg.textContent = '—';
    dom.epStrip.replaceChildren();
    dom.epNote.textContent = '';
    dom.recGrid.replaceChildren();
    dom.recStatus.textContent = 'Finding similar shows…';
  }

  function openDrawer(id, trigger) {
    const wasOpen = dom.drawer.classList.contains('open');
    if (!wasOpen) state.lastFocus = trigger || document.activeElement;
    const seq = ++state.drawerSeq;
    setDrawerLoading();
    dom.drawerScroll.scrollTop = 0;

    dom.overlay.hidden = false;
    dom.drawer.classList.add('open');
    dom.drawer.setAttribute('aria-hidden', 'false');
    document.body.classList.add('locked');
    for (const n of [dom.header, dom.hero, dom.catalog, dom.footer]) n.inert = true;
    if (!wasOpen) dom.drawer.focus();

    loadDetails(id, seq);
    loadRecommendations(id, seq);
  }

  function closeDrawer() {
    if (!dom.drawer.classList.contains('open')) return;
    state.drawerSeq++;
    dom.drawer.classList.remove('open');
    dom.drawer.setAttribute('aria-hidden', 'true');
    dom.overlay.hidden = true;
    document.body.classList.remove('locked');
    for (const n of [dom.header, dom.hero, dom.catalog, dom.footer]) n.inert = false;
    const back = state.lastFocus;
    state.lastFocus = null;
    if (back && typeof back.focus === 'function' && document.contains(back)) back.focus();
  }

  async function loadDetails(id, seq) {
    try {
      const show = await api(`/api/shows/${id}`);
      if (seq !== state.drawerSeq) return;

      dom.drawerArt.replaceChildren(createTile(show));
      dom.drawerTitle.textContent = show.name || 'Untitled';
      fillRating(dom.drawerRating, show);
      dom.drawerVotes.textContent = show.vote_count > 0 ? plural(show.vote_count, 'vote') : '';
      dom.drawerYear.textContent = year(show);

      dom.drawerGenres.replaceChildren();
      for (const g of genreList(show)) dom.drawerGenres.appendChild(el('span', { className: 'chip', text: g }));

      dom.drawerOverview.textContent =
        show.overview && show.overview.trim() ? show.overview : 'No overview is available for this show.';

      const seasons = show.number_of_seasons || 0;
      const episodes = show.number_of_episodes || 0;
      dom.epSeasons.textContent = seasons ? seasons.toLocaleString() : '—';
      dom.epEpisodes.textContent = episodes ? episodes.toLocaleString() : '—';
      dom.epAvg.textContent = seasons && episodes ? (episodes / seasons).toFixed(1) : '—';

      dom.epStrip.replaceChildren();
      if (seasons > 0 && seasons <= 40) {
        for (let i = 0; i < seasons; i++) dom.epStrip.appendChild(document.createElement('span'));
      }
      dom.epNote.textContent = seasons || episodes
        ? 'Totals come from the dataset. Per-episode titles, air dates and ratings are not part of it, so none are shown.'
        : 'No season or episode counts are recorded for this show.';
    } catch (err) {
      if (seq !== state.drawerSeq) return;
      dom.drawerArt.replaceChildren();
      dom.drawerTitle.textContent = 'Could not load this show';
      dom.drawerOverview.textContent = err.message;
    }
  }

  async function loadRecommendations(id, seq) {
    try {
      const data = await api(`/api/shows/${id}/recommendations`);
      if (seq !== state.drawerSeq) return;
      const recs = data.recommendations || [];
      dom.recGrid.replaceChildren();
      if (!recs.length) {
        dom.recStatus.textContent = data.warning || 'No similar shows were found.';
        return;
      }
      dom.recStatus.textContent = '';
      for (const r of recs) dom.recGrid.appendChild(createCard(r, (rid) => openDrawer(rid)));
    } catch (err) {
      if (seq !== state.drawerSeq) return;
      dom.recGrid.replaceChildren();
      dom.recStatus.textContent = `Similar shows are unavailable: ${err.message}`;
    }
  }

  function trapFocus(e) {
    if (e.key !== 'Tab' || !dom.drawer.classList.contains('open')) return;
    const f = dom.drawer.querySelectorAll('button:not([disabled]), [href], input, select, [tabindex]:not([tabindex="-1"])');
    if (!f.length) return;
    const first = f[0];
    const last = f[f.length - 1];
    if (e.shiftKey && (document.activeElement === first || document.activeElement === dom.drawer)) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  }

  // ---------------------------------------------------------------- events
  function bindEvents() {
    dom.search.addEventListener('input', () => {
      clearTimeout(state.searchTimer);
      const val = dom.search.value.trim();
      dom.searchClear.hidden = !val;
      state.searchTimer = setTimeout(() => {
        if (val === state.query) return;
        state.query = val;
        state.page = 1;
        syncControls();
        loadShows();
        if (val) ensureCatalogVisible();
      }, 350);
    });
    dom.search.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && dom.search.value) { e.preventDefault(); clearTimeout(state.searchTimer); setQuery(''); }
    });
    dom.searchClear.addEventListener('click', () => { clearTimeout(state.searchTimer); setQuery(''); dom.search.focus(); });

    dom.tabs.forEach((t) => t.addEventListener('click', () => {
      applyPreset(t.dataset.preset);
      state.page = 1;
      syncControls();
      loadShows();
      ensureCatalogVisible();
    }));

    dom.sortSelect.addEventListener('change', () => {
      state.preset = null;
      state.minRating = null;
      state.minVotes = null;
      state.sortBy = dom.sortSelect.value;
      state.order = state.sortBy === 'name' ? 'asc' : 'desc';
      state.page = 1;
      syncControls();
      loadShows();
    });
    dom.orderBtn.addEventListener('click', () => {
      state.order = state.order === 'desc' ? 'asc' : 'desc';
      state.page = 1;
      syncControls();
      loadShows();
    });
    dom.resetBtn.addEventListener('click', resetAll);
    dom.gridEmptyReset.addEventListener('click', resetAll);
    dom.gridRetry.addEventListener('click', loadShows);
    dom.brand.addEventListener('click', (e) => { e.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); });

    dom.pageFirst.addEventListener('click', () => goToPage(1));
    dom.pagePrev.addEventListener('click', () => goToPage(state.page - 1));
    dom.pageNext.addEventListener('click', () => goToPage(state.page + 1));
    dom.pageLast.addEventListener('click', () => goToPage(state.totalPages));

    dom.heroDetails.addEventListener('click', () => { if (state.featuredId) openDrawer(state.featuredId, dom.heroDetails); });
    dom.heroSimilar.addEventListener('click', () => {
      if (!state.featuredId) return;
      openDrawer(state.featuredId, dom.heroSimilar);
      setTimeout(() => { dom.recGrid.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 450);
    });

    dom.drawerClose.addEventListener('click', closeDrawer);
    dom.overlay.addEventListener('click', closeDrawer);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && dom.drawer.classList.contains('open')) closeDrawer();
      trapFocus(e);
    });
  }

  bindEvents();
  syncControls();
  initGenres();
  initHero();
  loadShows();
})();
