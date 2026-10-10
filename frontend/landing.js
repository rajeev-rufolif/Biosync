// Landing-page behaviour only. chat.html and results.html still use script.js.
(() => {
  const nav = document.querySelector(".nav");
  const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 8);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  const reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        e.target.classList.add("in");
        io.unobserve(e.target); // reveal once
      });
    }, { threshold: 0.12 });
    reveals.forEach((el) => io.observe(el));

    // Scroll-spy: mark the nav link whose section is mid-viewport.
    const links = [...document.querySelectorAll(".nav__links a")];
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.toggleAttribute("aria-current", a.getAttribute("href") === "#" + e.target.id));
        links.forEach((a) => { if (!a.hasAttribute("aria-current")) return; a.setAttribute("aria-current", "true"); });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    links.forEach((a) => { const s = document.querySelector(a.getAttribute("href")); if (s) spy.observe(s); });
  } else {
    reveals.forEach((el) => el.classList.add("in"));
  }

  // Cursor spotlight on cards via CSS variables (skipped on touch).
  document.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return;
    const card = e.target.closest && e.target.closest(".card");
    if (!card) return;
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", e.clientX - r.left + "px");
    card.style.setProperty("--my", e.clientY - r.top + "px");
  }, { passive: true });
})();

// Mobile menu: the section links are hidden below 880px, so they live in a full-screen menu there.
(() => {
  const btn = document.querySelector('.nav__menu'), menu = document.getElementById('menu');
  if (!btn || !menu) return;
  const isOpen = () => !menu.hidden;
  const set = open => {
    menu.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
    btn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    document.body.classList.toggle('menu-open', open);
    if (open) menu.querySelector('a').focus();
  };
  btn.addEventListener('click', () => set(!isOpen()));
  menu.addEventListener('click', e => { if (e.target.closest('a')) set(false); });
  document.addEventListener('keydown', e => {
    if (!isOpen()) return;
    if (e.key === 'Escape') { set(false); btn.focus(); return; }
    if (e.key !== 'Tab') return;
    // keep Tab inside the open menu: the toggle button plus its links
    const f = [btn, ...menu.querySelectorAll('a')];
    if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
    else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
  });
  matchMedia('(min-width:880px)').addEventListener('change', e => { if (e.matches && isOpen()) set(false); });
})();

// Phone-only floating CTA: appears once the hero buttons have scrolled away, hides again at the final CTA and footer.
(() => {
  const hero = document.querySelector('.hero .actions');
  const stops = [document.querySelector('.cta'), document.querySelector('.foot')].filter(Boolean);
  if (!hero || !('IntersectionObserver' in window)) return;
  const bar = document.createElement('a');
  bar.className = 'btn btn--primary sticky-cta';
  bar.href = 'chat.html'; bar.textContent = 'Build my schedule'; bar.tabIndex = -1; bar.setAttribute('aria-hidden', 'true');
  document.body.append(bar);
  let heroIn = true; const stopsIn = new Set();
  const sync = () => {
    const on = !heroIn && !stopsIn.size;
    bar.classList.toggle('is-on', on); bar.tabIndex = on ? 0 : -1; bar.setAttribute('aria-hidden', String(!on));
  };
  new IntersectionObserver(([e]) => { heroIn = e.isIntersecting; sync(); }).observe(hero);
  const io = new IntersectionObserver(es => { es.forEach(e => e.isIntersecting ? stopsIn.add(e.target) : stopsIn.delete(e.target)); sync(); });
  stops.forEach(el => io.observe(el));
})();
