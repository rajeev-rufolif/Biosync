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
