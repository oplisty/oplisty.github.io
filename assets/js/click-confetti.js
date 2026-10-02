(() => {
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const COLORS = ['#f94144', '#f3722c', '#f8961e', '#f9c74f', '#90be6d', '#43aa8b', '#577590', '#b5179e', '#f72585'];
  const MAX_LIVE = 150;
  let live = 0;

  const spawnBurst = (x, y) => {
    if (live > MAX_LIVE || !document.body) return;
    const count = 10 + Math.floor(Math.random() * 5);

    for (let i = 0; i < count; i++) {
      const p = document.createElement('span');
      const size = 6 + Math.random() * 5;
      p.style.cssText =
        'position:fixed;pointer-events:none;z-index:99999;' +
        'left:' + (x - size / 2) + 'px;top:' + (y - size / 2) + 'px;' +
        'width:' + size + 'px;height:' + (size * 0.6) + 'px;' +
        'background:' + COLORS[(Math.random() * COLORS.length) | 0] + ';' +
        'border-radius:' + (Math.random() < 0.3 ? '50%' : '2px') + ';';
      document.body.appendChild(p);
      live++;

      const angle = Math.random() * Math.PI * 2;
      const dist = 40 + Math.random() * 70;
      const dx = Math.cos(angle) * dist;
      const dy = Math.sin(angle) * dist * 0.6 - 30;
      const rot = (Math.random() - 0.5) * 720;

      const anim = p.animate(
        [
          { transform: 'translate(0,0) rotate(0deg)', opacity: 1 },
          { transform: 'translate(' + dx * 0.7 + 'px,' + dy * 0.7 + 'px) rotate(' + rot * 0.6 + 'deg)', opacity: 1, offset: 0.45 },
          { transform: 'translate(' + dx + 'px,' + (dy + 110) + 'px) rotate(' + rot + 'deg)', opacity: 0 }
        ],
        { duration: 700 + Math.random() * 400, easing: 'cubic-bezier(.16,.84,.44,1)' }
      );

      const cleanup = () => {
        p.remove();
        live--;
      };
      anim.onfinish = cleanup;
      anim.oncancel = cleanup;
    }
  };

  document.addEventListener('click', (e) => spawnBurst(e.clientX, e.clientY), { passive: true });
})();
