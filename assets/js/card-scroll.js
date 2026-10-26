// Keep the first two cards fully visible; subsequent cards scroll within the region.
(() => {
  const init = () => {
    document.querySelectorAll('.two-card-scroll').forEach((panel) => {
      const cards = Array.from(panel.children);
      const visibleCards = cards.slice(0, 2);
      if (!visibleCards.length) return;

      const resize = () => {
        const style = getComputedStyle(panel);
        const px = (value) => Number.parseFloat(value) || 0;
        const height = visibleCards.reduce((sum, card) => sum + card.getBoundingClientRect().height, 0)
          + px(style.rowGap) * (visibleCards.length - 1)
          + px(style.paddingTop) + px(style.paddingBottom)
          + px(style.borderTopWidth) + px(style.borderBottomWidth);
        panel.style.height = `${Math.ceil(height)}px`;
      };

      resize();
      if ('ResizeObserver' in window) {
        const observer = new ResizeObserver(resize);
        visibleCards.forEach((card) => observer.observe(card));
      } else {
        window.addEventListener('resize', resize);
        window.addEventListener('load', resize, { once: true });
      }
      if (document.fonts) document.fonts.ready.then(resize);
    });
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
