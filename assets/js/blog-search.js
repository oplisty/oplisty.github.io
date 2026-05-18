document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("blog-search-input");
  const clearButton = document.getElementById("blog-search-clear");
  const status = document.getElementById("blog-search-status");
  const emptyState = document.getElementById("blog-search-empty");
  const cards = Array.from(document.querySelectorAll(".blog-card"));

  if (!input || !clearButton || !status || cards.length === 0) {
    return;
  }

  const updateStatus = (visibleCount, query) => {
    if (!query) {
      status.textContent = `${visibleCount} post${visibleCount === 1 ? "" : "s"} total`;
      return;
    }

    status.textContent = `${visibleCount} result${visibleCount === 1 ? "" : "s"} for “${query}”`;
  };

  const runSearch = () => {
    const query = input.value.trim().toLowerCase();
    let visibleCount = 0;

    cards.forEach((card) => {
      const haystack = card.dataset.blogSearch || "";
      const matched = !query || haystack.includes(query);
      card.hidden = !matched;
      if (matched) {
        visibleCount += 1;
      }
    });

    if (emptyState) {
      emptyState.hidden = visibleCount !== 0;
    }

    updateStatus(visibleCount, input.value.trim());
  };

  clearButton.addEventListener("click", () => {
    input.value = "";
    runSearch();
    input.focus();
  });

  input.addEventListener("input", runSearch);
  runSearch();
});
