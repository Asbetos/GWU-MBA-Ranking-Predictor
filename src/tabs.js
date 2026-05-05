/**
 * Tab navigation — toggles `.active` on `.tab-btn` and `.hidden` on `.tab-pane`.
 * Emits a `tab:activated` CustomEvent so consumers can lazy-init their pane.
 */

const PANE_PREFIX = 'tab-';

export function initTabs() {
  const buttons = Array.from(document.querySelectorAll('.tab-btn'));
  const panes = Array.from(document.querySelectorAll('.tab-pane'));

  function activate(tabKey) {
    for (const btn of buttons) {
      const active = btn.dataset.tab === tabKey;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    }
    for (const pane of panes) {
      pane.classList.toggle('hidden', pane.id !== `${PANE_PREFIX}${tabKey}`);
    }
    document.dispatchEvent(new CustomEvent('tab:activated', { detail: { tabKey } }));
  }

  for (const btn of buttons) {
    btn.addEventListener('click', () => activate(btn.dataset.tab));
  }

  // Activate whichever tab starts marked active in markup.
  const initial = buttons.find((b) => b.classList.contains('active'))?.dataset.tab || buttons[0]?.dataset.tab;
  if (initial) activate(initial);

  return { activate };
}
