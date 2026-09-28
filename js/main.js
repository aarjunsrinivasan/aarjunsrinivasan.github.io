/* Homepage: theme toggle and sidebar section highlight. The initial theme is set inline in <head>. */
function toggleTheme() {
    const html = document.documentElement;
    const newTheme = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    try { localStorage.setItem('theme', newTheme); } catch (e) {}
}

// Follow system theme changes unless the user has picked one
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    let saved = null;
    try { saved = localStorage.getItem('theme'); } catch (err) {}
    if (!saved) {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
    }
});

// Highlight the sidebar link for the section currently at the top of the viewport
(function () {
    const links = document.querySelectorAll('.side-nav a');
    const sections = [...links].map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
    if (!sections.length) return;

    function update() {
        let current = sections[0];
        const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
        if (atBottom) {
            current = sections[sections.length - 1];
        } else {
            sections.forEach(s => {
                if (s.getBoundingClientRect().top <= 120) current = s;
            });
        }
        links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + current.id));
    }

    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
})();
