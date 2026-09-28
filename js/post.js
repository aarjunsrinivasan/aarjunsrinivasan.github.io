/* Blog post enhancements: margin sidenotes built from footnotes, heading accent on scroll */
(function () {
    var article = document.querySelector('.blog-content');
    if (!article) return;

    // Sidenotes: clone each footnote next to its reference; CSS shows them only on wide screens.
    var refs = article.querySelectorAll('.footnote-ref a[href^="#fn"]');
    var notes = [];
    refs.forEach(function (ref) {
        var fn = document.getElementById(ref.getAttribute('href').slice(1));
        if (!fn) return;
        var aside = document.createElement('aside');
        aside.className = 'sidenote';
        aside.innerHTML = fn.innerHTML;
        aside.querySelectorAll('a[href^="#fnref"]').forEach(function (a) { a.remove(); });
        var num = document.createElement('span');
        num.className = 'sidenote-num';
        num.textContent = ref.textContent;
        aside.prepend(num);
        article.appendChild(aside);
        notes.push({ ref: ref, aside: aside });
        ref.addEventListener('click', function (e) {
            if (getComputedStyle(aside).display === 'none') return;
            e.preventDefault();
            aside.animate([{ opacity: 0.3 }, { opacity: 1 }], { duration: 400 });
        });
    });

    function layout() {
        var bottom = 0;
        var top0 = article.getBoundingClientRect().top;
        notes.forEach(function (n) {
            var y = n.ref.getBoundingClientRect().top - top0;
            y = Math.max(y, bottom);
            n.aside.style.top = y + 'px';
            bottom = y + n.aside.offsetHeight + 16;
        });
    }

    if (notes.length) {
        article.classList.add('has-sidenotes');
        layout();
        window.addEventListener('resize', layout);
        window.addEventListener('load', layout);
        if (document.fonts) document.fonts.ready.then(layout);
    }

    // Heading accent: add .in-view once each h2 scrolls into view.
    var headings = article.querySelectorAll('h2');
    if (!('IntersectionObserver' in window)) {
        headings.forEach(function (h) { h.classList.add('in-view'); });
        return;
    }
    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (!entry.isIntersecting) return;
            entry.target.classList.add('in-view');
            observer.unobserve(entry.target);
        });
    }, { threshold: 0.6 });
    headings.forEach(function (h) { observer.observe(h); });
})();
