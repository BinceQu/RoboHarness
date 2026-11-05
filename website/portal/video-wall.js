(() => {
  const wall = document.querySelector('.video-wall');
  if (!wall) return;
  const video = wall.querySelector('video');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let nearby = false, loaded = false, loading = false;
  const active = () => nearby && !document.hidden && !motion.matches;
  const update = () => {
    if (!active() || !loaded) { video.pause(); return; }
    video.play().catch(() => wall.classList.remove('is-playing'));
  };
  const load = async () => {
    if (loading || loaded || motion.matches) return;
    loading = true;
    try {
      // Buffer once; the chosen recording already contains the complete scrolling effect.
      const response = await fetch(video.dataset.src);
      if (!response.ok) throw new Error(`Video wall: HTTP ${response.status}`);
      const blob = await response.blob();
      video.src = URL.createObjectURL(blob);
      video.load();
    } catch (error) {
      loading = false;
      console.warn('Unable to load the video wall.', error);
    }
  };
  video.addEventListener('canplay', () => { loaded = true; update(); });
  video.addEventListener('playing', () => wall.classList.add('is-playing'));
  ['waiting', 'pause', 'ended', 'error'].forEach((event) =>
    video.addEventListener(event, () => wall.classList.remove('is-playing')));
  new IntersectionObserver(([entry]) => {
    nearby = entry.isIntersecting;
    load(); update();
  }, {rootMargin: '180px 0px'}).observe(wall);
  document.addEventListener('visibilitychange', update);
  motion.addEventListener('change', () => { load(); update(); });
  window.addEventListener('online', load);
  window.addEventListener('pageshow', update);
  load();
})();
