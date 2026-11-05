(() => {
  const wall = document.querySelector('.video-wall');
  if (!wall) return;
  const videos = [...wall.querySelectorAll('video')];
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const started = performance.now();
  let nearby = false;

  // Keep repeated tiles at the same point in the clip as the wall loops.
  const sync = (video) => {
    if (Number.isFinite(video.duration) && video.duration > 0) {
      video.currentTime = ((performance.now() - started) / 1000) % video.duration;
    }
  };
  const update = () => {
    const playing = nearby && !document.hidden && !motion.matches;
    wall.classList.toggle('is-playing', playing);
    videos.forEach((video) => {
      if (!playing) { video.pause(); return; }
      if (!video.hasAttribute('src')) video.src = video.dataset.src;
      sync(video);
      video.play().catch(() => {});
    });
  };
  videos.forEach((video) => video.addEventListener('loadedmetadata', () => sync(video)));
  new IntersectionObserver(([entry]) => {
    nearby = entry.isIntersecting;
    update();
  }, {rootMargin: '180px 0px'}).observe(wall);
  document.addEventListener('visibilitychange', update);
  motion.addEventListener('change', update);
})();
