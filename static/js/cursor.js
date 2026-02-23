// Custom Cursor with enhanced effects
const cursorRing = document.getElementById('cursorRing');
const cursorDot = document.getElementById('cursorDot');
const isTouch = window.matchMedia('(pointer: coarse)').matches;

if (!isTouch && cursorRing && cursorDot) {
  let mouseX = 0, mouseY = 0;
  let ringX = 0, ringY = 0;
  let dotX = 0, dotY = 0;

  document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  }, { passive: true });

  function animateCursor() {
    ringX += (mouseX - ringX) * 0.12;
    ringY += (mouseY - ringY) * 0.12;
    cursorRing.style.left = ringX + 'px';
    cursorRing.style.top = ringY + 'px';

    dotX += (mouseX - dotX) * 0.5;
    dotY += (mouseY - dotY) * 0.5;
    cursorDot.style.left = dotX + 'px';
    cursorDot.style.top = dotY + 'px';

    requestAnimationFrame(animateCursor);
  }
  animateCursor();

  // Hover effects
const interactiveElements = document.querySelectorAll(
  'a, button, label, input, .cursor-pointer, .cursor-interactive'
);

  interactiveElements.forEach((el) => {
    el.addEventListener('mouseenter', () => {
      gsap.to(cursorRing, {
        scale: 2,
        borderColor: 'rgba(6, 182, 212, 0.8)',
        duration: 0.2
      });
      gsap.to(cursorDot, { scale: 0.5, duration: 0.2 });
    });

    el.addEventListener('mouseleave', () => {
      gsap.to(cursorRing, {
        scale: 1,
        borderColor: 'rgba(139, 92, 246, 0.6)',
        duration: 0.2
      });
      gsap.to(cursorDot, { scale: 1, duration: 0.2 });
    });
  });
}


