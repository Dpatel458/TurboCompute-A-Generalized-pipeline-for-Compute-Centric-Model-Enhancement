
// =========================================================
// Navbar blur on scroll – NeuralPrune
// =========================================================

const navbar = document.getElementById("navbar");

if (navbar) {
  window.addEventListener("scroll", () => {
    if (window.scrollY > 40) {
      navbar.classList.add("navbar-scrolled");
    } else {
      navbar.classList.remove("navbar-scrolled");
    }
  });
}

const toggle = document.getElementById("smToggle");
const plus = document.getElementById("smPlus");
const text = document.getElementById("smToggleInner");
const panel = document.getElementById("smPanel");
const layers = document.querySelectorAll(".sm-layer");
const items = document.querySelectorAll(".sm-item");

if (toggle && panel) {
  let open = false;

  gsap.set([panel, layers], { xPercent: 100 });

  toggle.addEventListener("click", () => {
    open = !open;

    if (open) {
      document.body.classList.add("menu-open");
      gsap.timeline()
        .to(layers[0], { xPercent: 0, duration: 0.45 })
        .to(layers[1], { xPercent: 0, duration: 0.45 }, 0.1)
        .to(panel, { xPercent: 0, duration: 0.6 }, 0.2)
        .to(items, {
          yPercent: 0,
          opacity: 1,
          stagger: 0.08,
          duration: 0.8
        }, 0.45);

      plus.style.transform = "rotate(90deg)";
      text.textContent = "Close";
      document.body.style.overflow = "hidden";
    } else {
      document.body.classList.remove("menu-open");
      gsap.timeline()
        .to(items, {
          yPercent: 120,
          opacity: 0,
          stagger: 0.05,
          duration: 0.3
        })
        .to([panel, layers], { xPercent: 100, duration: 0.4 });

      plus.style.transform = "rotate(0deg)";
      text.textContent = "Menu";
      document.body.style.overflow = "";
    }
  });
}


