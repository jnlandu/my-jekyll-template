---
layout: default
title: "News & Updates"
permalink: /news/
description: "All news, achievements, fellowships, and updates from Jeremie Nlandu Mabiala"
---

<style>
/* Force disable scroll-snap and ensure page starts at top */
html, body, *, .container {
  scroll-snap-type: none !important;
  scroll-snap-align: none !important;
  scroll-snap-stop: normal !important;
}

/* Ensure page starts at very top */
body {
  scroll-padding-top: 0 !important;
}
</style>

<section class="news-header-section" id="news-top">
  <div class="container">
    <header class="page-header">
      <!-- <div class="page-eyebrow">All Updates</div> -->
      <h1 class="page-title">News & Updates</h1>
      <p class="page-subtitle">
       Latest updates about me, my achievements, fellowships, conferences, and exciting events that I attended in chronological order.
      </p>
    </header>
  </div>
  
</section>

<section class="news-content-section" id="news-content">
  <div class="container">
    <div class="news-page-grid">
      <div class="news-main">
        <div class="news-timeline-full">
          {% assign all_news = site.news | sort: "date" | reverse %}
          {% for news in all_news %}
          <article class="news-item-full">
            <div class="news-date-full">{{ news.date | date: "%B %d, %Y" }}</div>
            <div class="news-content-full">
              <h2 class="news-title-full">
                <i class="{{ news.icon | default: 'fas fa-newspaper' }} news-icon"></i>
                {{ news.title }}
              </h2>
              <div class="news-description-full">
                {% assign content_text = news.content | strip_html | strip_newlines %}
                {% if content_text.size > 150 %}
                  {{ content_text | truncate: 150 | append: '... ' }}<a href="{{ news.url | relative_url }}" class="news-more-link">more</a>
                {% else %}
                  {{ news.content | markdownify }}
                {% endif %}
              </div>
              {% if news.tags and news.tags.size > 0 %}
              <div class="news-tags-full">
                {% for tag in news.tags %}
                  <span class="tag chip">{{ tag }}</span>
                {% endfor %}
              </div>
              {% endif %}
            </div>
          </article>
          {% endfor %}
        </div>
      </div>

      <aside class="news-sidebar-full">
        <div class="sidebar-section">
          <h3 class="sidebar-title">Categories</h3>
          <div class="category-list">
            {% assign all_tags = site.news | map: "tags" | join: "," | split: "," | uniq | sort %}
            {% for tag in all_tags %}
              {% if tag != "" %}
              <a href="#" class="category-tag" onclick="filterByTag('{{ tag }}')">
                {{ tag }}
              </a>
              {% endif %}
            {% endfor %}
          </div>
        </div>

        <div class="sidebar-section">
          <h3 class="sidebar-title">Timeline</h3>
          <div class="year-list">
            {% assign years = site.news | group_by_exp: "item", "item.date | date: '%Y'" | sort: "name" | reverse %}
            {% for year in years %}
            <a href="#" class="year-link" onclick="filterByYear('{{ year.name }}')">
              {{ year.name }} ({{ year.items.size }})
            </a>
            {% endfor %}
          </div>
        </div>

        {% assign current_project = site.current_projects | where: "featured", true | sort: "date" | reverse | first %}
        {% if current_project %}
        <div class="sidebar-section">
          <h3 class="sidebar-title">Current Project</h3>
          <div class="current-project-mini">
            <h4>{{ current_project.title }}</h4>
            {% if current_project.subtitle %}
              <p class="project-subtitle">{{ current_project.subtitle }}</p>
            {% endif %}
            <div class="project-status-mini">
              <span class="status-badge">{{ current_project.status }}</span>
              {% if current_project.progress %}
                <div class="progress-bar">
                  <div class="progress-fill" style="width: {{ current_project.progress }}%"></div>
                </div>
              {% endif %}
            </div>
          </div>
        </div>
        {% endif %}
      </aside>
    </div>
  </div>
</section>

<!-- Dots Navigation for News Page (reuse home styles) -->
<nav class="section-dots" aria-label="News Sections">
  <a href="#news-top" class="dot" aria-label="Top"></a>
  <a href="#news-content" class="dot" aria-label="Content"></a>
  <a href="#footer" class="dot" aria-label="Footer"></a>
  
</nav>

<script>
// Intersection observers for dots and animations
const dots = document.querySelectorAll('.section-dots .dot');
const footerEl = document.getElementById('footer') || document.querySelector('.site-footer');
const sections = [
  document.getElementById('news-top'),
  document.getElementById('news-content'),
  footerEl
].filter(Boolean);

function setActiveDot(index) {
  dots.forEach((d, i) => d.classList.toggle('active', i === index));
}

function bindDots() {
  dots.forEach((dot, i) => {
    dot.addEventListener('click', (e) => {
      e.preventDefault();
      sections[i]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

function observeSections() {
  const headerH = getComputedStyle(document.documentElement).getPropertyValue('--header-h').trim() || '80px';
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const idx = sections.indexOf(entry.target);
      if (idx >= 0 && entry.isIntersecting) setActiveDot(idx);
    });
  }, { threshold: [0.15, 0.3, 0.5], rootMargin: `-${headerH} 0px 40% 0px` });

  sections.forEach(s => obs.observe(s));

  // Fallback: update active dot on scroll by checking nearest section
  window.addEventListener('scroll', () => {
    const footerIdx = sections.length - 1;
  const doc = document.documentElement;
  const maxScroll = doc.scrollHeight - window.innerHeight;
    const progress = maxScroll > 0 ? (window.scrollY / maxScroll) : 0;
  const atBottom = (window.innerHeight + window.scrollY) >= (doc.scrollHeight - 16);
  // Flip to footer dot earlier when progress passes 0.85, or at absolute bottom
  if (progress >= 0.85 || atBottom) { setActiveDot(footerIdx); return; }

    let closestIdx = 0;
    let minDist = Infinity;
  const scrollMid = window.scrollY + window.innerHeight / 2;
    sections.forEach((s, i) => {
      const rect = s.getBoundingClientRect();
      const center = rect.top + window.scrollY + rect.height / 2;
      const dist = Math.abs(center - scrollMid);
      if (dist < minDist) { minDist = dist; closestIdx = i; }
    });
    setActiveDot(closestIdx);
  }, { passive: true });
}

function animateNewsItems() {
  const items = Array.from(document.querySelectorAll('.news-item-full'));
  // initialize stagger delays to mirror home
  items.forEach((el, idx) => {
    el.style.setProperty('--slide-delay', `${Math.min(idx * 120, 480)}ms`);
  });

  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -10% 0px' });

  items.forEach(el => io.observe(el));
}

// Disable scroll-snap and ensure page starts at top
function disableScrollSnapAndScrollToTop() {
  // Force disable scroll-snap on all possible containers
  document.documentElement.style.scrollSnapType = 'none !important';
  document.body.style.scrollSnapType = 'none !important';
  
  // Find and disable scroll-snap on any containers that might have it
  const containers = document.querySelectorAll('*');
  containers.forEach(el => {
    const computed = window.getComputedStyle(el);
    if (computed.scrollSnapType && computed.scrollSnapType !== 'none') {
      el.style.scrollSnapType = 'none !important';
    }
  });
  
  // Force scroll to absolute top
  window.scrollTo(0, 0);
  document.documentElement.scrollTop = 0;
  document.body.scrollTop = 0;
  
  // Override any CSS that might be setting scroll-snap
  const style = document.createElement('style');
  style.textContent = `
    html, body, * { 
      scroll-snap-type: none !important; 
      scroll-snap-align: none !important;
      scroll-padding-top: 0 !important;
    }
  `;
  document.head.appendChild(style);
}

// Run immediately
disableScrollSnapAndScrollToTop();

document.addEventListener('DOMContentLoaded', () => {
  disableScrollSnapAndScrollToTop();
  bindDots();
  observeSections();
  animateNewsItems();
  setupLocalNewsSectionNavigation();
});

// Also run after everything loads
window.addEventListener('load', () => {
  disableScrollSnapAndScrollToTop();
});

// Local one-section navigation between header and content for directional slide
function setupLocalNewsSectionNavigation() {
  const header = document.getElementById('news-top');
  const content = document.getElementById('news-content');
  if (!header || !content) return;

  let isAnimating = false;
  const slideInDown = 'section-slide-in-from-down';
  const slideInUp = 'section-slide-in-from-up';
  const slideOutUp = 'section-slide-out-up';
  const slideOutDown = 'section-slide-out-down';

  function transition(fromEl, toEl, direction) {
    if (isAnimating) return;
    isAnimating = true;
    const outClass = direction === 'down' ? slideOutUp : slideOutDown;
    const inClass = direction === 'down' ? slideInFromDownClass() : slideInFromUpClass();

    fromEl.classList.add(outClass);
    setTimeout(() => { fromEl.classList.remove(outClass); }, 400);
    setTimeout(() => {
      toEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      toEl.classList.add(inClass);
      setTimeout(() => { toEl.classList.remove(inClass); isAnimating = false; }, 650);
    }, 100);
  }

  function slideInFromDownClass() { return 'section-slide-in-from-down'; }
  function slideInFromUpClass() { return 'section-slide-in-from-up'; }

  // Helper to detect if viewport is within the header or near top of content
  function nearHeaderOrTopContent() {
    const y = window.scrollY || document.documentElement.scrollTop;
    const headerBottom = header.getBoundingClientRect().bottom + window.scrollY;
    const contentTop = content.getBoundingClientRect().top + window.scrollY;
    const threshold = 120; // px leeway
    return y < contentTop - threshold || y < headerBottom;
  }

  // Wheel navigation only when near header ↔ content boundary; otherwise let page scroll
  window.addEventListener('wheel', (e) => {
    if (!nearHeaderOrTopContent() || isAnimating) return; // don't hijack deep inside content
    const dy = e.deltaY;
    if (Math.abs(dy) < 30) return;
    if (dy > 0) {
      e.preventDefault();
      transition(header, content, 'down');
    } else if (dy < 0) {
      e.preventDefault();
      transition(content, header, 'up');
    }
  }, { passive: false });

  // Keyboard navigation when viewport is near the boundary
  document.addEventListener('keydown', (e) => {
    if (isAnimating || !nearHeaderOrTopContent()) return;
    if (e.code === 'ArrowDown' || e.code === 'Space') {
      e.preventDefault();
      transition(header, content, 'down');
    } else if (e.code === 'ArrowUp') {
      e.preventDefault();
      transition(content, header, 'up');
    }
  });
}
function filterByTag(tag) {
  const items = document.querySelectorAll('.news-item-full');
  items.forEach(item => {
    const tags = item.querySelector('.news-tags-full');
    if (tags && tags.textContent.includes(tag)) {
      item.style.display = 'block';
    } else {
      item.style.display = 'none';
    }
  });
  // refresh animations for visible items
  items.forEach(item => item.classList.remove('in-view'));
  animateNewsItems();
}

function filterByYear(year) {
  const items = document.querySelectorAll('.news-item-full');
  items.forEach(item => {
    const date = item.querySelector('.news-date-full');
    if (date && date.textContent.includes(year)) {
      item.style.display = 'block';
    } else {
      item.style.display = 'none';
    }
  });
  // refresh animations for visible items
  items.forEach(item => item.classList.remove('in-view'));
  animateNewsItems();
}

// Show all items
function showAll() {
  const items = document.querySelectorAll('.news-item-full');
  items.forEach(item => {
    item.style.display = 'block';
  });
  items.forEach(item => item.classList.remove('in-view'));
  animateNewsItems();
}
</script>
