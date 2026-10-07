// Offline support: cache the whole app on install, serve from cache first.
// Bump VERSION whenever any cached file changes so phones pick up the update.
const VERSION = "v3";
const CACHE = `toray-colour-${VERSION}`;
const FILES = [
  "./",
  "index.html",
  "css/app.css",
  "js/main.js",
  "js/color.js",
  "js/yarncalc.js",
  "data/dataset.json",
  "manifest.webmanifest",
  "icons/icon.svg",
  "icons/icon-192.png",
  "icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    caches.match(event.request, { ignoreSearch: true }).then((hit) => hit || fetch(event.request)),
  );
});
