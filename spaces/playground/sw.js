/* Service Worker for PySAD Playground offline support.
 *
 * Caches static playground assets on install and serves them
 * cache-first so the page loads without a network connection.
 * Pyodide and third-party CDN responses are cached on first use
 * (stale-while-revalidate) so a prior online visit enables
 * fully offline operation in air-gapped environments.
 */

const CACHE_NAME = "pysad-playground-v1";

const PRECACHE_URLs = [
  "./",
  "./worker.js",
  "./engine.py",
  "./manifest.json",
  "./initial_snapshot.json",
  "./data/nab/realKnownCause/nyc_taxi.csv",
  "./data/nab/combined_windows.json",
  "./data/arrhythmia.mat",
];

const CACHEABLE_ORIGINS = [
  "https://cdn.jsdelivr.net",
  "https://cdn.plot.ly",
  "https://fonts.googleapis.com",
  "https://fonts.gstatic.com",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLs))
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) {
          return cached;
        }
        return fetch(event.request).then((response) => {
          if (
            event.request.method === "GET" &&
            response.ok &&
            response.type === "basic"
          ) {
            return caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, response.clone());
              return response;
            });
          }
          return response;
        });
      })
    );
    return;
  }

  if (CACHEABLE_ORIGINS.some((origin) => url.origin === origin)) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        const networkFetch = fetch(event.request).then((response) => {
          if (event.request.method === "GET" && response.ok) {
            return caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, response.clone());
              return response;
            });
          }
          return response;
        });
        return cached || networkFetch;
      })
    );
    return;
  }
});
