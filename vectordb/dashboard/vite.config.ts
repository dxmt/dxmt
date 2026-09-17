import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `base` is absolute rather than relative because the server mounts this at `/dashboard/` and
// the app has nested client-side routes: from `/dashboard/tables/x/search`, a relative asset
// URL resolves against `/dashboard/tables/x/`, which does not exist.
//
// The dev proxy is the point of the dev setup. The alternative is pointing the app at
// `http://127.0.0.1:9999` and relying on the server's CORS allow-list, which permits exactly
// `localhost:3000` and `127.0.0.1:3000` -- so it works only on that one port, and only for the
// methods that list happens to include. Proxying makes dev same-origin, which is what
// production is, so the two cannot diverge.
export default defineConfig({
  base: '/dashboard/',
  build: {
    outDir: 'dist',
    // Every file the server will hash-cache lives here; `api/dashboard.rs` decides
    // `immutable` by this prefix, so renaming it breaks caching silently.
    assetsDir: 'assets',
    sourcemap: false,
  },
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://127.0.0.1:9999',
      '/status': 'http://127.0.0.1:9999',
      '/health': 'http://127.0.0.1:9999',
      '/live': 'http://127.0.0.1:9999',
      '/ready': 'http://127.0.0.1:9999',
    },
  },
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./test/setup.ts'],
  },
});
