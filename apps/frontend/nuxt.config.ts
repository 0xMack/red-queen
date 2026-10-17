import tailwindcss from '@tailwindcss/vite'

// The public site (docs/design/0019) is a static build: `NUXT_PUBLIC_API_MODE=static pnpm generate` -- a single-page app
// (no server to render on) reading the API's static copy. Development and local use stay `live` against apis/backend.
const staticSite = process.env.NUXT_PUBLIC_API_MODE === 'static'

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: !staticSite },
  ssr: !staticSite,
  modules: ['@pinia/nuxt'],
  // Components live in folders by domain (ui/, lab/, game/, versus/, ...) but are named by file alone, so moving
  // one never renames it -- file names are therefore unique across folders.
  components: [{ path: '~/components', pathPrefix: false }],
  // Fonts are served from the site itself, not Google Fonts: the page is cross-origin isolated (multithreaded ONNX
  // Runtime needs it, docs/design/0019), and a third-party stylesheet would be blocked.
  css: [
    '@fontsource-variable/geist',
    '@fontsource-variable/geist-mono',
    '@fontsource/instrument-serif/400.css',
    '@fontsource/instrument-serif/400-italic.css',
    '~/assets/css/main.css',
  ],
  app: {
    pageTransition: { name: 'page', mode: 'out-in' },
    head: {
      htmlAttrs: { lang: 'en' },
      meta: [{ name: 'theme-color', content: '#0c0b0a' }],
      link: [{ rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    },
  },
  vite: {
    plugins: [tailwindcss()],
    // inference/runtime.ts: where ONNX Runtime's own .mjs/.wasm load from (empty: bundled with the app). A build-time
    // constant, because it's read inside web workers, where Nuxt's runtime config doesn't reach.
    define: { __ORT_BASE__: JSON.stringify(process.env.NUXT_PUBLIC_ORT_BASE ?? '') },
  },
  runtimeConfig: {
    // Overridable via NUXT_PUBLIC_API_BASE -- see apis/backend/README.md for running that service. In `static` mode
    // (NUXT_PUBLIC_API_MODE) it is where the API's static copy lives, e.g. https://redqueen-data.mackinations.ca/api.
    public: {
      apiBase: 'http://127.0.0.1:8000',
      apiMode: 'live' as 'live' | 'static',
    },
  },
})
