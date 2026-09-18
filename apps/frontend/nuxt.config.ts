import tailwindcss from '@tailwindcss/vite'

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  modules: ['@pinia/nuxt'],
  css: ['~/assets/css/main.css'],
  app: {
    pageTransition: { name: 'page', mode: 'out-in' },
    head: {
      titleTemplate: (title?: string) => (title ? `${title} · Red Queen` : 'Red Queen'),
      htmlAttrs: { lang: 'en' },
      meta: [{ name: 'theme-color', content: '#090b10' }],
      link: [
        { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' },
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' },
        {
          rel: 'stylesheet',
          href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap',
        },
      ],
    },
  },
  vite: {
    plugins: [tailwindcss()],
  },
  runtimeConfig: {
    // Overridable via NUXT_PUBLIC_API_BASE -- see apis/backend/README.md for running that service.
    public: {
      apiBase: 'http://127.0.0.1:8000',
    },
  },
})
