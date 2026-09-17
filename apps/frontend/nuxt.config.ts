import tailwindcss from '@tailwindcss/vite'

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  modules: ['@pinia/nuxt'],
  css: ['~/assets/css/main.css'],
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
