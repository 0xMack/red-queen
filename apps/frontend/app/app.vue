<script setup lang="ts">
// Here, not in nuxt.config.ts: a function there can't be serialized into the build, so it was silently dropped.
useHead({ titleTemplate: (title?: string) => (title ? `${title} · Red Queen` : "Red Queen") })
</script>

<template>
  <div class="flex min-h-screen flex-col">
    <NuxtRouteAnnouncer />
    <AppHeader />
    <div class="flex-1">
      <NuxtPage />
    </div>
    <AppFooter />
  </div>
</template>
