<script setup lang="ts">
const NAV_LINKS = [
  { to: "/", label: "Home" },
  { to: "/games", label: "Games" },
  { to: "/learn", label: "Learn" },
  { to: "/runs", label: "Runs" },
]
</script>

<template>
  <div class="min-h-screen bg-slate-50">
    <NuxtRouteAnnouncer />
    <nav class="border-b border-slate-200 bg-white px-6 py-3 text-sm">
      <NuxtLink
        v-for="link in NAV_LINKS"
        :key="link.to"
        :to="link.to"
        class="mr-4 font-medium text-slate-600 hover:text-slate-900"
        exact-active-class="text-slate-900"
      >
        {{ link.label }}
      </NuxtLink>
    </nav>
    <NuxtPage />
  </div>
</template>
