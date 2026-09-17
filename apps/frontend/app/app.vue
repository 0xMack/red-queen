<template>
  <div class="min-h-screen bg-slate-50">
    <NuxtRouteAnnouncer />
    <nav class="border-b border-slate-200 bg-white px-6 py-3 text-sm">
      <NuxtLink to="/" class="font-medium text-slate-900 hover:underline">Runs</NuxtLink>
      <NuxtLink to="/play/snake" class="ml-4 text-slate-600 hover:underline">Play Snake</NuxtLink>
    </nav>
    <NuxtPage />
  </div>
</template>
