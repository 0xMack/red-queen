<script setup lang="ts">
// The crown mark -- same drawing as public/favicon.svg. The gradient id is per-instance so two marks
// on one page (header + footer) don't collide.
const id = `brand-${useId()}`
</script>

<template>
  <svg viewBox="0 0 32 32" aria-hidden="true">
    <defs>
      <linearGradient :id="id" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#ff5c7a" />
        <stop offset="1" stop-color="#9b1535" />
      </linearGradient>
    </defs>
    <rect width="32" height="32" rx="8" :fill="`url(#${id})`" />
    <path d="M7 22 L8.5 11 L13 16 L16 8.5 L19 16 L23.5 11 L25 22 Z" fill="#fff" />
    <rect x="7" y="23.5" width="18" height="2.5" rx="1.2" fill="#fff" opacity=".85" />
  </svg>
</template>
