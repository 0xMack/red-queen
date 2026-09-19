<script setup lang="ts">
// Playing now lives on the game page itself (docs/design/0007: watch the algorithms first, then
// "Play it yourself" on the same stage). Kept as a redirect so old /play/<game> links still work.
const route = useRoute()
await navigateTo({ path: `/games/${route.params.game}`, query: { mode: "play" } }, { redirectCode: 301, replace: true })
</script>

<template>
  <div />
</template>
