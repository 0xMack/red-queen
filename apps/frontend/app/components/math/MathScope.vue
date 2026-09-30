<script setup lang="ts">
// One figure's term scope (docs/design/0012): wrap a formula, the prose around it and the lab it describes, and they
// share one focus and one set of term colours. The scope is in the slot props for a parent that binds values.
const scope = provideTermScope()
defineExpose({ scope })
</script>

<template>
  <div class="math-scope">
    <slot :scope="scope" />
  </div>
</template>
