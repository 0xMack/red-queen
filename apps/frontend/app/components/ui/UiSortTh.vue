<script setup lang="ts" generic="K extends string">
// A sortable table header: click to sort by `column`, again to flip the direction. The arrow shows only on the
// active column.
defineProps<{ column: K; active: K; desc: boolean; align?: "left" | "right"; title?: string }>()
const emit = defineEmits<{ sort: [column: K] }>()
</script>

<template>
  <th
    class="cursor-pointer px-3 py-3 font-medium whitespace-nowrap transition select-none hover:text-fg"
    :class="[align === 'right' ? 'text-right' : '', active === column ? 'text-fg' : '']"
    :title="title"
    :aria-sort="active === column ? (desc ? 'descending' : 'ascending') : 'none'"
    @click="emit('sort', column)"
  >
    <slot />
    <span class="ml-1 inline-block w-2 text-queen-300">{{ active === column ? (desc ? "↓" : "↑") : "" }}</span>
  </th>
</template>
