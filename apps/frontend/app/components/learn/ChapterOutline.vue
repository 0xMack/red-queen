<script setup lang="ts">
// "On this page": the chapter's sections (from useChapterOutline), the one being read marked.
defineProps<{ outline: { id: string; text: string }[]; activeId: string | null }>()
const toTop = () => window.scrollTo({ top: 0, behavior: "smooth" })
</script>

<template>
  <div v-if="outline.length" class="sticky top-20">
    <p class="label">On this page</p>
    <ol class="mt-3 space-y-px border-l border-line text-[13px]">
      <li v-for="(h, i) in outline" :key="h.id">
        <a
          :href="`#${h.id}`"
          class="-ml-px flex gap-2 border-l py-1 pl-3 leading-snug transition"
          :class="activeId === h.id ? 'border-queen-400 text-fg' : 'border-transparent text-fg-subtle hover:text-fg'"
        >
          <span class="num shrink-0 text-[10.5px] leading-5 opacity-60">{{ String(i + 1).padStart(2, "0") }}</span>
          {{ h.text }}
        </a>
      </li>
    </ol>
    <button class="mt-6 text-xs text-fg-subtle transition hover:text-fg" @click="toTop">↑ Back to top</button>
  </div>
</template>
