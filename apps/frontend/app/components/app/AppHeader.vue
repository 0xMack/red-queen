<script setup lang="ts">
import { isNavActive, NAV_LINKS } from "~/data/nav"

// Sticky, not fixed: stays pinned while scrolling without every page having to pad for it. Transparent at the
// top of a page, ruled once it scrolls.
const route = useRoute()
const menuOpen = ref(false)
watch(() => route.fullPath, () => (menuOpen.value = false))

const scrolled = ref(false)
function onScroll() {
  scrolled.value = window.scrollY > 4
}
onMounted(() => {
  onScroll()
  window.addEventListener("scroll", onScroll, { passive: true })
})
onUnmounted(() => window.removeEventListener("scroll", onScroll))
</script>

<template>
  <header
    class="sticky top-0 z-50 border-b transition-colors duration-300"
    :class="scrolled || menuOpen ? 'border-line bg-bg/85 backdrop-blur-xl' : 'border-transparent'"
  >
    <div class="mx-auto flex h-14 max-w-[1600px] items-center gap-8 px-4 sm:px-6 lg:px-8">
      <NuxtLink to="/" class="group flex items-center gap-2.5" aria-label="Red Queen, home">
        <BrandMark class="size-7 transition duration-300 group-hover:-rotate-6" />
        <span class="font-display text-[22px] leading-none tracking-[-0.01em]">Red Queen</span>
      </NuxtLink>

      <nav class="hidden items-center gap-1 sm:flex" aria-label="Sections">
        <NuxtLink
          v-for="link in NAV_LINKS"
          :key="link.to"
          :to="link.to"
          class="relative flex items-center gap-2 rounded-[6px] px-3 py-1.5 text-sm transition"
          :class="isNavActive(link, route.path) ? 'text-fg' : 'text-fg-muted hover:text-fg'"
        >
          <span
            class="size-1.5 rotate-45 transition"
            :class="isNavActive(link, route.path) ? 'bg-queen-400' : 'bg-transparent'"
          />
          {{ link.label }}
        </NuxtLink>
      </nav>

      <div class="ml-auto flex items-center gap-2">
        <NuxtLink :to="{ path: '/games/snake', query: { mode: 'play' } }" class="btn-ghost btn-sm hidden md:inline-flex">
          <span class="size-1.5 rounded-full bg-life-400" /> Play Snake
        </NuxtLink>
        <button
          class="rounded-[6px] p-2 text-fg-muted hover:bg-raised hover:text-fg sm:hidden"
          :aria-expanded="menuOpen"
          aria-label="Toggle navigation"
          @click="menuOpen = !menuOpen"
        >
          <svg viewBox="0 0 20 20" class="size-5" fill="none" stroke="currentColor" stroke-width="1.6">
            <path v-if="!menuOpen" d="M3 7h14M3 13h14" stroke-linecap="round" />
            <path v-else d="M5 5l10 10M15 5L5 15" stroke-linecap="round" />
          </svg>
        </button>
      </div>
    </div>

    <nav v-if="menuOpen" class="border-t border-line px-4 py-3 sm:hidden" aria-label="Sections">
      <NuxtLink
        v-for="link in NAV_LINKS"
        :key="link.to"
        :to="link.to"
        class="block rounded-[6px] px-3 py-2.5"
        :class="isNavActive(link, route.path) ? 'bg-raised' : ''"
      >
        <span class="font-display text-xl">{{ link.label }}</span>
        <span class="block text-xs text-fg-subtle">{{ link.blurb }}</span>
      </NuxtLink>
    </nav>
  </header>
</template>
