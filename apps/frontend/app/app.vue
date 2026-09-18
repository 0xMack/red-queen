<script setup lang="ts">
const NAV_LINKS = [
  { to: "/", label: "Home", exact: true },
  { to: "/games", label: "Games", exact: false },
  { to: "/learn", label: "Learn", exact: false },
  { to: "/runs", label: "Runs", exact: false },
]

const route = useRoute()
const menuOpen = ref(false)
watch(() => route.fullPath, () => (menuOpen.value = false))

// /play and /watch belong to Games; everything else matches by prefix.
function isActive(link: (typeof NAV_LINKS)[number]): boolean {
  if (link.exact) return route.path === link.to
  if (link.to === "/games" && (route.path.startsWith("/play") || route.path.startsWith("/watch"))) return true
  return route.path.startsWith(link.to)
}

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
  <div class="flex min-h-screen flex-col">
    <NuxtRouteAnnouncer />

    <!-- Sticky, not fixed: stays pinned while scrolling without every page having to pad for it. -->
    <header
      class="sticky top-0 z-50 border-b backdrop-blur-xl transition-colors"
      :class="scrolled ? 'border-line bg-bg/80' : 'border-transparent bg-bg/40'"
    >
      <div class="mx-auto flex h-14 max-w-[1600px] items-center gap-6 px-4 sm:px-6 lg:px-8">
        <NuxtLink to="/" class="group flex items-center gap-2.5">
          <BrandMark class="size-7 transition group-hover:rotate-[-6deg]" />
          <span class="font-display text-[17px] font-semibold tracking-tight">
            Red<span class="text-queen-400">Queen</span>
          </span>
        </NuxtLink>

        <nav class="hidden items-center gap-1 sm:flex">
          <NuxtLink
            v-for="link in NAV_LINKS"
            :key="link.to"
            :to="link.to"
            class="relative rounded-md px-3 py-1.5 text-sm font-medium transition"
            :class="isActive(link) ? 'text-fg' : 'text-fg-muted hover:bg-raised hover:text-fg'"
          >
            {{ link.label }}
            <span
              v-if="isActive(link)"
              class="absolute inset-x-3 -bottom-[11px] h-0.5 rounded-full bg-queen-400 shadow-[0_0_12px_rgb(255_92_122/0.8)]"
            />
          </NuxtLink>
        </nav>

        <div class="ml-auto flex items-center gap-2">
          <NuxtLink to="/play/snake" class="btn-ghost btn-sm hidden md:inline-flex">
            <span class="size-1.5 rounded-full bg-life-400" /> Play Snake
          </NuxtLink>
          <button
            class="rounded-md p-2 text-fg-muted hover:bg-raised hover:text-fg sm:hidden"
            :aria-expanded="menuOpen"
            aria-label="Toggle navigation"
            @click="menuOpen = !menuOpen"
          >
            <svg viewBox="0 0 20 20" class="size-5" fill="none" stroke="currentColor" stroke-width="1.8">
              <path v-if="!menuOpen" d="M3 6h14M3 10h14M3 14h14" stroke-linecap="round" />
              <path v-else d="M5 5l10 10M15 5L5 15" stroke-linecap="round" />
            </svg>
          </button>
        </div>
      </div>

      <nav v-if="menuOpen" class="border-t border-line px-4 py-2 sm:hidden">
        <NuxtLink
          v-for="link in NAV_LINKS"
          :key="link.to"
          :to="link.to"
          class="block rounded-md px-3 py-2 text-sm font-medium"
          :class="isActive(link) ? 'bg-raised text-fg' : 'text-fg-muted'"
        >
          {{ link.label }}
        </NuxtLink>
      </nav>
    </header>

    <div class="flex-1">
      <NuxtPage />
    </div>

    <footer class="mt-24 border-t border-line">
      <div
        class="mx-auto flex max-w-[1600px] flex-col gap-4 px-4 py-8 text-sm text-fg-subtle sm:flex-row sm:items-center sm:px-6 lg:px-8"
      >
        <div class="flex items-center gap-2">
          <BrandMark class="size-5 opacity-80" />
          <span>Red Queen -- evolution and learning, built from scratch.</span>
        </div>
        <div class="flex gap-5 sm:ml-auto">
          <NuxtLink v-for="link in NAV_LINKS.slice(1)" :key="link.to" :to="link.to" class="hover:text-fg">
            {{ link.label }}
          </NuxtLink>
        </div>
      </div>
    </footer>
  </div>
</template>
