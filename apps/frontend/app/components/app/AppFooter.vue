<script setup lang="ts">
import { NAV_LINKS } from "~/data/nav"
</script>

<template>
  <footer class="mt-32 border-t border-line">
    <div class="mx-auto grid max-w-[1600px] gap-10 px-4 py-12 sm:px-6 md:grid-cols-[minmax(0,1.4fr)_repeat(3,minmax(0,1fr))] lg:px-8">
      <div>
        <div class="flex items-center gap-2.5">
          <BrandMark class="size-6" />
          <span class="font-display text-xl">Red Queen</span>
        </div>
        <p class="mt-3 max-w-sm font-display text-lg leading-snug text-fg-muted italic">
          “It takes all the running you can do, to keep in the same place.”
        </p>
        <p class="mt-2 text-xs text-fg-subtle">Evolution and learning, built from scratch -- and run in your browser.</p>
      </div>
      <NuxtLink v-for="link in NAV_LINKS" :key="link.to" :to="link.to" class="group">
        <p class="label group-hover:text-queen-300">{{ link.label }}</p>
        <p class="mt-2 text-sm text-fg-muted transition group-hover:text-fg">{{ link.blurb }}</p>
      </NuxtLink>
    </div>
  </footer>
</template>
