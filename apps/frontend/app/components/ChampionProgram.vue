<script setup lang="ts">
// A linear-GP champion's actual program, as register-machine instructions, with structural introns
// (instructions that can't affect the output register) dimmed -- see utils/linearProgram.ts.
const props = defineProps<{ runId: string; championRef: string }>()

const api = useApi()
const raw = ref<string | null>(null)
const failed = ref(false)

watch(
  () => props.championRef,
  async (ref) => {
    failed.value = false
    try {
      raw.value = await api.fetch<string>(`/runs/${props.runId}/artifacts/${ref}`, { responseType: "text" })
    } catch {
      failed.value = true
    }
  },
  { immediate: true },
)

const program = computed(() => (raw.value ? parseLinearProgram(raw.value) : null))
</script>

<template>
  <div>
    <p v-if="failed" class="text-sm text-fg-subtle">Couldn't load this champion's artifact.</p>
    <template v-else-if="program">
      <div class="mb-3 flex flex-wrap gap-2 text-[11px]">
        <span class="chip">{{ program.instructions.length }} instructions</span>
        <span class="chip text-life-300">{{ program.effectiveCount }} effective</span>
        <span class="chip">{{ program.numRegisters }} registers · {{ program.numInputs }} input</span>
        <span class="chip">output: r0</span>
      </div>
      <ol class="overflow-hidden rounded-lg border border-line bg-sunken font-mono text-[13px]">
        <li
          v-for="(instr, i) in program.instructions"
          :key="i"
          class="flex items-center gap-4 border-b border-line/50 px-3 py-1.5 last:border-0"
          :class="instr.effective ? 'text-fg' : 'text-fg-subtle/60 line-through decoration-fg-subtle/30'"
        >
          <span class="w-5 text-right text-[11px] text-fg-subtle">{{ i }}</span>
          <span>{{ instr.text }}</span>
          <span v-if="!instr.effective" class="ml-auto text-[10px] tracking-wide text-fg-subtle uppercase no-underline">intron</span>
        </li>
      </ol>
      <p class="mt-2 text-xs text-fg-subtle">
        Struck-through instructions are structural introns -- they never influence r0, so evolution is
        free to mutate them without changing behavior.
      </p>
    </template>
    <pre v-else-if="raw" class="overflow-x-auto rounded-lg border border-line bg-sunken p-3 text-xs text-fg-muted">{{ raw }}</pre>
    <div v-else class="h-40 animate-pulse rounded-lg bg-raised" />
  </div>
</template>
