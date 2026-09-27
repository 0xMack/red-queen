<script setup lang="ts">
// A lab's control row: Train (or Train again), Pause/Resume while it runs, a training speed, and the lab's own
// knobs in the default slot. `status` is the lab's (useRlLab / useSelfPlayLab); nothing here knows the algorithm.
const props = withDefaults(
  defineProps<{
    status: "idle" | "running" | "paused" | "done" | "error"
    disabled?: boolean
    speeds?: readonly { label: string; stepsPerSecond: number }[]
    trainLabel?: string
    againLabel?: string
    hint?: string | false
  }>(),
  { disabled: false, speeds: () => [], trainLabel: "Train", againLabel: "Train again", hint: undefined },
)
const emit = defineEmits<{ train: []; pause: []; resume: [] }>()
const speed = defineModel<number>("speed")
const started = computed(() => props.status !== "idle")
const active = computed(() => props.status === "running" || props.status === "paused")
</script>

<template>
  <div>
    <div class="flex flex-wrap items-center gap-2">
      <button v-if="status === 'running'" class="btn-ghost btn-sm" @click="emit('pause')">Pause</button>
      <button v-else-if="status === 'paused'" class="btn-primary btn-sm" @click="emit('resume')">Resume</button>
      <button :class="active ? 'btn-ghost btn-sm' : 'btn-accent btn-sm'" :disabled="disabled" @click="emit('train')">
        {{ started ? againLabel : trainLabel }}
      </button>
      <UiSegmented
        v-if="speeds.length && speed !== undefined"
        v-model="speed"
        class="ml-auto"
        :options="speeds.map((s) => ({ value: s.stepsPerSecond, label: s.label }))"
        aria-label="Training speed"
      />
    </div>
    <div v-if="$slots.default" class="mt-4 space-y-2.5">
      <slot />
    </div>
    <p v-if="hint !== false" class="mt-2 text-[11px] text-fg-subtle">
      {{ hint ?? `Changes apply when you press ${started ? againLabel : trainLabel}.` }}
    </p>
  </div>
</template>
