// The dedicated /play/[game].vue page's session: window-scoped keyboard capture, since that page
// owns the whole viewport. LiveSnakeDemo.vue (embedded in Learn/landing pages) needs
// element-scoped input instead -- rather than parameterize this composable for both, it composes
// useSnakeSession()/createHeadingTracker() directly. Two concrete, genuinely different wiring
// needs; forcing a shared "attach to window or element" option in here would be a speculative
// abstraction for a two-case scenario (docs/CODING_GUIDELINES.md's "pluggable only at proven
// extension points").
export function usePlaySession() {
  const session = useSnakeSession()
  const heading = createHeadingTracker()

  function onKeydown(event: KeyboardEvent) {
    const action = heading.translate(event.key)
    if (action !== null) session.sendInput(action)
  }

  function start() {
    heading.reset()
    session.start()
  }

  function restart() {
    heading.reset()
    session.restart()
  }

  onMounted(() => window.addEventListener("keydown", onKeydown))
  onUnmounted(() => {
    window.removeEventListener("keydown", onKeydown)
    session.stop()
  })

  return { ...session, start, restart }
}
