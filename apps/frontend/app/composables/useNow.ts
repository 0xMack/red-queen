// A clock shared by server and client so relative times ("3 h ago") hydrate without a mismatch: the server's
// value is reused on the client, then refreshed after hydration. `tickSeconds` keeps it moving (a live table).
export function useNow(tickSeconds?: number) {
  const now = useState("clock:now", () => Date.now() / 1000)
  let timer: ReturnType<typeof setInterval> | null = null
  onMounted(() => {
    now.value = Date.now() / 1000
    if (tickSeconds) timer = setInterval(() => (now.value = Date.now() / 1000), tickSeconds * 1000)
  })
  onUnmounted(() => timer && clearInterval(timer))
  return now
}
