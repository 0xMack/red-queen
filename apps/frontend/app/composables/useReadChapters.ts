// Which chapters this browser has read to the end -- a per-viewer convenience for the map and the sidebar, so it
// lives in localStorage and everything still renders (as unread) when storage is unavailable. Module-level state, so
// every component on the page sees a chapter marked read the moment it happens.
const KEY = "redqueen.learn.read"
const read = ref<Set<string>>(new Set())
let loaded = false

function load() {
  if (loaded || !import.meta.client) return
  loaded = true
  try {
    const stored = JSON.parse(localStorage.getItem(KEY) ?? "[]")
    if (Array.isArray(stored)) read.value = new Set(stored.filter((s) => typeof s === "string"))
  } catch {
    // Private window, blocked storage, or a corrupt value: start empty.
  }
}

export function useReadChapters() {
  onMounted(load)

  function markRead(slug: string) {
    if (read.value.has(slug)) return
    read.value = new Set([...read.value, slug])
    try {
      localStorage.setItem(KEY, JSON.stringify([...read.value]))
    } catch {
      // Not persisted; still marked for this page view.
    }
  }

  return { read: readonly(read), isRead: (slug: string) => read.value.has(slug), markRead }
}
