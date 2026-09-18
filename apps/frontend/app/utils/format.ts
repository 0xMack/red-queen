// Small display formatters shared by the runs pages, home page, and watch widgets. Auto-imported
// (Nuxt scans app/utils/).

export function formatFitness(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "--"
  const abs = Math.abs(value)
  if (abs !== 0 && (abs >= 1e5 || abs < 1e-3)) return value.toExponential(2)
  return value.toFixed(digits)
}

export function formatSigned(value: number, digits = 2): string {
  return `${value >= 0 ? "+" : ""}${formatFitness(value, digits)}`
}

/** Seconds -> "2m 13s" / "1h 4m" / "850ms". */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "--"
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`
  const m = Math.floor(seconds / 60)
  if (m < 60) return `${m}m ${Math.round(seconds % 60)}s`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ${m % 60}m`
  return `${Math.floor(h / 24)}d ${h % 24}h`
}

/** Unix seconds -> "3 days ago". `now` is injectable so a table can tick every row off one clock. */
export function formatRelative(unixSeconds: number, now = Date.now() / 1000): string {
  const delta = now - unixSeconds
  if (delta < 0) return "just now"
  if (delta < 45) return "just now"
  if (delta < 3600) return `${Math.round(delta / 60)} min ago`
  if (delta < 86400) return `${Math.round(delta / 3600)} h ago`
  const days = Math.round(delta / 86400)
  if (days < 30) return `${days} day${days === 1 ? "" : "s"} ago`
  return new Date(unixSeconds * 1000).toLocaleDateString("en-US")
}

export function formatTimestamp(unixSeconds: number): string {
  // Fixed locale: SSR (Node) and the browser can default to different ones -> hydration mismatch.
  return new Date(unixSeconds * 1000).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function shortId(id: string): string {
  return id.slice(0, 8)
}
