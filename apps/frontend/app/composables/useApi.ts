type FetchOptions = NonNullable<Parameters<typeof $fetch>[1]>

// apis/backend as every page, component and store reaches it: `runtimeConfig.public.apiBase`
// (NUXT_PUBLIC_API_BASE). Call it where the base URL can still be read -- in setup, or at the top of a store
// action -- not after an `await`: a plain function has lost Nuxt's context by then.
//
// In `static` mode (NUXT_PUBLIC_API_MODE, the public site, docs/design/0019) there is no server: the base is the API's
// static copy (`redqueen publish site`), where every route the site reads is a file at its own path plus `.json`. Only
// reads exist there -- `live` is false, so pages hide what needs a server (run control, live streams), and anything
// else fails like a missing route.
export function useApi() {
  const config = useRuntimeConfig().public
  const base = config.apiBase as string
  const live = config.apiMode !== "static"

  /** `$fetch` against the backend: `path` is `/runs`, `/games/snake/leaderboard`, ... */
  function fetch<T>(path: string, options: FetchOptions = {}): Promise<T> {
    if (live) return $fetch<T>(path, { ...options, baseURL: base }) as Promise<T>
    const method = (options.method ?? "GET").toString().toUpperCase()
    if (method !== "GET") return Promise.reject(new Error(`${method} ${path}: the public site is read-only`))
    // The static copy holds each route's default answer: query parameters aren't part of a file's name.
    const { query: _query, params: _params, ...rest } = options as FetchOptions & { params?: unknown }
    return $fetch<T>(`${path.split("?")[0]}.json`, { ...rest, baseURL: base }) as Promise<T>
  }

  /**
   * A champion as a model package: exported on demand by a live backend, or (static) the export the site snapshot
   * ran ahead of time for each run's latest champion.
   */
  function packageChampion<T>(runId: string, ref: string): Promise<T> {
    const path = `/runs/${runId}/artifacts/${ref}/package`
    return live ? fetch<T>(path, { method: "POST" }) : fetch<T>(path)
  }

  /** An absolute backend URL, for what `$fetch` doesn't do (an `EventSource`). */
  function url(path: string): string {
    return `${base}${path}`
  }

  return { fetch, packageChampion, url, live }
}
