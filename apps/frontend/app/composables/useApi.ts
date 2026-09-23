type FetchOptions = NonNullable<Parameters<typeof $fetch>[1]>

// apis/backend as every page, component and store reaches it: `runtimeConfig.public.apiBase`
// (NUXT_PUBLIC_API_BASE). Call it where the base URL can still be read -- in setup, or at the top of a store
// action -- not after an `await`: a plain function has lost Nuxt's context by then.
export function useApi() {
  const base = useRuntimeConfig().public.apiBase as string

  /** `$fetch` against the backend: `path` is `/runs`, `/games/snake/leaderboard`, ... */
  function fetch<T>(path: string, options: FetchOptions = {}): Promise<T> {
    return $fetch<T>(path, { ...options, baseURL: base }) as Promise<T>
  }

  /** An absolute backend URL, for what `$fetch` doesn't do (an `EventSource`). */
  function url(path: string): string {
    return `${base}${path}`
  }

  return { fetch, url }
}
