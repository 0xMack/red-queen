// The site's top-level sections, shared by the header, the mobile menu and the footer. `also` lists other path
// prefixes that belong to a section (a champion on /watch is part of Games).
export interface NavLink {
  to: string
  label: string
  exact?: boolean
  also?: string[]
  blurb: string
}

export const NAV_LINKS: NavLink[] = [
  { to: "/games", label: "Games", also: ["/play", "/watch"], blurb: "Watch trained models play, then take them on." },
  { to: "/learn", label: "Learn", blurb: "The interactive textbook: every mechanism, running." },
  { to: "/runs", label: "Runs", blurb: "Every training run, live and recorded." },
]

export function isNavActive(link: NavLink, path: string): boolean {
  if (link.exact) return path === link.to
  return [link.to, ...(link.also ?? [])].some((p) => path.startsWith(p))
}
