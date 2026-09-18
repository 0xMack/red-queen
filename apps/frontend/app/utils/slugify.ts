/** Heading text -> anchor id. Shared by learn.vue (assigns ids to chapter h2s) and the Learn search index. */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
}
