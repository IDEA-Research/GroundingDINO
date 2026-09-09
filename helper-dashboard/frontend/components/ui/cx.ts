// Tiny class-name joiner (no dependency).
export function cx(
  ...parts: Array<string | false | null | undefined | 0>
): string {
  return parts.filter(Boolean).join(" ");
}
