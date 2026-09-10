/**
 * Shared palette.
 *
 * Lives apart from the scene because react-refresh requires a component file to
 * export only components — exporting a constant alongside them breaks HMR.
 */

/** Agent colour by current action, as CSS strings. */
export const ACTION_COLORS: Record<string, string> = {
  idle: '#8a8a96',
  travel: '#f0f0f5',
  sleep: '#6b7fd7',
  eat: '#d78b4a',
  study: '#9e6bd7',
  work: '#d7d74a',
  socialize: '#d76b9e',
  exercise: '#4ad78b',
}
