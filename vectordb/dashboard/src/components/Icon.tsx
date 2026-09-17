/**
 * The handful of icons this console needs, inline.
 *
 * Hand-drawn rather than an icon font or a package: ten glyphs is about sixty lines, where
 * `@tabler/icons-webfont` is a megabyte of glyphs for the other nine hundred and ninety, and
 * every one of them would be compiled into the server binary. They also have to be inline
 * rather than fetched, because the page's CSP is `default-src 'self'` and the console has to
 * work on a host with no route to the internet.
 *
 * They exist for the collapsed sidebar, where the label is gone and the icon is the only thing
 * left to recognise a destination by — so each one is drawn to be legible at 18px and to say
 * something about its page rather than being generically decorative.
 */
export type IconName =
  | 'overview'
  | 'tables'
  | 'records'
  | 'search'
  | 'visualize'
  | 'quality'
  | 'collapse';

const PATHS: Record<IconName, React.ReactNode> = {
  // A gauge: the health page.
  overview: (
    <>
      <path d="M3 13a9 9 0 0 1 18 0" />
      <path d="M12 13l4-3.5" />
      <circle cx="12" cy="13" r="1.4" />
    </>
  ),
  // Stacked rows with a header: a table.
  tables: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M3 9h18M9 9v11" />
    </>
  ),
  // Rows of differing length: a list of records.
  records: (
    <>
      <path d="M4 6h16M4 12h11M4 18h14" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="6" />
      <path d="m20 20-4.3-4.3" />
    </>
  ),
  // Scattered points, which is literally what the page draws.
  visualize: (
    <>
      <path d="M4 20V4M4 20h16" />
      <circle cx="9" cy="14" r="1.3" />
      <circle cx="13" cy="9" r="1.3" />
      <circle cx="17" cy="13" r="1.3" />
      <circle cx="11" cy="17" r="1.3" />
    </>
  ),
  // A target with a hit: measuring how close the index gets.
  quality: (
    <>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="3.4" />
      <path d="M12 4v2M12 18v2M4 12h2M18 12h2" />
    </>
  ),
  collapse: (
    <>
      <path d="m14 7-5 5 5 5" />
      <path d="M19 5v14" />
    </>
  ),
};

export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {PATHS[name]}
    </svg>
  );
}
