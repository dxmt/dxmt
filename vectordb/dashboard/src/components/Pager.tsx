import { count } from '../lib/format';

/**
 * Explicit paging only.
 *
 * No infinite scroll and no prefetch-the-next-page, because `records/list` is answered by
 * collecting every matching record, sorting the whole set by `created_at` and slicing — so a
 * page is proportional to the table, not to the page size. Anything that fetches a page the
 * user did not ask for is a full table scan they did not ask for.
 */
export function Pager({
  offset,
  limit,
  total,
  onOffset,
  busy,
}: {
  offset: number;
  limit: number;
  total: number;
  onOffset: (next: number) => void;
  busy: boolean;
}) {
  const last = Math.max(0, Math.floor((total - 1) / limit) * limit);
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + limit, total);

  return (
    <div className="pager">
      <button type="button" disabled={busy || offset === 0} onClick={() => onOffset(0)}>
        ⟨⟨
      </button>
      <button
        type="button"
        disabled={busy || offset === 0}
        onClick={() => onOffset(Math.max(0, offset - limit))}
      >
        prev
      </button>
      <span className="muted">
        {count(from)}–{count(to)} of {count(total)}
      </span>
      <button
        type="button"
        disabled={busy || offset + limit >= total}
        onClick={() => onOffset(offset + limit)}
      >
        next
      </button>
      <button type="button" disabled={busy || offset >= last} onClick={() => onOffset(last)}>
        ⟩⟩
      </button>
    </div>
  );
}
