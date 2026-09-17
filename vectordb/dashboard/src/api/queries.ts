import { useQuery } from '@tanstack/react-query';
import { api } from './client';

/**
 * How often each thing is asked for, and why.
 *
 * `records/list` is the one that matters: the server collects every matching record and sorts
 * the whole table before slicing out a page, so a refetch on a tab switch is a full table scan
 * the user did not ask for. `staleTime: Infinity` and an explicit refresh, therefore -- and the
 * record browser is the reason `@tanstack/react-query` is a dependency at all.
 */
export const REFRESH = {
  /** Cheap and genuinely live. */
  health: 5_000,
  /** Cheap, but nothing on it changes second to second. */
  status: 15_000,
  /** One map lookup per table on the server; still, tables do not appear often. */
  tables: 30_000,
} as const;

export function useStatus() {
  return useQuery({
    queryKey: ['status'],
    queryFn: ({ signal }) => api.status(signal).then((r) => r.value),
    refetchInterval: REFRESH.status,
    refetchIntervalInBackground: false,
  });
}

export function useMetrics() {
  return useQuery({
    queryKey: ['metrics'],
    queryFn: ({ signal }) => api.metrics(signal).then((r) => r.value),
    refetchInterval: REFRESH.health,
    refetchIntervalInBackground: false,
  });
}

export function useTables(limit = 200, offset = 0) {
  return useQuery({
    queryKey: ['tables', limit, offset],
    queryFn: ({ signal }) => api.tables(limit, offset, signal).then((r) => r.value),
    refetchInterval: REFRESH.tables,
    refetchIntervalInBackground: false,
  });
}
