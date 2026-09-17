import type { StorageHealthSnapshot } from '../api/types';

/**
 * Durability trouble, on every page rather than only on the overview.
 *
 * Someone reading records needs to know the data they are reading may not survive a restart,
 * and they will not have gone to a status page to find out. A read-only console is unusually
 * well placed to say so, because it is the only thing anyone opens when they are already
 * suspicious.
 *
 * `wal_enabled: false` is deliberately *not* a warning: embedded use and most tests run without
 * a WAL directory on purpose, and colouring that red teaches people to ignore the banner.
 */
export function DegradedBanner({ storage }: { storage: StorageHealthSnapshot | undefined }) {
  if (!storage) return null;

  const problems: { title: string; detail?: string | null }[] = [];

  if (!storage.ready) {
    problems.push({ title: 'The server reports it is not ready' });
  }
  if (storage.durability_degraded) {
    problems.push({
      title: 'Durability degraded — writes may not survive a restart',
      detail: storage.durability_error,
    });
  }
  if (storage.wal_enabled && !storage.wal_directory_accessible) {
    problems.push({ title: 'The WAL directory is not accessible' });
  }
  if (storage.snapshot_present && !storage.snapshot_manifest_valid) {
    problems.push({ title: 'The snapshot manifest does not validate' });
  }
  if (storage.snapshot_failures > 0) {
    problems.push({
      title: `${storage.snapshot_failures} snapshot ${storage.snapshot_failures === 1 ? 'failure' : 'failures'}`,
      detail: storage.last_snapshot_error,
    });
  }

  if (problems.length === 0) return null;

  return (
    <div className="banner bad">
      {problems.map((problem) => (
        <div key={problem.title}>
          <strong>{problem.title}</strong>
          {problem.detail ? <span className="detail">{problem.detail}</span> : null}
        </div>
      ))}
    </div>
  );
}
