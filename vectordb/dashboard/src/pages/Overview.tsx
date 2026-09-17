import { Card, Stat } from '../components/Stat';
import { DegradedBanner } from '../components/DegradedBanner';
import { ErrorBanner } from '../components/ErrorBanner';
import { useMetrics, useStatus } from '../api/queries';
import { count, since } from '../lib/format';

/**
 * The two questions a status page should answer and usually does not: is it healthy, and how
 * much would a restart have to replay?
 *
 * Both numbers below are derived rather than served. `wal_next_sequence -
 * snapshot_wal_next_sequence` is the size of the replay window -- the operational meaning of
 * the two snapshot fields, which the API exposes separately and never subtracts. And
 * `resident_table_count / table_count` is how much of the database is actually in memory, which
 * matters here because tables are evicted when cold and the first query against one pays to
 * bring it back.
 */
export function Overview() {
  const status = useStatus();
  const metrics = useMetrics();

  if (status.error) return <ErrorBanner error={status.error} />;

  const storage = status.data?.persistence ?? metrics.data?.storage;
  const ops = metrics.data?.ongoing_operations;

  const replay =
    storage && storage.snapshot_present
      ? storage.wal_next_sequence - storage.snapshot_wal_next_sequence
      : storage?.wal_next_sequence;

  return (
    <>
      <h2>Overview</h2>
      <p className="sub">
        {status.data ? `dxmt-vectordb ${status.data.version}` : 'connecting…'}
      </p>

      <DegradedBanner storage={storage} />

      <div className="cards">
        <Card title="Server">
          <Stat
            label="status"
            value={
              <span className={`badge ${status.data?.status === 'ready' ? 'ok' : 'warn'}`}>
                {status.data?.status ?? '…'}
              </span>
            }
          />
          <Stat label="tables" value={count(status.data?.storage.table_count ?? 0)} />
          <Stat label="records" value={count(status.data?.storage.total_records ?? 0)} />
          <Stat
            label="resident"
            value={
              storage ? `${storage.resident_table_count} / ${storage.table_count} tables` : '—'
            }
            title="Cold tables are evicted; the next query against one pays to bring it back."
          />
        </Card>

        <Card title="In flight">
          <Stat label="searches" value={count(ops?.searches ?? 0)} />
          <Stat label="inserts" value={count(ops?.inserts ?? 0)} />
          <Stat label="deletes" value={count(ops?.deletes ?? 0)} />
          <Stat label="table operations" value={count(ops?.table_operations ?? 0)} />
          <Stat label="completed" value={count(metrics.data?.completed_operations ?? 0)} />
          <Stat
            label="index building"
            value={
              <span className={`badge ${metrics.data?.index_status.building ? 'warn' : 'ok'}`}>
                {metrics.data?.index_status.building ? 'rebuilding' : 'no'}
              </span>
            }
            title="While an index is rebuilding, query timings are not representative."
          />
          <Stat label="recent errors" value={count(metrics.data?.errors.recent_count ?? 0)} />
        </Card>

        <Card title="Durability">
          <Stat
            label="write-ahead log"
            value={
              <span className="badge">{storage?.wal_enabled ? 'on' : 'off (in memory)'}</span>
            }
          />
          <Stat label="WAL sequence" value={count(storage?.wal_next_sequence ?? 0)} mono />
          <Stat
            label="to replay on restart"
            value={replay === undefined ? '—' : `${count(replay)} entries`}
            title="wal_next_sequence minus the snapshot's — how much of the log a restart reads."
          />
          <Stat
            label="snapshot"
            value={
              <span className={`badge ${storage?.snapshot_present ? 'ok' : 'warn'}`}>
                {storage?.snapshot_present ? 'present' : 'none yet'}
              </span>
            }
          />
          <Stat
            label="last snapshot"
            value={since(storage?.last_snapshot_at ?? null)}
            title={storage?.last_snapshot_at ?? undefined}
          />
          <Stat
            label="last WAL sync"
            value={since(storage?.last_wal_sync_at ?? null)}
            title={storage?.last_wal_sync_at ?? undefined}
          />
        </Card>
      </div>

      <p className="note">
        This console only reads. The server itself has no authentication — anything that can
        reach this port can also write to it.
      </p>
    </>
  );
}
