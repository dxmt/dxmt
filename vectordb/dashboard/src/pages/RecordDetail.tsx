import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { normalizeRecord, vectorFor } from '../api/normalize';
import { ErrorBanner } from '../components/ErrorBanner';
import { JsonBlock } from '../components/JsonBlock';
import { RequestPreview } from '../components/RequestPreview';
import { CopyButton } from '../components/CopyButton';
import { Card, Stat } from '../components/Stat';
import { since } from '../lib/format';

/**
 * One record, in full — the only view that always has the vector.
 *
 * `GET /records/{id}` reads through to the row file, so it answers with the vector whatever the
 * storage mode. That makes this the place to copy one from, and (once the search inspector
 * lands) the place to send one to it from.
 */
export function RecordDetail() {
  const { name = '', id = '' } = useParams();

  const query = useQuery({
    queryKey: ['record', name, id],
    queryFn: async ({ signal }) => {
      const response = await api.record(name, id, signal);
      return { record: normalizeRecord(name, response.value), ms: response.ms, wire: response.bytes };
    },
  });

  if (query.error) return <ErrorBanner error={query.error} />;
  const record = query.data?.record;
  const vector = vectorFor(name, id);

  return (
    <>
      <h2>{id}</h2>
      <p className="sub">
        in <Link to={`/tables/${encodeURIComponent(name)}/records`}>{name}</Link>
      </p>

      <RequestPreview
        method="GET"
        path={`/api/v1/tables/${encodeURIComponent(name)}/records/${encodeURIComponent(id)}`}
        ms={query.data?.ms}
        wire={query.data?.wire}
      />

      {record ? (
        <div className="cards">
          <Card title="Record">
            <Stat label="dimensions" value={record.dim} />
            <Stat label="‖v‖" value={record.norm.toFixed(6)} mono />
            <Stat label="version" value={record.version ?? '—'} />
            <Stat label="status" value={record.status ?? '—'} />
            <Stat
              label="created"
              value={since(record.createdAt)}
              title={record.createdAt ?? undefined}
            />
            <Stat
              label="updated"
              value={since(record.updatedAt)}
              title={record.updatedAt ?? undefined}
            />
          </Card>

          <section className="card">
            <h3>Payload</h3>
            {record.payloadError ? (
              <>
                <p className="err">payload_json did not parse: {record.payloadError}</p>
                <JsonBlock raw={record.payloadRaw ?? ''} />
              </>
            ) : (
              <JsonBlock value={record.payload} />
            )}
          </section>
        </div>
      ) : null}

      {vector && vector.length > 0 ? (
        <section className="card" style={{ marginTop: 16 }}>
          <h3>Vector</h3>
          <div className="jsonblock-tools">
            <CopyButton text={JSON.stringify(Array.from(vector))} label="copy as JSON" />
          </div>
          <pre className="vecdump">
            {Array.from(vector)
              .map((value, index) => `${String(index).padStart(4)}  ${value.toFixed(6)}`)
              .join('\n')}
          </pre>
        </section>
      ) : null}
    </>
  );
}
