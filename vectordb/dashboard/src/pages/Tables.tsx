import { Link } from 'react-router-dom';
import { ErrorBanner } from '../components/ErrorBanner';
import { useTables } from '../api/queries';
import { count } from '../lib/format';

/** A warning threshold, not a limit: listing a table's records scans all of it, every page. */
const BIG = 50_000;

export function Tables() {
  const { data, error, isLoading } = useTables();

  if (error) return <ErrorBanner error={error} />;

  return (
    <>
      <h2>Tables</h2>
      <p className="sub">
        {data ? `${count(data.total)} table${data.total === 1 ? '' : 's'}` : 'loading…'}
      </p>

      {isLoading ? null : (
        <table>
          <thead>
            <tr>
              <th>name</th>
              <th className="num">records</th>
              <th className="num">dim</th>
              <th>metric</th>
              <th>index</th>
              <th>full text</th>
            </tr>
          </thead>
          <tbody>
            {data?.tables.map((table) => (
              <tr key={table.id}>
                <td>
                  <Link to={`/tables/${encodeURIComponent(table.name)}`}>{table.name}</Link>
                  {table.record_count > BIG ? (
                    <span
                      className="badge warn"
                      style={{ marginLeft: 8 }}
                      title="Listing this table's records scans the whole table on every page."
                    >
                      large
                    </span>
                  ) : null}
                </td>
                <td className="num">{count(table.record_count)}</td>
                <td className="num">{table.dimension}</td>
                <td>{table.distance_metric}</td>
                <td className="mono">{table.index_type}</td>
                <td>{table.enable_full_text_search ? 'yes' : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {data && data.tables.length === 0 ? (
        <p className="note">No tables yet. This console cannot create one — use the API.</p>
      ) : null}
      <p className="note">
        Sorting and filtering this list would have to happen in the browser: the endpoint offers
        neither, only <span className="mono">limit</span> and <span className="mono">offset</span>.
      </p>
    </>
  );
}
