import { Link } from 'react-router-dom';
import type { RecallReport } from './recall';

/**
 * What a recall run actually found, which is more than one number.
 *
 * A mean of 0.95 made of "five per cent of queries are catastrophic" is a completely different
 * index from one made of "every query misses one neighbour", and the mean cannot tell them
 * apart — hence the spread and the histogram. Two runs both at 0.90 are also very different if
 * one substituted a neighbour at similarity 0.998 and the other at 0.61 — hence the score
 * ratio. And recall without the latency it bought is half a result, so the speedup sits beside
 * it rather than in a footnote.
 */
export function Report({ table, reports }: { table: string; reports: RecallReport[] }) {
  if (reports.length === 0) return null;
  const k = reports[0]!.k;
  const groundTruth = reports[0]!.groundTruth;

  return (
    <>
      <table className="report">
        <thead>
          <tr>
            <th>mode</th>
            <th className="num">{groundTruth ? `recall@${k}` : `agreement@${k}`}</th>
            <th className="num">@1</th>
            <th className="num">score ratio</th>
            <th className="num">ms</th>
            <th className="num">vs truth</th>
            <th>spread</th>
          </tr>
        </thead>
        <tbody>
          {reports.map((report) => (
            <tr key={report.mode}>
              <td className="mono">{report.mode}</td>
              <td className="num">
                <b>{report.meanRecall.toFixed(3)}</b>
                <span className="muted"> ±{report.stdevRecall.toFixed(3)}</span>
              </td>
              <td className="num">{report.recallAt1.toFixed(2)}</td>
              <td
                className="num"
                title="The approximate run's summed similarity over the exact run's. Near 1.0 means the misses were near-ties."
              >
                {report.meanScoreRatio.toFixed(4)}
              </td>
              <td className="num">{report.medianAnnMs}</td>
              <td className="num" title="Median exhaustive time divided by median approximate time.">
                {report.medianAnnMs > 0
                  ? `${(report.medianExactMs / report.medianAnnMs).toFixed(1)}×`
                  : '—'}
              </td>
              <td>
                <Histogram report={report} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="worst">
        {reports.map((report) =>
          report.worst.filter((p) => p.recall < 1).length === 0 ? null : (
            <section key={report.mode}>
              <h4>
                worst under <span className="mono">{report.mode}</span>
              </h4>
              <ul>
                {report.worst
                  .filter((probe) => probe.recall < 1)
                  .map((probe) => (
                    <li key={probe.id}>
                      <Link
                        to={`/tables/${encodeURIComponent(table)}/records/${encodeURIComponent(probe.id)}`}
                      >
                        {probe.id}
                      </Link>
                      <span className="muted">
                        {' '}
                        {probe.recall.toFixed(2)} — missed {probe.missing.slice(0, 3).join(', ')}
                        {probe.missing.length > 3 ? ` +${probe.missing.length - 3}` : ''}
                      </span>
                    </li>
                  ))}
              </ul>
            </section>
          ),
        )}
      </div>

      {groundTruth ? null : (
        <p className="warn-text">
          These are agreements with <span className="mono">accurate</span>, not recall.
          <span className="mono"> accurate</span> is 4× the traversal budget and about 0.98
          itself, so every figure above is an upper bound and is blind to the neighbours both
          modes miss.
        </p>
      )}
    </>
  );
}

/** Per-point recall in tenths. The shape says what the mean cannot. */
function Histogram({ report }: { report: RecallReport }) {
  const peak = Math.max(...report.histogram, 1);
  return (
    <span className="hist" title="Per-query recall, 0.0 on the left to 1.0 on the right">
      {report.histogram.map((n, bucket) => (
        <i
          key={bucket}
          style={{ height: `${Math.round((n / peak) * 14) + 1}px` }}
          className={n === 0 ? 'empty' : bucket === 10 ? 'full' : ''}
          title={`${n} ${n === 1 ? 'query' : 'queries'} at ${(bucket / 10).toFixed(1)}`}
        />
      ))}
    </span>
  );
}
