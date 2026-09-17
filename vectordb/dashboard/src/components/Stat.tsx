export function Stat({
  label,
  value,
  title,
  mono,
}: {
  label: string;
  value: React.ReactNode;
  title?: string;
  mono?: boolean;
}) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd className={mono ? 'mono' : undefined} title={title}>
        {value}
      </dd>
    </div>
  );
}

export function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="card">
      <h3>{title}</h3>
      <dl>{children}</dl>
    </section>
  );
}
