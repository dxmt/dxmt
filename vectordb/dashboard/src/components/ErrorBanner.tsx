import { ApiFailure } from '../api/envelope';

/**
 * What went wrong, said in the terms the user can act on.
 *
 * The five failure shapes need five different sentences. A network rejection is ambiguous in
 * the browser -- the server being down and the browser blocking the request are the same
 * `TypeError` -- so the message names both rather than guessing; and the extractor rejection,
 * which is what an older server answers when asked for a field it does not know, says that it
 * never reached a handler, because "the server rejected this" and "the server does not have
 * this" call for very different next steps.
 */
export function ErrorBanner({ error }: { error: unknown }) {
  if (error instanceof ApiFailure) {
    const { failure } = error;
    switch (failure.kind) {
      case 'network':
        return (
          <Banner title="Cannot reach the server">
            It is not answering, or the browser blocked the request. Those look identical from
            here. <span className="detail">{failure.message}</span>
          </Banner>
        );
      case 'envelope':
        return (
          <Banner title={failure.error.code} kind={error.isFeatureDisabled ? 'warn' : 'bad'}>
            <span className="detail">{failure.error.message}</span>
          </Banner>
        );
      case 'text':
        return (
          <Banner title={`HTTP ${failure.status} — the body was refused before any handler ran`}>
            Usually this means the request carried a field or a value this build does not know
            about. <span className="detail">{failure.text.slice(0, 400)}</span>
          </Banner>
        );
      case 'rawJson':
        return (
          <Banner title={`HTTP ${failure.status}`}>
            <span className="detail">{JSON.stringify(failure.body).slice(0, 400)}</span>
          </Banner>
        );
      case 'parse':
        return (
          <Banner title="The response was not in the expected shape">
            <span className="detail">{failure.text.slice(0, 400)}</span>
          </Banner>
        );
    }
  }
  return <Banner title="Something went wrong">{String(error)}</Banner>;
}

function Banner({
  title,
  kind = 'bad',
  children,
}: {
  title: string;
  kind?: 'bad' | 'warn';
  children: React.ReactNode;
}) {
  return (
    <div className={`banner ${kind}`}>
      <strong>{title}</strong>
      {children}
    </div>
  );
}
