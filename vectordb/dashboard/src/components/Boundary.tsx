import { Component, type ErrorInfo, type ReactNode } from 'react';

/**
 * A render crash shows the crash, not a white page.
 *
 * Without this, one bad payload shape or one undefined field takes the whole console down to a
 * blank document with the message only in the browser's own console — and this is a tool people
 * open *because* something is already wrong, so a blank page is the least useful thing it could
 * do at the moment it matters most.
 *
 * Scoped to the page rather than the app, so the navigation survives and the reader can get
 * somewhere else without reloading.
 */
interface State {
  error: Error | null;
  stack: string | null;
}

export class Boundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null, stack: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ error, stack: info.componentStack ?? null });
  }

  render(): ReactNode {
    const { error, stack } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="banner bad" role="alert">
        <strong>This page crashed while rendering</strong>
        <span className="detail">{error.message}</span>
        {stack ? (
          <details>
            <summary>where</summary>
            <pre className="runlog">{stack.trim()}</pre>
          </details>
        ) : null}
        <p style={{ marginBottom: 0 }}>
          <button type="button" onClick={() => this.setState({ error: null, stack: null })}>
            Try again
          </button>
        </p>
      </div>
    );
  }
}
