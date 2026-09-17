/**
 * The crash path, which is the one nobody exercises until it happens.
 *
 * This console is opened because something is already wrong. A render error that takes it down
 * to a blank document, with the message only in the browser's own console, is the least useful
 * thing it could do at exactly the moment it matters most.
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Boundary } from '../src/components/Boundary';

function Explodes({ when }: { when: boolean }): React.ReactElement {
  if (when) throw new Error('payload_json was not what anyone expected');
  return <p>the page</p>;
}

describe('Boundary', () => {
  it('shows the error instead of a blank page', () => {
    // React logs the caught error itself; that noise is not the subject.
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <Boundary>
        <Explodes when />
      </Boundary>,
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/payload_json was not what anyone expected/)).toBeInTheDocument();
    quiet.mockRestore();
  });

  it('lets the page be retried without reloading the whole console', async () => {
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => {});
    const user = userEvent.setup();

    // The test decides when it stops failing, rather than counting renders: React renders a
    // failing subtree more than once while recovering, and in development renders twice
    // anyway, so a render counter makes this assert something about React's scheduling.
    let broken = true;
    function Flaky() {
      return <Explodes when={broken} />;
    }

    render(
      <Boundary>
        <Flaky />
      </Boundary>,
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();

    broken = false;
    await user.click(screen.getByRole('button', { name: /try again/i }));
    expect(screen.getByText('the page')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    quiet.mockRestore();
  });

  it('stays out of the way when nothing is wrong', () => {
    render(
      <Boundary>
        <Explodes when={false} />
      </Boundary>,
    );
    expect(screen.getByText('the page')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});
