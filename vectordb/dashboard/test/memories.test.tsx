/**
 * The Memories page: the one place this console writes, so the writes are what is pinned.
 *
 * A correction must keep history by default -- a new memory that `supersedes` the old one, not
 * an overwrite -- and forgetting must not happen on the first click. Both are properties that
 * look identical in a screenshot whether they hold or not.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Memories } from '../src/pages/Memories';
import { MemoryDetail } from '../src/pages/MemoryDetail';
import { renderAt, stubFetch } from './render';

const now = Math.floor(Date.now() / 1000);

const tea = {
  id: 'm1',
  text: 'Prefers green tea',
  kind: 'preference',
  tags: ['drinks'],
  source: 'claude',
  at: now - 120,
  project: 'kitchen',
};
const shanghai = {
  id: 'm0',
  text: 'Lives in Shanghai',
  kind: 'fact',
  tags: [],
  at: now - 86_400 * 3,
  superseded_by: 'm9',
};

interface Call {
  url: string;
  method: string;
  body: unknown;
}

/** Record every request, and answer by method and path. */
function serve(answer: (call: Call) => unknown) {
  const calls: Call[] = [];
  vi.stubGlobal(
    'fetch',
    stubFetch((url, init) => {
      const call = {
        url,
        method: init?.method ?? 'GET',
        body: typeof init?.body === 'string' ? JSON.parse(init.body) : undefined,
      };
      calls.push(call);
      return answer(call);
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

describe('the memory list', () => {
  it('shows text, kind, tags, source and age, and marks a superseded memory', async () => {
    const calls = serve(() => ({ memories: [tea, shanghai], total: 2 }));
    renderAt('/memories', '/memories', <Memories />);

    await screen.findByText('Prefers green tea');
    expect(screen.getByText('preference', { selector: '.badge' })).toBeInTheDocument();
    expect(screen.getByText('drinks')).toBeInTheDocument();
    expect(screen.getByText('claude')).toBeInTheDocument();
    expect(screen.getByText('2 min ago')).toBeInTheDocument();
    expect(screen.getByText('superseded')).toHaveAttribute('title', expect.stringContaining('m9'));
    expect(calls[0]).toMatchObject({ method: 'GET', url: '/api/v1/memories?limit=25&offset=0' });
  });

  it('filters by kind through the server, not in the browser', async () => {
    const user = userEvent.setup();
    const calls = serve(() => ({ memories: [tea], total: 1 }));
    renderAt('/memories', '/memories', <Memories />);
    await screen.findByText('Prefers green tea');

    await user.selectOptions(screen.getByRole('combobox', { name: /kind/i }), 'preference');
    await waitFor(() =>
      expect(calls.at(-1)?.url).toBe('/api/v1/memories?kind=preference&limit=25&offset=0'),
    );
  });

  it('explains, when empty, that memory needs a model and how agents write one', async () => {
    serve(() => ({ memories: [], total: 0 }));
    renderAt('/memories', '/memories', <Memories />);

    await screen.findByText('No memories yet');
    expect(screen.getByText(/needs an embedding model on the server/i)).toBeInTheDocument();
    expect(screen.getByText('remember')).toBeInTheDocument();
    expect(screen.getByText(/POST \/api\/v1\/memories/)).toBeInTheDocument();
  });
});

describe('recall', () => {
  it('searches with recall, and asks for superseded memories only when ticked', async () => {
    const user = userEvent.setup();
    const calls = serve((call) =>
      call.url.endsWith('/recall') ? { memories: [] } : { memories: [tea], total: 1 },
    );
    renderAt('/memories', '/memories', <Memories />);
    await screen.findByText('Prefers green tea');

    await user.type(screen.getByRole('searchbox', { name: /search memories/i }), 'tea');
    await user.click(screen.getByRole('button', { name: 'Recall' }));
    await waitFor(() => expect(calls.at(-1)?.url).toBe('/api/v1/memories/recall'));
    expect(calls.at(-1)).toMatchObject({ method: 'POST', body: { query: 'tea', limit: 25 } });
    expect(calls.at(-1)?.body).not.toHaveProperty('include_superseded');

    await user.click(screen.getByRole('checkbox', { name: /include superseded/i }));
    await waitFor(() =>
      expect(calls.at(-1)?.body).toMatchObject({ query: 'tea', include_superseded: true }),
    );
  });

  it('shows similarity when the server sends it, and the score otherwise', async () => {
    serve(() => ({
      memories: [
        { ...tea, score: 1.0, similarity: 0.8123 },
        { ...shanghai, id: 'm2', superseded_by: undefined, score: 0.5 },
      ],
    }));
    renderAt('/memories?q=tea', '/memories', <Memories />);

    await screen.findByText('similarity 0.812');
    expect(screen.getByText('score 0.500')).toBeInTheDocument();
    // The rank is not shown beside a similarity it would be mistaken for.
    expect(screen.queryByText('score 1.000')).not.toBeInTheDocument();
  });
});

describe('a memory', () => {
  const detail = (memory: object, onWrite: (call: Call) => unknown = () => ({})) =>
    serve((call) =>
      call.method === 'GET' ? { memories: [memory, shanghai], total: 2 } : onWrite(call),
    );

  it('shows every field, metadata included', async () => {
    detail(tea);
    renderAt('/memories/m1', '/memories/:id', <MemoryDetail />);

    await screen.findByRole('heading', { name: 'Prefers green tea' });
    expect(screen.getByText(/"project"/)).toBeInTheDocument();
    expect(screen.getByText(/"kitchen"/)).toBeInTheDocument();
  });

  it('links a superseded memory to what replaced it, and offers no correction', async () => {
    detail(shanghai);
    renderAt('/memories/m0', '/memories/:id', <MemoryDetail />);

    await screen.findByRole('heading', { name: 'Lives in Shanghai' });
    for (const link of screen.getAllByRole('link', { name: 'm9' })) {
      expect(link).toHaveAttribute('href', '/memories/m9');
    }
    expect(screen.queryByRole('button', { name: /save correction/i })).not.toBeInTheDocument();
  });

  it('corrects by a new memory that supersedes the old one, carrying source and metadata', async () => {
    const user = userEvent.setup();
    const calls = detail(tea, () => ({ id: 'm2', superseded: ['m1'] }));
    renderAt('/memories/m1', '/memories/:id', <MemoryDetail />);

    const text = await screen.findByRole('textbox', { name: 'Memory text' });
    await user.clear(text);
    await user.type(text, 'Prefers oolong tea');
    await user.click(screen.getByRole('button', { name: 'Save correction' }));

    await waitFor(() => expect(calls.some((c) => c.method === 'POST')).toBe(true));
    const write = calls.find((c) => c.method === 'POST')!;
    expect(write.url).toBe('/api/v1/memories');
    expect(write.body).toEqual({
      text: 'Prefers oolong tea',
      kind: 'preference',
      tags: ['drinks'],
      source: 'claude',
      metadata: { project: 'kitchen' },
      supersedes: ['m1'],
    });
  });

  it('fixes a typo in place only when asked, by the same id', async () => {
    const user = userEvent.setup();
    const calls = detail(tea, () => ({ id: 'm1', superseded: [] }));
    renderAt('/memories/m1', '/memories/:id', <MemoryDetail />);

    const text = await screen.findByRole('textbox', { name: 'Memory text' });
    await user.clear(text);
    await user.type(text, 'Prefers green tea.');
    await user.click(screen.getByRole('radio', { name: /fix a typo in place/i }));
    await user.click(screen.getByRole('button', { name: 'Fix in place' }));

    await waitFor(() => expect(calls.some((c) => c.method === 'POST')).toBe(true));
    const write = calls.find((c) => c.method === 'POST')!;
    expect(write.body).toMatchObject({ id: 'm1', text: 'Prefers green tea.' });
    expect(write.body).not.toHaveProperty('supersedes');
  });

  it('forgets only after a confirmation that names the memory', async () => {
    const user = userEvent.setup();
    const calls = detail(tea, () => ({ forgotten: 1 }));
    renderAt('/memories/m1', '/memories/:id', <MemoryDetail />);
    await screen.findByRole('heading', { name: 'Prefers green tea' });

    await user.click(screen.getByRole('button', { name: 'Forget…' }));
    const dialog = screen.getByRole('alertdialog');
    expect(dialog).toHaveTextContent('“Prefers green tea”');
    expect(dialog).toHaveTextContent(/no undo/i);
    // Opening the dialog is not consent, and the safe button is where Enter lands.
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false);
    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false);

    await user.click(screen.getByRole('button', { name: 'Forget…' }));
    await user.click(screen.getByRole('button', { name: 'Forget permanently' }));
    await waitFor(() =>
      expect(calls.filter((c) => c.method === 'DELETE').map((c) => c.url)).toEqual([
        '/api/v1/memories/m1',
      ]),
    );
  });
});
