/**
 * Collapsing changes what is drawn, never what is reachable.
 *
 * That is the property worth a test. A sidebar that unmounts its links when collapsed looks
 * identical and quietly removes four destinations from the keyboard — and nobody notices,
 * because the people who would notice are not the people who build it.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { Sidebar } from '../src/components/Sidebar';

const draw = () =>
  render(
    <MemoryRouter>
      <Sidebar />
    </MemoryRouter>,
  );

describe('the sidebar', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.unstubAllGlobals();
  });

  it('keeps every link reachable when collapsed', async () => {
    const user = userEvent.setup();
    draw();
    const before = screen.getAllByRole('link').map((a) => a.getAttribute('href'));

    await user.click(screen.getByRole('button', { name: /collapse/i }));

    const after = screen.getAllByRole('link').map((a) => a.getAttribute('href'));
    expect(after).toEqual(before);
    // And each still says what it is, because the icon on its own is a guess.
    for (const link of screen.getAllByRole('link')) {
      expect(link.getAttribute('title')).toBeTruthy();
    }
  });

  it('says which state it is in, for anyone not looking at it', async () => {
    const user = userEvent.setup();
    draw();
    const toggle = screen.getByRole('button');
    expect(toggle).toHaveAttribute('aria-expanded', 'true');

    await user.click(toggle);
    expect(screen.getByRole('button')).toHaveAttribute('aria-expanded', 'false');
  });

  it('remembers the choice', async () => {
    const user = userEvent.setup();
    const first = draw();
    await user.click(screen.getByRole('button', { name: /collapse/i }));
    first.unmount();

    draw();
    expect(screen.getByRole('button')).toHaveAttribute('aria-expanded', 'false');
  });

  it('still renders when storage refuses, which it does in a private window', () => {
    // `localStorage` throws rather than returning null there, and a sidebar preference is not
    // worth a blank console.
    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('denied');
      },
      setItem: () => {
        throw new Error('denied');
      },
    });

    draw();
    expect(screen.getAllByRole('link').length).toBeGreaterThan(0);
    expect(screen.getByRole('button')).toHaveAttribute('aria-expanded', 'true');
  });
});
