import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import type { ReactElement } from 'react';

/** Render a page at a URL, with the providers it expects and no retries to wait through. */
export function renderAt(route: string, path: string, element: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>
        <Routes>
          <Route path={path} element={element} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

/** Answer one request shape with one body; anything else fails the test loudly. */
export function stubFetch(handler: (url: string, init?: RequestInit) => unknown) {
  return (url: string, init?: RequestInit) => {
    const value = handler(url, init);
    return Promise.resolve(
      new Response(JSON.stringify({ success: true, data: value, error: null, timestamp: 't' }), {
        headers: { 'content-type': 'application/json' },
      }),
    );
  };
}
