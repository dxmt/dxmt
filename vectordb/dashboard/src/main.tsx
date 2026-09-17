import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { App } from './App';
import './styles/tokens.css';
import './styles/app.css';

/**
 * `basename` comes from Vite's `base`, so the router and the asset URLs cannot disagree about
 * where the app is mounted. A mismatch there produces a page that loads and then navigates to
 * a blank route, which is a confusing way to find out.
 */
const client = new QueryClient({
  defaultOptions: {
    queries: {
      // A console is opened *because* something looks wrong, so silently retrying an error
      // hides the thing the user came to see. One attempt, and show what happened.
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <BrowserRouter basename={import.meta.env.BASE_URL}>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
