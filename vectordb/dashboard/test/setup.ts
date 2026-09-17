import '@testing-library/jest-dom/vitest';

// The client derives the API base from the page's own URL, so the tests need a page.
Object.defineProperty(window, 'location', {
  value: new URL('http://localhost/dashboard/'),
  writable: true,
});
