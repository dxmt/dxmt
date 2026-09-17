import '@testing-library/jest-dom/vitest';

// The client derives the API base from the page's own URL, so the tests need a page.
Object.defineProperty(window, 'location', {
  value: new URL('http://localhost/dashboard/'),
  writable: true,
});

// jsdom has no canvas, and `getContext` there logs a "not implemented" page of stack trace on
// every render that touches one. Returning null instead is both quieter and more honest: a real
// browser can also refuse a context, and `VectorCell` has to draw nothing rather than throw
// when it does. Stubbing it here keeps that path exercised without the noise.
HTMLCanvasElement.prototype.getContext = () => null;
