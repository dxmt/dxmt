import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';

export default tseslint.config(
  { ignores: ['dist/**', 'node_modules/**'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['src/**/*.{ts,tsx}'],
    plugins: { 'react-hooks': reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
    },
  },
  {
    // Read-only is enforced by the type of `post`'s path parameter, which only works if every
    // request goes through that module. A stray `fetch` would route around the whole guarantee,
    // so it is a lint error rather than something to notice in review.
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/api/client.ts'],
    rules: {
      'no-restricted-globals': [
        'error',
        {
          name: 'fetch',
          message:
            'Requests go through src/api/client.ts. Its `post` accepts only the read-only ' +
            'paths, which is what keeps this console from being able to write.',
        },
      ],
    },
  },
);
