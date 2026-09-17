import { useState } from 'react';

export function CopyButton({ text, label = 'copy' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className="ghost"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(
          () => {
            setDone(true);
            setTimeout(() => setDone(false), 1200);
          },
          // A clipboard write can be refused (insecure origin, denied permission). Saying so
          // beats a button that appears to do nothing.
          () => window.prompt('Copy this:', text),
        );
      }}
    >
      {done ? 'copied' : label}
    </button>
  );
}
