import { CopyButton } from './CopyButton';

/**
 * Pretty-printed JSON with the structure coloured in.
 *
 * Thirty-odd lines of regex rather than a JSON-view component, and deliberately so: for a
 * console whose job is showing what the server returned, raw formatted JSON is the better
 * artifact, not the worse one. A tree widget that collapses, reorders or re-types values is
 * showing its interpretation of the response.
 */
export function JsonBlock({ value, raw, copy = true }: { value?: unknown; raw?: string; copy?: boolean }) {
  const text = raw ?? JSON.stringify(value, null, 2) ?? 'null';
  return (
    <div className="jsonblock">
      {copy ? <div className="jsonblock-tools"><CopyButton text={text} /></div> : null}
      <pre dangerouslySetInnerHTML={{ __html: highlight(text) }} />
    </div>
  );
}

const ESCAPES: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;' };

function highlight(text: string): string {
  const escaped = text.replace(/[&<>]/g, (ch) => ESCAPES[ch] ?? ch);
  return escaped.replace(
    /("(?:\\.|[^"\\])*"\s*:)|("(?:\\.|[^"\\])*")|(\b-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(\btrue\b|\bfalse\b|\bnull\b)/g,
    (match, key, string, number, literal) => {
      if (key) return `<span class="j-key">${key}</span>`;
      if (string) return `<span class="j-str">${string}</span>`;
      if (number) return `<span class="j-num">${number}</span>`;
      if (literal) return `<span class="j-lit">${literal}</span>`;
      return match;
    },
  );
}
