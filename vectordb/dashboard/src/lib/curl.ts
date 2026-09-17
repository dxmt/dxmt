/**
 * The request this page just made, as something you can paste into a terminal.
 *
 * A console is where someone notices a result they did not expect; the next thing they want is
 * to reproduce it outside the console, in a bug report or a script. Rebuilding the request by
 * hand from what the screen shows is exactly where a detail gets lost.
 */
export function asCurl(method: string, path: string, body?: unknown): string {
  const url = `${window.location.origin}${path}`;
  if (body === undefined) {
    return method === 'GET' ? `curl -s '${url}'` : `curl -s -X ${method} '${url}'`;
  }
  // Single-quoted for the shell, with the one escape that needs: a literal quote inside JSON.
  const json = JSON.stringify(body).replace(/'/g, `'\\''`);
  return `curl -s -X ${method} '${url}' \\\n  -H 'Content-Type: application/json' \\\n  -d '${json}'`;
}
