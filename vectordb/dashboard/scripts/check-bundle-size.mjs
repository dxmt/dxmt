import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

// The bundle is compiled into the server binary, so its size is the binary's size. A budget
// here is what stops "one more dependency" from being free.
const LIMIT = 4 * 1024 * 1024;

function total(dir) {
  return readdirSync(dir, { withFileTypes: true }).reduce((sum, entry) => {
    const path = join(dir, entry.name);
    return sum + (entry.isDirectory() ? total(path) : statSync(path).size);
  }, 0);
}

const size = total('dist');
const mb = (size / 1024 / 1024).toFixed(2);
if (size > LIMIT) {
  console.error(`dist/ is ${mb} MB, over the ${LIMIT / 1024 / 1024} MB budget.`);
  console.error('It is embedded in the server binary, so this is the binary growing.');
  process.exit(1);
}
console.log(`dist/ is ${mb} MB, within budget.`);
