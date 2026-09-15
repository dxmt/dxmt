<!--
Everything below is prompts, not a form. Delete what does not apply.
Security fixes do not go through a public pull request — see SECURITY.md.
-->

## What this changes

<!-- What behaviour is different afterwards, and why it should be. -->

## How it was checked

<!--
Paste real output, not "should work". At minimum, from `vectordb/`:

    make verify

If the change touches the HTTP surface:

    scripts/api_tests.sh          # both storage modes

If it touches a client package, the tests for that package, run against a live server rather
than allowed to skip themselves.

If it is a performance claim, the before and the after from the same machine and the same
binary, with the one thing you changed being the only difference.
-->

## Notes for the reviewer

<!--
Anything that would be hard to see from the diff: a test that fails without the fix, an
invariant you are relying on, a path you deliberately did not change.
-->

- [ ] If this adds a route or changes a path an SDK calls, `tests/api_contract_test.rs` knows about it
- [ ] If this changes behaviour the README describes, the README changed too
- [ ] Anything under `src/` or `examples/` that this adds is reachable from a binary or a build
