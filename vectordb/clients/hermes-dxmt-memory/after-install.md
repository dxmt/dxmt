# DXMT memory is installed

Three steps remain.

1. **Install the client into Hermes's interpreter** (it is shared across profiles, so once):

   ```sh
   uv pip install --python ~/.hermes/hermes-agent/venv/bin/python3 dxmt-python-sdk pyyaml
   ```

2. **Have a DXMT server running.** One binary or one container, with an embedding model
   (a local `ollama pull bge-m3` is found on its own):

   ```sh
   docker run -d -p 127.0.0.1:9999:9999 -v dxmt:/data dxmtai/dxmt
   ```

   The server does the embedding, so Hermes needs no embedding key.

3. **Activate the provider** and pick `dxmt`:

   ```sh
   hermes memory setup
   ```

Then `hermes memory status` should show `Provider: dxmt`. The server is not on
`localhost:9999`? Set `DXMT_URL`, or `plugins.dxmt.server.url` in `~/.hermes/config.yaml`.
