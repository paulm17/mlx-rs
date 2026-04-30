# Stopping In-Flight LLM Generation

The mlx-server does not expose a dedicated `/llm/stop` endpoint. To abort
generation, close the SSE connection. The server detects the disconnect,
sets the internal stop signal, and the generation loop exits at the next
iteration, returning whatever was generated so far.

## JavaScript (SSE via `fetch`)

```js
let controller = null;

async function sendChat(messages) {
  controller = new AbortController();

  const response = await fetch("http://127.0.0.1:3000/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      stream: true,
      max_tokens: 2048,
    }),
    signal: controller.signal,
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const payload = line.slice(6);
        if (payload === "[DONE]") return;
        const chunk = JSON.parse(payload);
        // Render chunk.choices[0].delta.content
      }
    }
  }
}

function stopGeneration() {
  if (controller) {
    controller.abort();
    controller = null;
  }
}
```

## JavaScript (EventSource — not recommended)

`EventSource` does not support `POST` or custom headers, so it cannot send
the request body or `Content-Type`. Always use `fetch` with a
`ReadableStream` reader as shown above.

## How it works server-side

1. Client calls `AbortController.abort()`, which tears down the TCP connection.
2. The server's next `write_sse_event()` call fails with a broken pipe.
3. The error handler sets the shared `AtomicBool` stop flag to `true`.
4. The generation loop checks the flag at the top of each iteration, breaks
   with `stop_reason = "cancelled"`, and returns the partial output.
