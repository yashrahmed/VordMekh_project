import http from 'node:http';

const PORT = Number(process.env.PORT) || 3000;

const server = http.createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/hello') {
    const payload = JSON.stringify({ message: 'Hello from the OpenAI Apps SDK demo!' });
    res.writeHead(200, {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(payload),
    });
    res.end(payload);
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not Found' }));
});

server.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
