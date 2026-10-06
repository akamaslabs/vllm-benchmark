"""Minimal OpenAI-compatible chat endpoint for a local AIPerf dry run (study 28; copied for study 30).

Streams 8 content chunks per request with 20 ms between them, answers /health and
/v1/models. Counts requests in /tmp/mock_openai.count (one line per request, with its time).
Usage: python3 mock_openai.py [port]   (default 18000)
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18000
COUNT = '/tmp/mock_openai.count'


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/health'):
            self._send(200, b'ok', 'text/plain')
        elif self.path.startswith('/v1/models'):
            self._send(200, json.dumps({'object': 'list', 'data': [{'id': 'qwen3-8b-mig', 'object': 'model'}]}).encode())
        else:
            self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        req = json.loads(self.rfile.read(n) or b'{}')
        with open(COUNT, 'a') as f:
            f.write('%.3f\n' % time.time())
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()

        def chunk(data):
            b = ('data: %s\n\n' % data).encode()
            self.wfile.write(b'%x\r\n%s\r\n' % (len(b), b))
            self.wfile.flush()

        base = {'id': 'mock', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.get('model')}
        for _ in range(8):
            chunk(json.dumps(dict(base, choices=[{'index': 0, 'delta': {'content': 'tok '}, 'finish_reason': None}])))
            time.sleep(0.02)
        chunk(json.dumps(dict(base, choices=[{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                              usage={'prompt_tokens': 10, 'completion_tokens': 8, 'total_tokens': 18})))
        chunk('[DONE]')
        self.wfile.write(b'0\r\n\r\n')
        self.wfile.flush()


ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
