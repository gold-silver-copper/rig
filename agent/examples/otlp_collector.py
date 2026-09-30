#!/usr/bin/env python3
"""A minimal OTLP/HTTP (JSON) trace collector: prints each span it receives.

    python3 examples/otlp_collector.py PORT
    rig-pi --otlp http://127.0.0.1:PORT
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


def value(attribute):
    inner = attribute.get("value", {})
    return next(iter(inner.values()), None) if inner else None


class Collector(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")
        if self.path != "/v1/traces":
            return
        for resource in json.loads(body).get("resourceSpans", []):
            service = next(
                (value(a) for a in resource.get("resource", {}).get("attributes", []) if a["key"] == "service.name"),
                "?",
            )
            for scope in resource.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    attributes = {a["key"]: value(a) for a in span.get("attributes", [])}
                    keep = {k: v for k, v in attributes.items() if k.startswith("gen_ai.") and not k.endswith("messages")}
                    print(
                        f"{service} trace={span['traceId'][:8]} span={span['spanId'][:8]} "
                        f"parent={(span.get('parentSpanId') or '-')[:8]} {span['name']} {json.dumps(keep, sort_keys=True)}",
                        flush=True,
                    )

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), Collector).serve_forever()
