"""Actual Claude CLI, deterministic local Messages API; no bridge or robot."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import shutil
import subprocess
import threading

import pytest


@pytest.mark.skipif(os.getenv('ROBODOJO_NATIVE_CLAUDE_TESTS') != '1',
                    reason='opt-in native CLI regression')
def test_native_compaction_and_tool_continuation(tmp_path):
    note = tmp_path/'note.txt'
    note.write_text('NATIVE_COMPACTION_MARKER\n')
    requests, summaries = [], []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if 'count_tokens' in self.path:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"input_tokens":1000}')
                return
            last = next((m for m in reversed(body['messages']) if m['role']=='user'), {})
            summary = 'Your task is to create a detailed summary' in json.dumps(last)
            if summary:
                summaries.append(body)
                content = {'type': 'text', 'text': '<summary>Two note reads completed. '
                    f'Read {note} once more, then say NATIVE_COMPACTION_OK. '
                    'The marker is NATIVE_COMPACTION_MARKER.</summary>'}
            else:
                requests.append(body)
                content = ({'type':'tool_use', 'id':f'tool_read_{len(requests)}',
                    'name':'Read', 'input':{'file_path':str(note)}} if len(requests)<=3
                    else {'type':'text', 'text':'NATIVE_COMPACTION_OK'})
            tool = content['type']=='tool_use'
            tokens = 190000 if not summary and len(requests)==2 else 1000
            message = {'id':f'msg_{len(requests)}_{len(summaries)}', 'type':'message',
                'role':'assistant','model':'Qwen3.8-Flash-Next-NVFP4', 'content':[],
                'stop_reason':None, 'stop_sequence':None,
                'usage':{'input_tokens':tokens,'output_tokens':0}}
            stop = 'tool_use' if tool else 'end_turn'
            if not body.get('stream'):
                message.update(content=[content], stop_reason=stop)
                message['usage']['output_tokens']=100
                encoded=json.dumps(message).encode()
                media='application/json'
            else:
                start={**content, 'input':{}} if tool else {'type':'text','text':''}
                delta=({'type':'input_json_delta','partial_json':json.dumps(content['input'])}
                       if tool else {'type':'text_delta','text':content['text']})
                events=[('message_start',{'message':message}),
                    ('content_block_start',{'index':0,'content_block':start}),
                    ('content_block_delta',{'index':0,'delta':delta}),
                    ('content_block_stop',{'index':0}),
                    ('message_delta',{'delta':{'stop_reason':stop,'stop_sequence':None},
                                      'usage':{'output_tokens':100}}),
                    ('message_stop',{})]
                encoded=''.join('event: '+kind+'\ndata: '+json.dumps({'type':kind,**value})+'\n\n'
                                for kind,value in events).encode()
                media='text/event-stream'
            self.send_response(200)
            self.send_header('Content-Type',media)
            self.send_header('Content-Length',str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server=ThreadingHTTPServer(('127.0.0.1',0),Endpoint)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    environment={k:v for k,v in os.environ.items() if not k.startswith(('ANTHROPIC_', 'CLAUDE_', 'EMBODIED_', 'QWEN_'))
                 and k not in {'DISABLE_COMPACT','DISABLE_AUTO_COMPACT','CLAUDECODE',
                              'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'}}
    environment.update(ANTHROPIC_BASE_URL=f'http://127.0.0.1:{server.server_port}',
        ANTHROPIC_AUTH_TOKEN='local-test', CLAUDE_CONFIG_DIR=str(tmp_path/'claude'),
        CLAUDE_CODE_AUTO_COMPACT_WINDOW='180000', CLAUDE_CODE_MAX_CONTEXT_TOKENS='262144',
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', DISABLE_TELEMETRY='1', DISABLE_ERROR_REPORTING='1')
    try:
        result=subprocess.run([shutil.which('claude'), '-p','--verbose','--output-format','stream-json',
            '--model','Qwen3.8-Flash-Next-NVFP4','--permission-mode','dontAsk',
            '--tools','Read','--allowedTools','Read','--strict-mcp-config','--mcp-config','{"mcpServers":{}}',
            '--setting-sources','project','--max-turns','8',
            f'Read {note} three times. Then say NATIVE_COMPACTION_OK.'],
            cwd=tmp_path,env=environment,text=True,capture_output=True,timeout=90)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    events=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert result.returncode==0, result.stderr+result.stdout[-4000:]
    assert len(summaries)==1, result.stdout[-5000:]
    assert any(e.get('subtype')=='compact_boundary' for e in events), result.stdout[-5000:]
    assert len(requests)==4
    assert len(requests[2]['messages']) < len(requests[1]['messages'])
    assert 'NATIVE_COMPACTION_MARKER' in json.dumps(requests[2])
    assert 'NATIVE_COMPACTION_OK' in result.stdout
