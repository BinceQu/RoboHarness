"""Read saved Codex Responses items without reconstructing hidden reasoning.

Response items are the model-facing conversation. CommandExecution events are
separate execution evidence, not extra stdout to insert into that conversation.
Nothing in a recorded JavaScript or shell command is evaluated here.
"""
import json
import re
import shlex
from urllib.parse import unquote, urlsplit


def local_path(value):
    return unquote(urlsplit(value).path) if value.startswith('file://') else value


def command_event(event):
    """Actual shell execution evidence, including commands with computed args."""
    payload = event.get('payload', {})
    item = payload.get('item', {})
    if event.get('type') != 'event_msg' or payload.get('type') != 'item_completed':
        return None
    if item.get('type') != 'CommandExecution':
        return None
    argv = item.get('command', [])
    command = (argv[-1] if len(argv) >= 3 and argv[-2] in ('-lc', '-c')
               else shlex.join(argv)) if isinstance(argv, list) else argv
    return {'type': 'assistant', 'cwd': local_path(item.get('cwd', '')),
            'message': {'content': [{'type': 'tool_use', 'name': 'Bash',
                                    'input': {'command': command}}]}}


def literal_commands(code):
    """Display literal cmd strings; computed commands stay in the original code.

Only JSON string literals following exec_command's cmd property are decoded.
Do not execute JavaScript or infer commands from mentions of Python filenames.
"""
    found = []
    pattern = r'\btools\.exec_command\(\s*\{\s*(?:"cmd"|cmd)\s*:\s*("(?:\\.|[^"\\])*")'
    for match in re.finditer(pattern, code):
        try:
            found.append(json.loads(match[1]))
        except ValueError:
            pass
    return found


def content(value, *, output=False):
    """Preserve saved text/image order; unwrap only the code tool's transport."""
    if isinstance(value, str):
        value = [{'type': 'input_text', 'text': value}]
    if not isinstance(value, list):
        return [{'type': 'text', 'text': json.dumps(value, ensure_ascii=False)}]
    result = []
    for block in value:
        kind = block.get('type')
        if kind in ('input_text', 'output_text', 'text', 'summary_text', 'reasoning_text'):
            text = block.get('text', '')
            if output and re.fullmatch(r'Script completed\nWall time [^\n]+\nOutput:\n?', text):
                continue
            if output:
                try:
                    envelope = json.loads(text)
                except ValueError:
                    envelope = None
                if (isinstance(envelope, dict) and isinstance(envelope.get('output'), str)
                        and ('chunk_id' in envelope or 'wall_time_seconds' in envelope)):
                    result.append({'type': 'text', 'text': envelope['output'],
                                   'exit_code': envelope.get('exit_code')})
                    continue
            result.append({'type': 'text', 'text': text})
        elif kind == 'input_image':
            result.append({'type': 'image', 'source': {
                'image_url': block.get('image_url'), 'sha256': block.get('sha256'),
                'detail': block.get('detail')}})
        else:
            result.append(block)
    return result


def response_event(event, group='codex'):
    """Normalize one saved item for the existing four-column renderer."""
    if event.get('type') != 'response_item':
        return event
    payload = event.get('payload', {})
    kind = payload.get('type')
    base = {'timestamp': event.get('timestamp')}
    if kind == 'message':
        role = payload.get('role')
        if role in ('developer', 'system'):
            return {**base, 'type': 'attachment', 'attachment': {
                'type': role.title() + ' · 已保存指令', 'content': content(payload.get('content', []))}}
        return {**base, 'type': role, 'message': {
            'id': group, 'content': content(payload.get('content', []))}}
    if kind == 'reasoning':
        readable = content(payload.get('summary') or payload.get('content') or [])
        blocks = [{'type': 'thinking', 'thinking': b.get('text', ''),
                   'label': 'Reasoning summary' if payload.get('summary') else 'Saved reasoning'}
                  for b in readable if b.get('text')]
        if not blocks:
            blocks = [{'type': 'redacted_thinking', 'text':
                '原生日志只保存了加密 reasoning，没有可读的 thinking 或摘要。'
                if payload.get('encrypted_content') else '此 reasoning 项没有保存可读文本。'}]
        return {**base, 'type': 'assistant', 'message': {'id': group, 'content': blocks}}
    if kind in ('custom_tool_call', 'function_call'):
        name = payload.get('name', '')
        if kind == 'custom_tool_call':
            code = payload.get('input', '')
            args = {'code': code}
            if name == 'exec':
                args['commands'] = literal_commands(code)
        else:
            raw = payload.get('arguments', '{}')
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except ValueError:
                args = {'__unparsedToolInput': raw}
        return {**base, 'type': 'assistant', 'message': {'id': group, 'content': [
            {'type': 'tool_use', 'id': payload.get('call_id'), 'name': name, 'input': args}]}}
    if kind in ('custom_tool_call_output', 'function_call_output'):
        returned = content(payload.get('output', ''), output=True)
        return {**base, 'type': 'user', 'message': {'content': [{
            'type': 'tool_result', 'tool_use_id': payload.get('call_id'), 'content': returned,
            'is_error': any(b.get('exit_code') not in (None, 0) for b in returned)}]}}
    return event


def conversation_rows(rows):
    """A tool return ends its preceding prefix; do not reuse old commentary."""
    groups, count = {}, 0
    for row in rows:
        event, path = row['event'], row['path']
        if event.get('type') == 'response_item':
            if path not in groups:
                count += 1
                groups[path] = f'codex-reply-{count}'
            converted = response_event(event, groups[path])
            kind = event.get('payload', {}).get('type')
            if kind in ('custom_tool_call_output', 'function_call_output') or (
                    kind == 'message' and event['payload'].get('role') != 'assistant'):
                groups.pop(path, None)
            yield {**row, 'event': converted}
        elif event.get('type') == 'compacted':
            groups.pop(path, None)
            yield {**row, 'event': {'type': 'system', 'subtype': 'compact_boundary'}}
            message = event.get('payload', {}).get('message')
            if message:
                yield {**row, 'event': {'type': 'user', 'isCompactSummary': True,
                                      'message': {'content': content(message)}}}
        else:
            yield row


def capabilities(rows):
    reasoning = readable = encrypted = images = 0
    for row in rows:
        event = row['event']
        if event.get('type') != 'response_item':
            continue
        payload = event.get('payload', {})
        if payload.get('type') == 'reasoning':
            reasoning += 1
            readable += bool(payload.get('summary') or payload.get('content'))
            encrypted += bool(payload.get('encrypted_content'))
        for block in payload.get('output', []) if isinstance(payload.get('output'), list) else []:
            images += block.get('type') == 'input_image'
    return {'format': 'codex', 'reasoning_items': reasoning, 'readable_reasoning_items': readable,
            'encrypted_reasoning_items': encrypted, 'returned_image_blocks': images}
