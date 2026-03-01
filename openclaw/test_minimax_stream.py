#!/usr/bin/env python3
"""
Diagnostic: Dump raw SSE lines from MiniMax streaming API to see actual event format.
"""
import os, json, sys, requests

# Load API key from same place telegram.py does
api_key = os.getenv("MINIMAX_API_KEY")
if not api_key:
    key_path = os.path.expanduser("~/.openclaw/credentials/keys.json")
    if os.path.exists(key_path):
        with open(key_path) as f:
            d = json.load(f)
        api_key = d.get("providers", {}).get("minimax", {}).get("default", {}).get("api_key")

if not api_key:
    print("ERROR: No MINIMAX_API_KEY found")
    sys.exit(1)

print(f"API key found (last 8 chars): ...{api_key[-8:]}\n")

url = "https://api.minimax.io/anthropic/v1/messages"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01"
}
data = {
    "model": "MiniMax-M2.5-Lightning",
    "max_tokens": 60,
    "stream": True,
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}]
}

print("Sending request...\n")
resp = requests.post(url, headers=headers, json=data, timeout=30, stream=True)
print(f"HTTP status: {resp.status_code}\n")

if resp.status_code != 200:
    print(f"Error body: {resp.text[:500]}")
    sys.exit(1)

print("--- RAW SSE LINES ---")
full_text = ""
line_count = 0
for raw_line in resp.iter_lines():
    if not raw_line:
        continue
    line = raw_line.decode('utf-8')
    print(f"LINE {line_count:03d}: {line}")
    line_count += 1

    if line.startswith('data: '):
        data_str = line[6:]
        if data_str == '[DONE]':
            print("  -> [DONE] marker")
            break
        try:
            event = json.loads(data_str)
            print(f"  -> JSON type: {event.get('type', 'N/A')}")
            # Try Anthropic-style
            if event.get('type') == 'content_block_delta':
                delta = event.get('delta', {})
                print(f"     delta type: {delta.get('type')}")
                if delta.get('type') == 'text_delta':
                    txt = delta.get('text', '')
                    full_text += txt
                    print(f"     text: {repr(txt)}")
            # Try OpenAI-style
            elif 'choices' in event:
                choices = event.get('choices', [])
                for ch in choices:
                    delta = ch.get('delta', {})
                    if 'content' in delta:
                        full_text += delta['content']
                        print(f"     [OpenAI-style] content: {repr(delta['content'])}")
            # Print full structure if neither matched
            else:
                print(f"     FULL EVENT: {json.dumps(event, indent=2)[:400]}")
        except json.JSONDecodeError as e:
            print(f"  -> JSON decode error: {e}")

print("\n--- FULL ASSEMBLED TEXT ---")
print(full_text)
