#!/usr/bin/env python3
"""
Standalone diagnostic: tests Telegram editMessage + MiniMax streaming.
Run from the openclaw directory: python diagnose_streaming.py
"""
import os, sys, json, time, threading
import requests

# ── Load credentials ──────────────────────────────────────────────────────────
def load_creds():
    api_key = os.getenv("MINIMAX_API_KEY")
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat  = os.getenv("TELEGRAM_CHAT_ID")

    key_path   = os.path.expanduser("~/.openclaw/credentials/keys.json")
    cfg_path   = os.path.expanduser("~/.openclaw/openclaw.json")

    if os.path.exists(key_path):
        with open(key_path) as f:
            d = json.load(f)
        api_key = api_key or d.get("providers",{}).get("minimax",{}).get("default",{}).get("api_key")

    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            c = json.load(f)
        tg_cfg   = c.get("channels",{}).get("telegram",{})
        tg_token = tg_token or tg_cfg.get("botToken")
        if not tg_chat:
            allow = tg_cfg.get("allowFrom",[])
            tg_chat = str(allow[0]) if allow else None

    return api_key, tg_token, tg_chat

api_key, tg_token, tg_chat = load_creds()

print("=== Credential check ===")
print(f"MINIMAX key : {'OK (..'+api_key[-6:]+')' if api_key else 'MISSING'}")
print(f"TG token    : {'OK (..'+tg_token[-6:]+')' if tg_token else 'MISSING'}")
print(f"TG chat_id  : {tg_chat or 'MISSING'}")

if not all([api_key, tg_token, tg_chat]):
    print("\n[FATAL] Missing credentials. Check env vars or ~/.openclaw/ config.")
    sys.exit(1)

TG_API = f"https://api.telegram.org/bot{tg_token}"

# ── Step 1: sendMessage ───────────────────────────────────────────────────────
print("\n=== Step 1: Send test message to Telegram ===")
r = requests.post(f"{TG_API}/sendMessage", json={"chat_id": tg_chat, "text": "🔧 Diagnostic running..."})
print(f"HTTP {r.status_code}: {r.text[:200]}")
if r.status_code != 200 or not r.json().get("ok"):
    print("[FAIL] Cannot send message. Check token/chat_id.")
    sys.exit(1)

msg_id = r.json()["result"]["message_id"]
print(f"[OK] Message sent, id={msg_id}")

# ── Step 2: editMessage ───────────────────────────────────────────────────────
print("\n=== Step 2: Test editMessage (needed for animation) ===")
time.sleep(0.5)
r2 = requests.post(f"{TG_API}/editMessageText", json={
    "chat_id": tg_chat, "message_id": msg_id, "text": "✏️ Edit test OK!"
})
print(f"HTTP {r2.status_code}: {r2.text[:200]}")
if r2.status_code != 200:
    print("[FAIL] editMessage failed — animation will not work!")
else:
    print("[OK] editMessage works.")

# ── Step 3: MiniMax streaming ─────────────────────────────────────────────────
print("\n=== Step 3: MiniMax streaming format test ===")
mm_url = "https://api.minimax.io/anthropic/v1/messages"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
}
payload = {
    "model": "MiniMax-M2.5-Lightning",
    "max_tokens": 80,
    "stream": True,
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Count from 1 to 5, one number per word."}],
}

print("Sending request to MiniMax...")
resp = requests.post(mm_url, headers=headers, json=payload, timeout=30, stream=True)
print(f"HTTP {resp.status_code}")

if resp.status_code != 200:
    print(f"[FAIL] MiniMax error: {resp.text[:300]}")
    sys.exit(1)

print("--- SSE lines received ---")
full_text = ""
chunks_received = 0
for raw in resp.iter_lines():
    if not raw:
        continue
    line = raw.decode() if isinstance(raw, bytes) else raw
    if not line.startswith("data: "):
        print(f"  [non-data]: {line}")
        continue
    data_str = line[6:].strip()
    if data_str == "[DONE]":
        print("  [DONE]")
        break
    try:
        ev = json.loads(data_str)
        ev_type = ev.get("type", "?")

        # Format A: Anthropic
        if ev_type == "content_block_delta":
            delta = ev.get("delta", {})
            if delta.get("type") == "text_delta":
                txt = delta.get("text", "")
                full_text += txt
                chunks_received += 1
                print(f"  [A-anthropic] chunk #{chunks_received}: {repr(txt)}")

        # Format B: OpenAI-style
        elif "choices" in ev:
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content"):
                    full_text += d["content"]
                    chunks_received += 1
                    print(f"  [B-openai] chunk #{chunks_received}: {repr(d['content'])}")

        # Other events
        else:
            print(f"  [event:{ev_type}] (no text chunk)")

    except json.JSONDecodeError as e:
        print(f"  [JSON ERR]: {e} — raw: {data_str[:100]}")

print(f"\n--- Full assembled text ({chunks_received} chunks) ---")
print(full_text)

# ── Step 4: Simulate the animation ───────────────────────────────────────────
print("\n=== Step 4: Simulate animation on Telegram ===")
r3 = requests.post(f"{TG_API}/sendMessage", json={"chat_id": tg_chat, "text": "🤔 Thinking..."})
if not r3.json().get("ok"):
    print("[SKIP] Could not send animation test message")
else:
    anim_id = r3.json()["result"]["message_id"]
    frames = ["⠋ 🟥🟨🟩⬜⬜⬜⬜ 💃 🎸","⠙ ⬜🟥🟨🟩⬜⬜⬜ 💃 🎸","⠹ ⬜⬜🟥🟨🟩⬜⬜ 💃 🎸","⠸ ⬜⬜⬜🟥🟨🟩⬜ 💃 🎸",]
    print("Cycling 4 animation frames (watch Telegram)...")
    for i, frame in enumerate(frames):
        time.sleep(1.1)
        r4 = requests.post(f"{TG_API}/editMessageText", json={
            "chat_id": tg_chat, "message_id": anim_id,
            "text": f"Thinking...\n{frame}"
        })
        print(f"  Frame {i+1}: HTTP {r4.status_code} {'[OK]' if r4.status_code==200 else '[FAIL] '+r4.text[:80]}")

    # Final result
    time.sleep(1.1)
    requests.post(f"{TG_API}/editMessageText", json={
        "chat_id": tg_chat, "message_id": anim_id,
        "text": f"✅ Diagnostic complete!\n\nMiniMax returned {chunks_received} chunks.\nFull text: {full_text[:200]}"
    })

print("\n=== Diagnostic done ===")
if chunks_received == 0:
    print("⚠️  MiniMax returned 0 text chunks — streaming format mismatch! Share the SSE lines above.")
else:
    print(f"✅ All good — {chunks_received} chunks received. Restart the bot and it should work.")
