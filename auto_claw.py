import http.server
import socketserver
import os
import time

PORT = 8765
last_trigger = 0

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global last_trigger

        # Send a successful response back to Tampermonkey
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        current_time = time.time()
        # Only press Alt+O if it hasn't been pressed in the last 3 seconds
        if current_time - last_trigger > 3:
            print(">>> TRUE PAGE LOAD DETECTED! Pressing Alt+O...")
            time.sleep(1.5) # Wait for page to finish rendering
            os.system("xdotool key --clearmodifiers alt+o")
            last_trigger = time.time()
        else:
            print("Ignoring duplicate load...")

# Start the server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Listening for Tampermonkey on port {PORT}...")
    httpd.serve_forever()

