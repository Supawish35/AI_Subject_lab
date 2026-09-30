import os
import sys
import webbrowser
import http.server
import socketserver

def start_server(port=8080, html_file="index.html"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, html_file)
    if not os.path.exists(file_path):
        from web_builder import build_reactive_website
        dataset_path = os.path.abspath(os.path.join(base_dir, '..', 'DataSet', 'student_dataset_full_normalized.csv'))
        build_reactive_website(dataset_path, file_path)

    os.chdir(base_dir)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            super().end_headers()

    while True:
        try:
            with socketserver.TCPServer(("", port), CustomHandler) as httpd:
                url = f"http://localhost:{port}/{html_file}"
                print("=" * 68)
                print(f" REACTIVE K-NN MODEL WEB SERVER")
                print("=" * 68)
                print(f"[*] Serving Reactive K-NN Simulator at: {url}")
                print("    - 10,000 Live Training Samples embedded")
                print("    - Sub-millisecond Client-Side Euclidean Prediction")
                print("    - Real-Time Feature Sliders & Scatter Projection")
                print(f"[*] Press Ctrl+C in terminal to stop server.")
                print("=" * 68 + "\n")
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
                httpd.serve_forever()
        except OSError as e:
            if "Address already in use" in str(e):
                port += 1
            else:
                raise e

if __name__ == '__main__':
    port = 8080
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port)
