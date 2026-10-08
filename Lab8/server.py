"""
Lab 8: Reactive ANN Server Launcher
Serves the Light-Mode interactive web simulator and REST API.
"""
import sys
from reactive_form.server import start_server

if __name__ == '__main__':
    port = 8080
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port)
