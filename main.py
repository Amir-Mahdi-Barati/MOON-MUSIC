"""
MOON MUSIC 2026 — by Amir Mahdi Barati
github.com/Amir-Mahdi-Barati

Install:
  pip install flask pygame mediapipe==0.10.21 opencv-python numpy

Run:
  python main.py  ->  http://127.0.0.1:5050
"""
from __future__ import annotations
import sys, os, threading, time, webbrowser
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.gesture import GestureEngine
from core.player import AudioEngine, PlayerController, TrackRepository
from core.server import create_app

HOST = "127.0.0.1"
PORT = 5050
MUSIC_DIR = Path(__file__).parent / "music"


def wire(gesture: GestureEngine, player: PlayerController) -> None:
    _prev_vol = [0.75]

    def on_mute():
        if player.volume > 0:
            _prev_vol[0] = player.volume
            player.set_volume(0.0)
        else:
            player.set_volume(_prev_vol[0])

    def on_like():
        if player.current:
            player.toggle_like(player.current.id)

    def on_pinch(pinch: float = 0.1):
        vol = max(0.0, min(1.0, 1.0 - pinch * 4.0))
        player.set_volume(vol)

    def on_rock():
        player.cycle_repeat()
        print("[Rock] Rock mode!")

    (gesture
     .on("palm", player.toggle)
     .on("swipe_right", player.next)
     .on("swipe_left", player.previous)
     .on("fist", on_mute)
     .on("thumb_up", on_like)
     .on("thumb_down", player.toggle_shuffle)
     .on("peace", player.toggle_shuffle)
     .on("point_up", player.cycle_repeat)
     .on("rock", on_rock)
     .on("call_me", player.cycle_repeat)
     .on("pinch", on_pinch)
     )


def main():
    print(__doc__)
    MUSIC_DIR.mkdir(exist_ok=True)

    repo = TrackRepository(MUSIC_DIR)
    audio = AudioEngine()
    player = PlayerController(repo, audio)
    gesture = GestureEngine()
    wire(gesture, player)
    app = create_app(player, gesture)

    threading.Thread(
        target=lambda: [time.sleep(1.2), webbrowser.open(f"http://{HOST}:{PORT}")],
        daemon=True).start()

    print(f"  MOON MUSIC  ->  http://{HOST}:{PORT}\n")
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
