"""
MOON MUSIC — core/server.py
Flask REST API
Author: Amir Mahdi Barati | github.com/Amir-Mahdi-Barati
"""
from __future__ import annotations
import os, threading, time
from flask import Flask, jsonify, request, render_template

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_app(player, gesture) -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE, "templates"),
        static_folder=os.path.join(BASE, "static"),
    )
    app.config["JSON_SORT_KEYS"] = False

    def _bg():
        while True:
            try:
                player.tick()
                t = player.current
                if t:
                    gesture.sync_player_state(t.title, t.artist, player.playing, player.volume)
            except Exception as e:
                print(f"[BG] {e}")
            time.sleep(0.4)
    threading.Thread(target=_bg, daemon=True, name="Ticker").start()

    def ok(**kw):
        return jsonify({"ok": True, **kw})

    def err(m, c=400):
        return jsonify({"ok": False, "error": m}), c

    def body():
        return request.get_json(silent=True) or {}

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/state")
    def api_state():
        return jsonify(player.state)

    @app.route("/api/library")
    def api_library():
        return jsonify(player.library_dicts)

    @app.route("/api/recommendations")
    def api_recs():
        n = min(int(request.args.get("limit", 6)), 20)
        return jsonify(player.recommendations(n))

    @app.route("/api/reload", methods=["POST"])
    def api_reload():
        n = player.reload()
        return ok(count=n, message=f"Loaded {n} tracks")

    @app.route("/api/play", methods=["POST"])
    def api_play():
        tid = body().get("track_id")
        if tid:
            if not player.play_by_id(tid):
                return err(f"Track not found: {tid}", 404)
        else:
            if not player.current:
                return err("No tracks loaded", 400)
            player.play_current()
        return jsonify(player.state)

    @app.route("/api/toggle", methods=["POST"])
    def api_toggle():
        player.toggle()
        return jsonify(player.state)

    @app.route("/api/next", methods=["POST"])
    def api_next():
        player.next()
        return jsonify(player.state)

    @app.route("/api/previous", methods=["POST"])
    def api_prev():
        player.previous()
        return jsonify(player.state)

    @app.route("/api/seek", methods=["POST"])
    def api_seek():
        try:
            s = float(body()["seconds"])
        except Exception:
            return err("'seconds' required")
        player.seek(s)
        return ok(position=round(s, 2))

    @app.route("/api/volume", methods=["POST"])
    def api_volume():
        try:
            v = float(body()["volume"])
        except Exception:
            return err("'volume' required")
        player.set_volume(v)
        return ok(volume=round(player.volume, 3))

    @app.route("/api/shuffle", methods=["POST"])
    def api_shuffle():
        player.toggle_shuffle()
        return jsonify(player.state)

    @app.route("/api/repeat", methods=["POST"])
    def api_repeat():
        player.cycle_repeat()
        return jsonify(player.state)

    @app.route("/api/like", methods=["POST"])
    def api_like():
        tid = body().get("track_id")
        if not tid:
            return err("'track_id' required")
        player.toggle_like(tid)
        return jsonify(player.state)

    @app.route("/api/gesture/status")
    def api_g_status():
        return jsonify(gesture.status)

    @app.route("/api/gesture/start", methods=["POST"])
    def api_g_start():
        if gesture.start():
            return ok(message="Gesture engine started")
        return err(gesture.error or "Could not start")

    @app.route("/api/gesture/stop", methods=["POST"])
    def api_g_stop():
        gesture.stop()
        return ok(message="Stopped")

    @app.errorhandler(404)
    def e404(_):
        return err("Not found", 404)

    @app.errorhandler(500)
    def e500(e):
        return err(f"Server error: {e}", 500)

    return app
