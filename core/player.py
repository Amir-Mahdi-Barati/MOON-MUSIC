"""
MOON MUSIC — core/player.py
Audio Engine + Track Model + Player Controller
Author: Amir Mahdi Barati | github.com/Amir-Mahdi-Barati
"""
from __future__ import annotations

import math
import random
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pygame

SUPPORTED = {".mp3", ".wav", ".ogg", ".flac"}
GENRES = ["Electronic", "Ambient", "Lo-Fi", "Jazz",
          "Synthwave", "Drone", "Classical", "Persian Traditional"]
SYMBOLS = ["◈", "⬡", "◉", "✦", "⊕", "◐", "⊞", "⊗", "♦", "◆", "⋈", "⊘"]


@dataclass
class Track:
    id: str
    title: str
    artist: str
    genre: str
    duration: float
    path: str
    play_count: int = 0
    liked: bool = False
    waveform: list = field(default_factory=list)

    @property
    def duration_str(self) -> str:
        m, s = divmod(int(max(0, self.duration)), 60)
        return f"{m}:{s:02d}"

    @property
    def symbol(self) -> str:
        idx = int(self.id[1:]) if self.id[1:].isdigit() else 0
        return SYMBOLS[idx % len(SYMBOLS)]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "genre": self.genre,
            "duration": round(self.duration, 2),
            "duration_str": self.duration_str,
            "path": self.path,
            "play_count": self.play_count,
            "liked": self.liked,
            "waveform": self.waveform,
            "symbol": self.symbol,
        }


class TrackRepository:
    """Scans the music folder. No demo-track generation — empty folder stays empty."""

    def __init__(self, music_dir: Path):
        self.dir = music_dir
        self.dir.mkdir(exist_ok=True)

    def load(self) -> List[Track]:
        files = sorted(
            f for f in self.dir.rglob("*")
            if f.suffix.lower() in SUPPORTED
        )
        return [self._build(f, i) for i, f in enumerate(files)]

    def _build(self, path: Path, idx: int) -> Track:
        stem = path.stem.replace("_", " ").replace("-", " ")
        parts = stem.split(" — ", 1)
        title = parts[0].strip()
        artist = parts[1].strip() if len(parts) > 1 else "Unknown Artist"
        return Track(
            id=f"t{idx:04d}",
            title=title,
            artist=artist,
            genre=GENRES[idx % len(GENRES)],
            duration=self._get_duration(path),
            path=str(path),
            waveform=self._make_waveform(64),
        )

    @staticmethod
    def _get_duration(path: Path) -> float:
        try:
            if path.suffix.lower() == ".wav":
                with wave.open(str(path)) as wf:
                    return wf.getnframes() / wf.getframerate()
        except Exception:
            pass
        return random.uniform(150, 320)

    @staticmethod
    def _make_waveform(n: int) -> list:
        return [
            round(min(1.0, max(0.05,
                .45 * abs(math.sin(i * .31))
                + .25 * abs(math.sin(i * .87 + 1.2))
                + .15 * abs(math.sin(i * 2.1))
                + random.uniform(0, .15)
            )), 3)
            for i in range(n)
        ]


class AudioEngine:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.mixer.init()
        self._lock = threading.Lock()
        self._start = 0.0
        self._offset = 0.0
        self.volume = 0.75
        self.is_playing = False
        pygame.mixer.music.set_volume(self.volume)

    def load_play(self, path: str) -> None:
        with self._lock:
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.set_volume(self.volume)
                pygame.mixer.music.play()
                self._start = time.time()
                self._offset = 0.0
                self.is_playing = True
            except Exception as e:
                print(f"[Audio] {e}")

    def pause(self) -> None:
        with self._lock:
            if self.is_playing:
                pygame.mixer.music.pause()
                self._offset += time.time() - self._start
                self.is_playing = False

    def resume(self) -> None:
        with self._lock:
            if not self.is_playing:
                pygame.mixer.music.unpause()
                self._start = time.time()
                self.is_playing = True

    def seek(self, s: float) -> None:
        with self._lock:
            try:
                pygame.mixer.music.set_pos(max(0.0, s))
                self._start = time.time()
                self._offset = s
            except Exception:
                pass

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, v))
        pygame.mixer.music.set_volume(self.volume)

    def busy(self) -> bool:
        return bool(pygame.mixer.music.get_busy())

    @property
    def position(self) -> float:
        if self.is_playing:
            return self._offset + (time.time() - self._start)
        return self._offset


class PlayerController:
    """NOTE: Does NOT autoplay on app launch — the user must press Play."""

    def __init__(self, repo: TrackRepository, audio: AudioEngine):
        self._repo = repo
        self._audio = audio
        self._lock = threading.Lock()
        self.library: List[Track] = []
        self.playlist: List[Track] = []
        self.index = 0
        self.shuffled = False
        self.repeat = "none"  # none | all | one
        self._load()

    def _load(self) -> None:
        self.library = self._repo.load()
        self.playlist = list(self.library)

    def reload(self) -> int:
        old_idx = self.index
        self._load()
        self.index = min(old_idx, max(0, len(self.playlist) - 1))
        return len(self.library)

    @property
    def current(self) -> Optional[Track]:
        if not self.playlist:
            return None
        return self.playlist[self.index % len(self.playlist)]

    @property
    def position(self) -> float:
        return self._audio.position

    @property
    def volume(self) -> float:
        return self._audio.volume

    @property
    def playing(self) -> bool:
        return self._audio.is_playing

    def play_current(self) -> None:
        t = self.current
        if not t:
            return
        self._audio.load_play(t.path)
        t.play_count += 1

    def play_by_id(self, tid: str) -> bool:
        for i, t in enumerate(self.playlist):
            if t.id == tid:
                with self._lock:
                    self.index = i
                self.play_current()
                return True
        return False

    def toggle(self) -> None:
        if (not self._audio.is_playing and self._audio.position == 0.0
                and not self._audio.busy() and self.current):
            self.play_current()
            return
        if self._audio.is_playing:
            self._audio.pause()
        else:
            self._audio.resume()

    def next(self) -> None:
        n = len(self.playlist)
        if not n:
            return
        with self._lock:
            self.index = (
                random.randint(0, n - 1) if self.shuffled
                else (self.index + 1) % n
            )
        self.play_current()

    def previous(self) -> None:
        if self._audio.position > 3.0:
            self._audio.seek(0.0)
            return
        with self._lock:
            self.index = (self.index - 1) % max(1, len(self.playlist))
        self.play_current()

    def seek(self, s: float) -> None:
        self._audio.seek(s)

    def set_volume(self, v: float) -> None:
        self._audio.set_volume(v)

    def toggle_shuffle(self) -> None:
        self.shuffled = not self.shuffled

    def cycle_repeat(self) -> None:
        self.repeat = {"none": "all", "all": "one", "one": "none"}[self.repeat]

    def toggle_like(self, tid: str) -> None:
        for t in self.library:
            if t.id == tid:
                t.liked = not t.liked
                return

    def tick(self) -> None:
        if self._audio.is_playing and not self._audio.busy():
            if self.repeat == "one":
                self.play_current()
            else:
                self.next()

    @property
    def state(self) -> dict:
        t = self.current
        pos = self.position
        dur = t.duration if t else 1.0
        return {
            "track": t.to_dict() if t else None,
            "is_playing": self.playing,
            "position": round(pos, 2),
            "progress": round(min(1.0, pos / dur), 4) if dur > 0 else 0,
            "volume": round(self.volume, 3),
            "shuffled": self.shuffled,
            "repeat": self.repeat,
            "playlist_len": len(self.playlist),
            "index": self.index,
            "has_tracks": len(self.library) > 0,
        }

    @property
    def library_dicts(self) -> list:
        return [t.to_dict() for t in self.library]

    def recommendations(self, limit: int = 6) -> list:
        if not self.library:
            return []
        reasons = [
            "Based on your history", "Trending in genre",
            "Because you liked it", "Curated for you",
            "Hidden gem", "Fan favourite",
        ]
        scored = sorted(
            self.library,
            key=lambda t: (
                t.liked * 60
                + math.log1p(t.play_count) * 12
                + random.uniform(0, 18)
            ),
            reverse=True,
        )
        out = []
        for t in scored[:limit]:
            d = t.to_dict()
            d["reason"] = reasons[hash(t.id) % len(reasons)]
            out.append(d)
        return out
