#!/usr/bin/env python3
"""
Local block editor for the blog.

    python3 tools/edit.py          then open http://127.0.0.1:4000/edit

Binds to the loopback interface only, so nothing outside this machine can
reach it. That is the whole security model: there is no password because
there is no remote access. Posts are written straight into blog/ as JSON;
publishing is `git push`, and the live site ships no editor code at all.
"""

import base64
import errno
import json
import re
import subprocess
from datetime import date
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
POSTS = CONTENT / "posts"
IMAGES = CONTENT / "images"
INDEX = CONTENT / "index.json"

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
ALLOWED_IMAGE = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".avif"}
MAX_UPLOAD = 12 * 1024 * 1024

# Only these paths are ever staged. Keeps a stray .env or key out of a
# commit made by one click, and keeps the diff reviewable.
PUBLISH_PATHS = ["content", "index.html", "style.css", "render.js", "birds.js", "img", "tools"]
MAIN_BRANCH = "main"

# The sections of the blog, drawn as birds on a branch. index.json holds
# the live list; this is only the starting set for a fresh index.
DEFAULT_BIRDS = [
    {"id": "birds", "name": "Birds", "color": "#e0513a", "blurb": ""},
    {"id": "projects", "name": "Projects", "color": "#ee9433", "blurb": ""},
    {"id": "musings", "name": "Musings", "color": "#3f7fc4", "blurb": ""},
    {"id": "coursework", "name": "Coursework", "color": "#4f9a4c", "blurb": ""},
]


def git(*args, check=True, raw=False):
    """Run one git command. List args only — never a shell string.

    raw=True keeps stdout byte-for-byte. Porcelain output needs it: each
    line starts with a two-column status field that may begin with a
    space, and stripping would silently eat the first line's leading
    space and therefore the first character of its path.
    """
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or f"git {' '.join(args)} failed")
    return r.stdout if raw else r.stdout.strip()


def git_status():
    # Before `git init` this is a perfectly normal state, not an error.
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                            cwd=ROOT, capture_output=True, text=True)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {"repo": False}

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    porcelain = git("status", "--porcelain", raw=True)

    changed, other = [], []
    for line in porcelain.splitlines():
        state, path = line[:2].strip(), line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ")[-1]
        entry = {"state": state, "path": path}
        root = path.split("/")[0]
        (changed if root in PUBLISH_PATHS else other).append(entry)

    ahead = 0
    try:
        ahead = int(git("rev-list", "--count", f"origin/{branch}..{branch}"))
    except RuntimeError:
        ahead = -1  # no upstream yet

    return {
        "repo": True,
        "branch": branch,
        "main": MAIN_BRANCH,
        "onMain": branch == MAIN_BRANCH,
        "changed": changed,
        "ignored": other,
        "ahead": ahead,
        "hasRemote": bool(git("remote", check=False)),
    }


def git_publish(message: str, merge: bool, push: bool):
    """Commit the whitelisted paths, optionally merge to main and push.

    Any failure mid-merge aborts and returns to the starting branch, so a
    conflict never strands the working tree on main.
    """
    log, start = [], git("rev-parse", "--abbrev-ref", "HEAD")
    switched = False

    try:
        existing = [p for p in PUBLISH_PATHS if (ROOT / p).exists()]
        git("add", "--", *existing)
        log.append(f"staged {', '.join(existing)}")

        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
            log.append("nothing new to commit")
        else:
            git("commit", "-m", message)
            log.append(f"committed on {start}: {message}")

        target = start
        if merge and start != MAIN_BRANCH:
            git("checkout", MAIN_BRANCH)
            switched = True
            log.append(f"checked out {MAIN_BRANCH}")
            try:
                git("merge", "--no-ff", start, "-m", f"Merge {start}: {message}")
            except RuntimeError as e:
                git("merge", "--abort", check=False)
                raise RuntimeError(f"merge conflict, aborted and left {MAIN_BRANCH} untouched — {e}")
            log.append(f"merged {start} into {MAIN_BRANCH}")
            target = MAIN_BRANCH

        if push:
            git("push", "origin", target)
            log.append(f"pushed {target} to origin")

        return {"ok": True, "log": log}

    finally:
        if switched:
            git("checkout", start, check=False)
            log.append(f"returned to {start}")


def read_index() -> dict:
    idx = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    return {
        "birds": idx.get("birds") or [dict(b) for b in DEFAULT_BIRDS],
        "posts": idx.get("posts") or [],
    }


def write_index(idx: dict) -> None:
    write_json(INDEX, {"birds": idx["birds"], "posts": idx["posts"]})


def clean_birds(raw) -> list:
    """Validate the bird list sent by the editor. Raises ValueError."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("need at least one bird")
    birds, seen = [], set()
    for b in raw:
        bid = str(b.get("id") or "").strip()
        name = str(b.get("name") or "").strip()
        color = str(b.get("color") or "").strip()
        if not SLUG_RE.match(bid):
            raise ValueError(f"bad bird id: {bid!r}")
        if bid in seen:
            raise ValueError(f"two birds called {bid!r}")
        if not name:
            raise ValueError(f"bird {bid!r} needs a name")
        if not COLOR_RE.match(color):
            raise ValueError(f"bird {bid!r} has a bad colour: {color!r}")
        seen.add(bid)
        birds.append({"id": bid, "name": name, "color": color.lower(),
                      "blurb": str(b.get("blurb") or "").strip()})
    return birds


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def upsert_index(meta: dict) -> None:
    """Keep index.json in sync with a saved post, newest first."""
    idx = read_index()
    posts = [p for p in idx["posts"] if p["slug"] != meta["slug"]]
    posts.append(meta)
    posts.sort(key=lambda p: (p.get("date") or "", p["slug"]), reverse=True)
    write_index({**idx, "posts": posts})


class Handler(SimpleHTTPRequestHandler):
    # ---------- plumbing ----------

    def log_message(self, fmt, *args):
        if "/api/" in str(args[0]):
            print(f"  {args[0]}")

    def end_headers(self):
        # Authoring tool: a stale render.js or stylesheet silently shows old
        # behaviour and looks like a code bug, so never let anything cache.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def send_head(self):
        # SimpleHTTPRequestHandler answers If-Modified-Since with a 304, which
        # would defeat the header above for a file the browser already holds.
        if "If-Modified-Since" in self.headers:
            del self.headers["If-Modified-Since"]
        return super().send_head()

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def fail(self, status, message):
        self.send_json({"error": message}, status)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("payload too large")
        return json.loads(self.rfile.read(n) or b"{}")

    @staticmethod
    def slug_from(path: str) -> str:
        slug = unquote(path.rsplit("/", 1)[-1])
        if not SLUG_RE.match(slug):
            raise ValueError(f"bad slug: {slug!r}")
        return slug

    # ---------- routes ----------

    def do_GET(self):
        route = urlparse(self.path).path

        if route == "/edit":
            self.path = "/tools/editor.html"
            return SimpleHTTPRequestHandler.do_GET(self)

        if route == "/api/index":
            return self.send_json(read_index())

        if route == "/api/git/status":
            try:
                return self.send_json(git_status())
            except RuntimeError as e:
                return self.fail(500, str(e))

        if route.startswith("/api/post/"):
            try:
                slug = self.slug_from(route)
            except ValueError as e:
                return self.fail(400, str(e))
            f = POSTS / f"{slug}.json"
            if not f.exists():
                return self.fail(404, "no such post")
            return self.send_json(json.loads(f.read_text()))

        if route.startswith("/api/"):
            return self.fail(404, "no such endpoint")

        return SimpleHTTPRequestHandler.do_GET(self)

    def do_PUT(self):
        route = urlparse(self.path).path

        if route == "/api/birds":
            try:
                birds = clean_birds(self.body().get("birds"))
            except (ValueError, AttributeError) as e:
                return self.fail(400, str(e))
            idx = read_index()
            ids = {b["id"] for b in birds}
            gone = {b["id"] for b in idx["birds"]} - ids
            orphans = [p["title"] for p in idx["posts"] if p.get("bird") in gone]
            if orphans:
                return self.fail(409, f"move these posts off a removed bird first: {', '.join(orphans)}")
            write_index({**idx, "birds": birds})
            print(f"  saved {len(birds)} birds")
            return self.send_json({"ok": True, "birds": birds})

        if not route.startswith("/api/post/"):
            return self.fail(404, "no such endpoint")
        try:
            slug = self.slug_from(route)
            data = self.body()
        except ValueError as e:
            return self.fail(400, str(e))

        bird = data.get("bird")
        if bird not in {b["id"] for b in read_index()["birds"]}:
            return self.fail(400, f"no such bird: {bird!r}")

        post = {
            "title": (data.get("title") or "Untitled").strip(),
            "date": data.get("date") or date.today().isoformat(),
            "bird": bird,
            "blocks": data.get("blocks") or [],
        }
        write_json(POSTS / f"{slug}.json", post)
        upsert_index({
            "slug": slug,
            "title": post["title"],
            "date": post["date"],
            "bird": bird,
            "summary": (data.get("summary") or "").strip() or None,
            "cover": data.get("cover") or None,
        })
        print(f"  saved {slug} ({len(post['blocks'])} blocks)")
        return self.send_json({"ok": True, "slug": slug})

    def do_DELETE(self):
        route = urlparse(self.path).path
        if not route.startswith("/api/post/"):
            return self.fail(404, "no such endpoint")
        try:
            slug = self.slug_from(route)
        except ValueError as e:
            return self.fail(400, str(e))

        (POSTS / f"{slug}.json").unlink(missing_ok=True)
        idx = read_index()
        write_index({**idx, "posts": [p for p in idx["posts"] if p["slug"] != slug]})
        print(f"  deleted {slug}")
        return self.send_json({"ok": True})

    def do_POST(self):
        route = urlparse(self.path).path

        if route == "/api/git/publish":
            try:
                data = self.body()
                message = (data.get("message") or "").strip()
                if not message:
                    return self.fail(400, "commit message required")
                result = git_publish(message, bool(data.get("merge")), bool(data.get("push")))
            except RuntimeError as e:
                return self.fail(409, str(e))
            except Exception as e:
                return self.fail(400, str(e))
            for line in result["log"]:
                print(f"  git: {line}")
            return self.send_json(result)

        if route != "/api/upload":
            return self.fail(404, "no such endpoint")
        try:
            data = self.body()
        except ValueError as e:
            return self.fail(413, str(e))

        name = SAFE_NAME_RE.sub("-", data.get("name") or "image").lstrip(".-")
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_IMAGE:
            return self.fail(400, f"unsupported file type: {suffix or 'none'}")
        if not Path(name).stem:
            name = f"image{suffix}"

        try:
            blob = base64.b64decode((data.get("data") or "").split(",")[-1], validate=True)
        except Exception:
            return self.fail(400, "could not decode file")
        if not blob:
            return self.fail(400, "file is empty")
        if len(blob) > MAX_UPLOAD:
            return self.fail(413, "file too large")

        IMAGES.mkdir(parents=True, exist_ok=True)
        target = IMAGES / name
        stem, n = target.stem, 1
        while target.exists():
            target = IMAGES / f"{stem}-{n}{suffix}"
            n += 1
        target.write_bytes(blob)

        rel = target.relative_to(ROOT).as_posix()
        print(f"  uploaded {rel} ({len(blob) // 1024} KB)")
        return self.send_json({"path": rel})


def main(port: int = 4000):
    POSTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    if not INDEX.exists() or "birds" not in json.loads(INDEX.read_text()):
        write_index(read_index())

    handler = partial(Handler, directory=str(ROOT))

    # An editor left running in another terminal is the usual cause, so say
    # so and step to the next free port rather than dumping a traceback.
    httpd, chosen = None, port
    for candidate in range(port, port + 6):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), handler)
            chosen = candidate
            break
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            print(f"  port {candidate} is busy"
                  + ("  (already running? `lsof -nP -iTCP:%d -sTCP:LISTEN`)" % candidate
                     if candidate == port else ""))

    if httpd is None:
        print(f"\n  no free port in {port}-{port + 5}. Stop the old editor, or pass "
              f"a port: python3 tools/edit.py 4100\n")
        return

    with httpd:
        if chosen != port:
            print(f"  using {chosen} instead")
        print(f"\n  editor   http://127.0.0.1:{chosen}/edit")
        print(f"  preview  http://127.0.0.1:{chosen}/")
        print("\n  loopback only — ctrl-c to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopped\n")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
