"""Exercise real ASGI routing with synthetic data; never load recognition models."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import server


PUBLIC = {
    "/": "index.html", "/index.html": "index.html", "/app.js": "app.js",
    "/styles.css": "styles.css", "/app-icon.svg": "app-icon.svg",
    "/manifest.webmanifest": "manifest.webmanifest", "/sw.js": "sw.js",
}
PRIVATE = (
    "jobs/sample/input/recording.wav", "jobs/sample/result.json",
    "voiceprints/profiles.json", "logs/server.log", "server.py", "main_worker.py",
    "config.json", ".env", "__pycache__/server.pyc", "validation/evidence.json",
    "future.js", "jobs/public.html", "logs/public.css", "voiceprints/icon.svg",
)


class StaticRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.public_dir = self.root / "app"
        self.public_dir.mkdir()
        for name in set(PUBLIC.values()):
            (self.public_dir / name).write_text("public: " + name, encoding="utf-8")
        for name in PRIVATE:
            path = self.public_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("PRIVATE-SENTINEL", encoding="utf-8")
        sibling = self.root / "app-private"
        sibling.mkdir()
        (sibling / "secret.json").write_text("PRIVATE-SENTINEL", encoding="utf-8")
        self.patcher = patch.object(server, "APP_DIR", self.public_dir)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        # ASGITransport does not run startup handlers or start a listening port.
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                       base_url="http://127.0.0.1:8767")
        self.addAsyncCleanup(self.client.aclose)

    async def test_exact_public_files_support_get_and_head(self):
        for url, filename in PUBLIC.items():
            with self.subTest(url=url):
                response = await self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.text, "public: " + filename)
                head = await self.client.head(url)
                self.assertEqual(head.status_code, 200)
                self.assertEqual(head.content, b"")
                self.assertEqual(head.headers["content-length"], response.headers["content-length"])
        response = await self.client.get("/sw.js")
        self.assertEqual(response.headers["cache-control"], "no-cache")

    async def test_private_files_and_public_extensions_in_private_directories_are_denied(self):
        for name in PRIVATE:
            for method in ("GET", "HEAD"):
                with self.subTest(name=name, method=method):
                    response = await self.client.request(method, "/" + name)
                    self.assertEqual(response.status_code, 404)
                    self.assertNotIn(b"PRIVATE-SENTINEL", response.content)

    async def test_encoded_paths_and_directory_boundaries_are_denied(self):
        paths = (
            "/jobs", "/jobs/", "/voiceprints/", "/logs/", "/app-private/secret.json",
            "/../app-private/secret.json", "/%2e%2e/app-private/secret.json",
            "/%2e%2e%2fapp-private%2fsecret.json", "/%252e%252e/app-private/secret.json",
            "/app.js/%2e%2e/server.py", "/app.js/../server.py",
            "/voiceprints%2fprofiles.json", "/voiceprints%252fprofiles.json",
            "/%76oiceprints/profiles.json", "/voiceprints/%70rofiles.json",
            "/%2fvoiceprints/profiles.json", "/voiceprints//profiles.json",
            "/./server.py", "/%2e/server.py",
            "/voiceprints/profiles.json%3fapp.js", "/voiceprints/profiles.json%23app.js",
            "/voiceprints%5cprofiles.json", "/app.js%5c..%5cserver.py",
            "/SERVER.PY", "/server.py/", "/server.py.", "/server.py%20",
            "/server.py::$DATA", "/server.py%00.js", "/.git/config",
            "/app.js/extra", "/index.html.bak", "/styles.css/../logs/server.log",
        )
        for path in paths:
            with self.subTest(path=path):
                response = await self.client.get(path)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(b"PRIVATE-SENTINEL", response.content)

    async def test_query_string_cannot_select_a_file(self):
        response = await self.client.get("/app.js?filename=server.py&path=../voiceprints/profiles.json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "public: app.js")
        self.assertEqual((await self.client.get("/server.py?file=app.js")).status_code, 404)

    async def test_new_unlisted_file_does_not_become_public(self):
        (self.public_dir / "new.html").write_text("PRIVATE-SENTINEL", encoding="utf-8")
        self.assertEqual((await self.client.get("/new.html")).status_code, 404)

    async def test_missing_allowed_file_returns_404(self):
        (self.public_dir / "app.js").unlink()
        self.assertEqual((await self.client.get("/app.js")).status_code, 404)

    async def test_asset_symlink_cannot_expose_another_file(self):
        asset = self.public_dir / "app.js"
        asset.unlink()
        try:
            asset.symlink_to(self.public_dir / "voiceprints/profiles.json")
        except OSError as exc:
            self.skipTest("Creating a symlink is not permitted on this host: " + str(exc))
        self.assertEqual((await self.client.get("/app.js")).status_code, 404)

    async def test_static_routes_do_not_accept_writes(self):
        for method in ("POST", "PUT", "DELETE"):
            with self.subTest(method=method):
                self.assertEqual((await self.client.request(method, "/app.js")).status_code, 405)

    async def test_same_origin_api_and_text_on_demand_warmup_still_work(self):
        response = await self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        with patch.object(server._main_engine, "warmup") as preload:
            response = await self.client.post("/api/warmup?mode=text")
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["mode"], "text")
            self.assertEqual(response.json()["status"], "on_demand")
            preload.assert_not_called()

    async def test_foreign_origin_gets_no_cors_permission(self):
        for origin in ("https://example.invalid", "null", "http://localhost:9000"):
            for url in ("/app.js", "/api/health", "/voiceprints/profiles.json"):
                with self.subTest(origin=origin, url=url):
                    response = await self.client.get(url, headers={"Origin": origin})
                    self.assertNotIn("access-control-allow-origin", response.headers)
                    self.assertNotIn("access-control-allow-credentials", response.headers)
        response = await self.client.options("/api/jobs", headers={
            "Origin": "https://example.invalid", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"})
        self.assertEqual(response.status_code, 405)
        self.assertNotIn("access-control-allow-origin", response.headers)

    async def test_api_routes_remain_in_openapi_without_static_routes(self):
        response = await self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]
        for path in ("/api/jobs", "/api/jobs/{job_id}", "/api/jobs/{job_id}/result",
                     "/api/jobs/{job_id}/client-timing", "/api/voiceprints", "/api/transcribe"):
            self.assertIn(path, paths)
        self.assertTrue(set(PUBLIC).isdisjoint(paths))

    async def test_actual_page_manifest_and_worker_assets_remain_accessible(self):
        with patch.object(server, "APP_DIR", Path(__file__).resolve().parent):
            for url, filename in PUBLIC.items():
                with self.subTest(url=url):
                    response = await self.client.get(url)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.content, (server.APP_DIR / filename).read_bytes())
            manifest = (await self.client.get("/manifest.webmanifest")).json()
            self.assertIn(manifest["start_url"], PUBLIC)
            self.assertTrue(all(icon["src"] in PUBLIC for icon in manifest["icons"]))


class StaticWorkerTests(unittest.TestCase):
    def test_actual_service_worker_cache_policy(self):
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node.js is required for service worker regression checks")
        result = subprocess.run([node, str(Path(__file__).with_name("test_static_worker.cjs"))],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
