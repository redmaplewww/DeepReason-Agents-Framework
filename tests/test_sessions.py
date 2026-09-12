import tempfile
import unittest
from pathlib import Path

from reasoning_agent_template.sessions import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_records_and_loads_session_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "sessions")
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            events = [{"kind": "runtime_completed"}]

            snapshot = store.record_snapshot(
                session_id="session-1",
                messages=messages,
                events=events,
                status="completed",
            )
            loaded = store.load("session-1")

        self.assertEqual(snapshot.session_id, "session-1")
        self.assertEqual(loaded.status, "completed")
        self.assertEqual(loaded.messages, messages)
        self.assertEqual(loaded.events, events)
        self.assertEqual(loaded.parent_session_id, None)

    def test_fork_copies_transcript_with_parent_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "sessions")
            store.record_snapshot(
                session_id="source",
                messages=[{"role": "user", "content": "original"}],
                events=[{"kind": "runtime_completed"}],
                status="completed",
            )

            forked = store.fork("source", "forked")

        self.assertEqual(forked.session_id, "forked")
        self.assertEqual(forked.parent_session_id, "source")
        self.assertEqual(forked.messages, [{"role": "user", "content": "original"}])
        self.assertTrue(any(event["kind"] == "session_forked" for event in forked.events))

    def test_rejects_session_ids_that_escape_store_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "sessions")

            with self.assertRaises(ValueError):
                store.record_snapshot(
                    session_id="../outside",
                    messages=[],
                    events=[],
                    status="completed",
                )

    def test_delete_removes_a_session_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "sessions")
            store.record_snapshot(session_id="delete-me", messages=[], events=[], status="completed")
            store.delete("delete-me")

            with self.assertRaises(FileNotFoundError):
                store.load("delete-me")


if __name__ == "__main__":
    unittest.main()
