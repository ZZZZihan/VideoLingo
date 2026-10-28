"""Deterministic checks for process-group cancellation and terminal persistence.

Only operating-system boundaries are simulated. Supervisor checks retain real
job files and locks so a cleanup error cannot masquerade as a durable result.
"""

from __future__ import annotations

import errno
from itertools import count
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, call, patch
from uuid import uuid4

from videolingo_mcp import worker
from videolingo_mcp.jobs import Service, read_json, write_json


class ProcessGroupSignalTests(unittest.TestCase):
    def child(self, *, exited: bool = True) -> Mock:
        child = Mock(pid=4242)
        child.poll.return_value = 0 if exited else None
        child.wait.return_value = 0
        return child

    @staticmethod
    def denied() -> PermissionError:
        return PermissionError(errno.EPERM, "process-group signal denied")

    def test_successful_signal_and_absent_group_keep_normal_os_semantics(self) -> None:
        child = self.child()
        with patch.object(worker.os, "killpg") as killpg, patch.object(worker.subprocess, "run") as ps:
            self.assertTrue(worker._signal_group(child, signal.SIGTERM))
            killpg.assert_called_once_with(child.pid, signal.SIGTERM)
            ps.assert_not_called()
        with patch.object(worker.os, "killpg", side_effect=ProcessLookupError(errno.ESRCH, "gone")), \
                patch.object(worker.subprocess, "run") as ps:
            self.assertFalse(worker._signal_group(child, 0))
            ps.assert_not_called()

    def test_darwin_exited_leader_and_confirmed_empty_group_are_not_a_live_process(self) -> None:
        for signum in [signal.SIGTERM, 0, signal.SIGKILL]:
            for listing in ["101 S\n202 Z\n4243 S+\n", "4242 Z\n4242 Zs\n99 Ss\n"]:
                with self.subTest(signal=signum, listing=listing):
                    child = self.child()
                    result = subprocess.CompletedProcess([], 0, stdout=listing, stderr="")
                    with patch.object(worker.sys, "platform", "darwin"), \
                            patch.object(worker.os, "killpg", side_effect=self.denied()), \
                            patch.object(worker.subprocess, "run", return_value=result) as ps:
                        self.assertFalse(worker._signal_group(child, signum))
                        ps.assert_called_once_with(
                            ["/bin/ps", "-axo", "pgid=,stat="], check=True,
                            capture_output=True, text=True, timeout=5,
                        )

    def test_darwin_live_descendant_keeps_original_permission_error(self) -> None:
        child = self.child()  # The leader has exited, but its group has not.
        denied = self.denied()
        result = subprocess.CompletedProcess([], 0, stdout="4242 Z\n4242 S+\n101 S\n", stderr="")
        with patch.object(worker.sys, "platform", "darwin"), \
                patch.object(worker.os, "killpg", side_effect=denied), \
                patch.object(worker.subprocess, "run", return_value=result), \
                patch.object(worker.time, "monotonic", side_effect=count(0.0, 0.1).__next__), \
                patch.object(worker.time, "sleep", side_effect=[None] * 26) as pause, \
                self.assertRaises(PermissionError) as raised:
            worker._signal_group(child, 0)
        self.assertIs(raised.exception, denied)
        self.assertGreater(pause.call_count, 0)

    def test_running_leader_or_non_darwin_never_infers_an_empty_group(self) -> None:
        for platform, exited in [("darwin", False), ("linux", True), ("linux", False)]:
            with self.subTest(platform=platform, exited=exited):
                child, denied = self.child(exited=exited), self.denied()
                with patch.object(worker.sys, "platform", platform), \
                        patch.object(worker.os, "killpg", side_effect=denied), \
                        patch.object(worker.subprocess, "run") as ps, \
                        patch.object(worker.time, "monotonic", side_effect=count(0.0, 0.1).__next__), \
                        patch.object(worker.time, "sleep", side_effect=[None] * 26) as pause, \
                        self.assertRaises(PermissionError) as raised:
                    worker._signal_group(child, 0)
                self.assertIs(raised.exception, denied)
                ps.assert_not_called()
                if platform == "darwin":
                    self.assertGreater(pause.call_count, 0)
                else:
                    pause.assert_not_called()

    def test_darwin_exit_transition_retries_until_the_group_state_is_known(self) -> None:
        for transition in ["leader_to_esrch", "live_group_to_absent", "leader_to_signal_success"]:
            with self.subTest(transition=transition):
                child = self.child(exited=transition == "live_group_to_absent")
                denied = self.denied()
                later = (ProcessLookupError(errno.ESRCH, "gone") if transition == "leader_to_esrch"
                         else denied if transition == "live_group_to_absent" else None)
                listings = [
                    subprocess.CompletedProcess([], 0, stdout="4242 S+\n101 Ss\n", stderr=""),
                    subprocess.CompletedProcess([], 0, stdout="101 Ss\n", stderr=""),
                ]
                with patch.object(worker.sys, "platform", "darwin"), \
                        patch.object(worker.os, "killpg", side_effect=[denied, later]) as killpg, \
                        patch.object(worker.subprocess, "run", side_effect=listings) as ps, \
                        patch.object(worker.time, "monotonic", side_effect=count(0.0, 0.02).__next__), \
                        patch.object(worker.time, "sleep", side_effect=[None] * 26) as pause:
                    self.assertIs(worker._signal_group(child, signal.SIGTERM),
                                  transition == "leader_to_signal_success")
                self.assertEqual(killpg.call_args_list,
                                 [call(child.pid, signal.SIGTERM), call(child.pid, signal.SIGTERM)])
                pause.assert_called_once_with(0.02)
                self.assertEqual(ps.call_count, 2 if transition == "live_group_to_absent" else 0)

    def test_ps_failure_timeout_and_malformed_rows_cannot_hide_a_live_group(self) -> None:
        uncertain = [
            subprocess.CalledProcessError(1, ["/bin/ps"]),
            subprocess.TimeoutExpired(["/bin/ps"], 5),
            OSError("ps could not be started"),
            "",
            "\n",
            "not-a-pgid S\n",
            "101 S\nbroken\n",
            "101 S extra-column\n",
            "4242\n",
        ]
        for observation in uncertain:
            with self.subTest(observation=repr(observation)):
                child, denied = self.child(), self.denied()
                arguments = ({"side_effect": observation} if isinstance(observation, Exception)
                             else {"return_value": subprocess.CompletedProcess([], 0, stdout=observation, stderr="")})
                with patch.object(worker.sys, "platform", "darwin"), \
                        patch.object(worker.os, "killpg", side_effect=denied), \
                        patch.object(worker.subprocess, "run", **arguments), \
                        self.assertRaises(PermissionError) as raised:
                    worker._signal_group(child, 0)
                self.assertIs(raised.exception, denied)

    def test_other_os_errors_are_not_misclassified_as_darwin_empty_groups(self) -> None:
        for error in [PermissionError(errno.EACCES, "access denied"), OSError(errno.EIO, "I/O failure")]:
            with self.subTest(error=repr(error)):
                with patch.object(worker.sys, "platform", "darwin"), \
                        patch.object(worker.os, "killpg", side_effect=error), \
                        patch.object(worker.subprocess, "run") as ps, \
                        self.assertRaises(type(error)) as raised:
                    worker._signal_group(self.child(), signal.SIGTERM)
                self.assertIs(raised.exception, error)
                ps.assert_not_called()

    def test_stop_child_escalates_if_an_exited_leader_still_has_a_live_group(self) -> None:
        child = self.child()
        with patch.object(worker, "_signal_group", side_effect=[True, True, False]) as send, \
                patch.object(worker.time, "monotonic", side_effect=[0.0, 0.1, 4.0]), \
                patch.object(worker.time, "sleep"):
            worker.stop_child(child)
        self.assertEqual(send.call_args_list, [call(child, signal.SIGTERM), call(child, 0), call(child, signal.SIGKILL)])
        child.wait.assert_called_once()


class SupervisorTerminalStateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="videolingo-worker-signals-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.service = Service(self.root)
        self.job_id, self.run_id = uuid4().hex, uuid4().hex
        self.directory = self.root / "jobs" / self.job_id
        self.directory.mkdir()
        write_json(self.directory / "job.json", {
            "job_id": self.job_id, "status": "queued", "launched_at": time.time(),
            "error": None, "artifacts": [],
            "run": {"id": self.run_id, "operation": "export", "options": {
                "output_path": str(self.directory / "unpublished.mp4"),
            }},
        })

    def test_already_cancelled_job_does_not_start_an_execute_process(self) -> None:
        (self.directory / ".cancel").touch()
        with patch.object(worker.signal, "signal"), patch.object(worker.subprocess, "Popen") as launch:
            worker.supervise(self.root, self.job_id, self.run_id)
        launch.assert_not_called()
        persisted = read_json(self.directory / "job.json")
        self.assertEqual(persisted["status"], "cancelled")
        self.assertIsNone(persisted["error"])
        self.assertEqual(self.service.get_job(self.job_id)["status"], "cancelled")
        self.assertFalse(list(self.directory.glob("task-*.log")))

    def test_signal_received_during_initialization_does_not_start_execute(self) -> None:
        def deliver_signal(signum, handler):
            handler(signum, None)

        with patch.object(worker.signal, "signal", side_effect=deliver_signal), \
                patch.object(worker.subprocess, "Popen") as launch:
            worker.supervise(self.root, self.job_id, self.run_id)
        launch.assert_not_called()
        self.assertEqual(read_json(self.directory / "job.json")["status"], "interrupted")

    def test_cleanup_failures_still_persist_failed_and_preserve_the_primary_exception(self) -> None:
        primary = RuntimeError("primary execution failure")
        child = Mock(pid=4242)
        child.poll.side_effect = primary
        signal_cleanup = PermissionError(errno.EPERM, "signal cleanup denied")
        file_cleanup = OSError("partial-file cleanup denied")
        with patch.object(worker.signal, "signal"), \
                patch.object(worker.subprocess, "Popen", return_value=child), \
                patch.object(worker, "stop_child", side_effect=signal_cleanup) as stop, \
                patch.object(worker, "clean_temporary_exports", side_effect=file_cleanup) as clean, \
                self.assertRaises(RuntimeError) as raised:
            worker.supervise(self.root, self.job_id, self.run_id)
        self.assertIs(raised.exception, primary)
        stop.assert_called_once_with(child)
        clean.assert_called_once_with(self.directory)
        persisted = read_json(self.directory / "job.json")
        self.assertEqual(persisted["status"], "failed")
        for message in ["primary execution failure", "signal cleanup denied", "partial-file cleanup denied"]:
            self.assertIn(message, persisted["error"])
        self.assertEqual(self.service.get_job(self.job_id)["status"], "failed")
        self.assertFalse(persisted["artifacts"])


if __name__ == "__main__":
    unittest.main()
