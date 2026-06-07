import asyncio
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

import api_training


class FakeCursor:
    def __init__(self, rows=None, all_rows=None):
        self.rows = list(rows or [])
        self.all_rows = list(all_rows or [])
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        if self.rows:
            return self.rows.pop(0)
        return None

    def fetchall(self):
        if self.all_rows:
            rows = self.all_rows
            self.all_rows = []
            return rows
        return []


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self.cursor_obj


class ImmediateThread:
    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True
        self.target(*self.args)

    def join(self, timeout=None):
        return None


class NoopThread(ImmediateThread):
    def start(self):
        self.started = True


class FakeProcess:
    def __init__(self, return_code=0, stdout="", stderr=""):
        self.return_code = return_code
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.terminated = False

    def wait(self, timeout=None):
        return self.return_code

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True


class LongRunningFakeProcess(FakeProcess):
    def __init__(self, timeouts_before_exit=1, return_code=0):
        super().__init__(return_code=return_code, stdout="", stderr="")
        self.timeouts_before_exit = timeouts_before_exit
        self.wait_calls = 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls <= self.timeouts_before_exit:
            raise api_training.subprocess.TimeoutExpired(cmd="training", timeout=timeout)
        return self.return_code


class ApiTrainingTests(unittest.TestCase):
    def setUp(self):
        api_training.active_job_id = None

    def tearDown(self):
        api_training.active_job_id = None

    def test_post_train_existing_id_fetches_row_and_starts_background_training(self):
        row = {"id": "job-1", "dataset_url": "s3://bucket/data.csv", "status": "ready"}

        with (
            patch.object(api_training, "accept_training_job", return_value=row) as accept_job,
            patch.object(api_training, "Thread") as thread_cls,
        ):
            thread = MagicMock()
            thread_cls.return_value = thread

            response = asyncio.run(api_training.trigger_training(api_training.TrainingRequest(id="job-1")))

        self.assertEqual(response, {"message": "Training process started.", "id": "job-1", "status": "pending"})
        accept_job.assert_called_once()
        thread_cls.assert_called_once()
        _, kwargs = thread_cls.call_args
        self.assertEqual(kwargs["args"], ("s3://bucket/data.csv", "job-1"))
        thread.start.assert_called_once()

    def test_post_train_without_id_creates_db_row_when_dataset_url_is_provided(self):
        row = {"id": "job-2", "dataset_url": "s3://bucket/data.csv", "status": "pending"}

        with (
            patch.object(api_training, "accept_training_job", return_value=row) as accept_job,
            patch.object(api_training, "Thread") as thread_cls,
        ):
            thread = MagicMock()
            thread_cls.return_value = thread

            response = asyncio.run(
                api_training.trigger_training(
                    api_training.TrainingRequest(
                        by_user="user@example.com",
                        dataset_url="s3://bucket/data.csv",
                        version="1",
                    )
                )
            )

        self.assertEqual(response["id"], "job-2")
        self.assertEqual(response["status"], "pending")
        accept_job.assert_called_once()
        thread.start.assert_called_once()

    def test_post_train_rejects_missing_dataset_url(self):
        error = HTTPException(status_code=400, detail="ERROR: dataset_url is required when id is not provided.")
        with self.assertRaises(HTTPException) as caught:
            with patch.object(api_training, "accept_training_job", side_effect=error):
                asyncio.run(api_training.trigger_training(api_training.TrainingRequest()))

        self.assertEqual(caught.exception.status_code, 400)

    def test_post_train_rejects_existing_pending_or_in_progress_job(self):
        for status in ("pending", "in_progress"):
            api_training.active_job_id = None
            active_job = {"id": f"job-{status}", "status": status, "updatedAt": "2026-01-01T00:00:00+00:00"}
            error = api_training.make_active_job_conflict(active_job)

            with patch.object(api_training, "accept_training_job", side_effect=error):
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(api_training.trigger_training(api_training.TrainingRequest(id=active_job["id"])))

            self.assertEqual(caught.exception.status_code, 409)
            self.assertEqual(caught.exception.detail["detail"], "A training job is already running.")
            self.assertEqual(caught.exception.detail["active_job"]["status"], status)

    def test_two_post_train_calls_cannot_both_start_jobs(self):
        accepted = {"id": "job-first", "dataset_url": "s3://bucket/data.csv", "status": "pending"}
        active_job = {"id": "job-first", "status": "pending", "updatedAt": "2026-01-01T00:00:00+00:00"}
        conflict = api_training.make_active_job_conflict(active_job)

        with (
            patch.object(api_training, "accept_training_job", side_effect=[accepted, conflict]),
            patch.object(api_training, "Thread") as thread_cls,
        ):
            thread = MagicMock()
            thread_cls.return_value = thread

            first_response = asyncio.run(
                api_training.trigger_training(api_training.TrainingRequest(dataset_url="s3://bucket/data.csv"))
            )
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(
                    api_training.trigger_training(api_training.TrainingRequest(dataset_url="s3://bucket/other.csv"))
                )

        self.assertEqual(first_response["id"], "job-first")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.detail["active_job"]["id"], "job-first")
        thread.start.assert_called_once()

    def test_accept_training_job_blocks_pending_and_in_progress_jobs(self):
        for status in ("pending", "in_progress"):
            cursor = FakeCursor(rows=[{"id": "active", "status": status, "updatedAt": "now"}])

            with (
                patch.object(api_training, "TRAINING_TABLE_NAME", "Training"),
                patch.object(api_training, "get_db_connection", return_value=FakeConnection(cursor)),
            ):
                with self.assertRaises(HTTPException) as caught:
                    api_training.accept_training_job(
                        api_training.TrainingRequest(dataset_url="s3://bucket/data.csv")
                    )

            self.assertEqual(caught.exception.status_code, 409)
            self.assertEqual(caught.exception.detail["active_job"]["status"], status)

    def test_in_progress_job_with_fresh_updated_at_blocks_new_jobs(self):
        active_job = {"id": "fresh-active", "status": "in_progress", "updatedAt": "fresh"}
        cursor = FakeCursor(rows=[active_job])

        with (
            patch.object(api_training, "TRAINING_TABLE_NAME", "Training"),
            patch.object(api_training, "TRAINING_STALE_TIMEOUT_MINUTES", 120),
            patch.object(api_training, "get_db_connection", return_value=FakeConnection(cursor)),
        ):
            with self.assertRaises(HTTPException) as caught:
                api_training.accept_training_job(
                    api_training.TrainingRequest(dataset_url="s3://bucket/data.csv")
                )

        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.detail["active_job"]["id"], "fresh-active")

    def test_accept_training_job_allows_completed_or_failed_jobs(self):
        for status in ("completed", "failed"):
            cursor = FakeCursor(rows=[None, {"id": f"job-{status}", "dataset_url": "s3://bucket/data.csv", "status": "pending"}])

            with (
                patch.object(api_training, "TRAINING_TABLE_NAME", "Training"),
                patch.object(api_training, "get_db_connection", return_value=FakeConnection(cursor)),
            ):
                job = api_training.accept_training_job(
                    api_training.TrainingRequest(dataset_url="s3://bucket/data.csv")
                )

            self.assertEqual(job["status"], "pending")

    def test_accept_training_job_uses_advisory_lock(self):
        cursor = FakeCursor(rows=[None, {"id": "job-lock", "dataset_url": "s3://bucket/data.csv", "status": "pending"}])

        with (
            patch.object(api_training, "TRAINING_TABLE_NAME", "Training"),
            patch.object(api_training, "get_db_connection", return_value=FakeConnection(cursor)),
        ):
            api_training.accept_training_job(api_training.TrainingRequest(dataset_url="s3://bucket/data.csv"))

        self.assertIn(("model_training_global_lock",), [params for _, params in cursor.executed if params])

    def test_fail_stale_training_jobs_marks_active_rows_failed(self):
        cursor = FakeCursor(all_rows=[{"id": "stale", "status": "failed", "updatedAt": "now"}])

        with patch.object(api_training, "TRAINING_TABLE_NAME", "Training"):
            rows = api_training.fail_stale_training_jobs(cursor)

        self.assertEqual(rows, [{"id": "stale", "status": "failed", "updatedAt": "now"}])
        query, params = cursor.executed[0]
        self.assertEqual(params[0], "failed")
        self.assertIn("Training job marked failed", params[1])
        self.assertIn("pending", params)
        self.assertIn("in_progress", params)

    def test_stale_cleanup_disabled_does_not_fail_active_rows(self):
        cursor = FakeCursor(all_rows=[{"id": "stale", "status": "failed", "updatedAt": "now"}])

        with (
            patch.object(api_training, "TRAINING_TABLE_NAME", "Training"),
            patch.object(api_training, "TRAINING_STALE_TIMEOUT_MINUTES", 0),
        ):
            rows = api_training.fail_stale_training_jobs(cursor)

        self.assertEqual(rows, [])
        self.assertEqual(cursor.executed, [])

    def test_old_in_progress_job_can_be_failed_when_stale_cleanup_enabled(self):
        cursor = FakeCursor(all_rows=[{"id": "old-active", "status": "failed", "updatedAt": "now"}])

        with (
            patch.object(api_training, "TRAINING_TABLE_NAME", "Training"),
            patch.object(api_training, "TRAINING_STALE_TIMEOUT_MINUTES", 120),
        ):
            rows = api_training.fail_stale_training_jobs(cursor)

        self.assertEqual(rows[0]["id"], "old-active")
        _, params = cursor.executed[0]
        self.assertIn("in_progress", params)
        self.assertEqual(params[-1], 120)

    def test_get_train_status_reads_postgresql_row(self):
        row = {
            "id": "job-3",
            "by_user": "user@example.com",
            "dataset_url": "s3://bucket/data.csv",
            "model_url": None,
            "status": "completed",
            "log": "done",
            "version": "1",
            "createdAt": None,
            "updatedAt": None,
        }

        with patch.object(api_training, "fetch_training_job", return_value=row) as fetch_job:
            response = asyncio.run(api_training.get_training_status("job-3"))

        fetch_job.assert_called_once_with("job-3")
        self.assertEqual(response, row)

    def test_update_training_job_uses_updated_at_identifier_and_appends_text_log(self):
        cursor = FakeCursor(rows=[{"log": "old log"}, {"id": "job-4", "log": "new log"}])
        identifiers = []
        original_identifier = api_training.sql.Identifier

        def record_identifier(*names):
            identifiers.append(names)
            return original_identifier(*names)

        with (
            patch.object(api_training, "TRAINING_TABLE_NAME", "Training"),
            patch.object(api_training, "get_db_connection", return_value=FakeConnection(cursor)),
            patch.object(api_training.sql, "Identifier", side_effect=record_identifier),
        ):
            api_training.update_training_job("job-4", status="failed", log_message="new event")

        self.assertIn(("updatedAt",), identifiers)
        self.assertIn(("log",), identifiers)
        self.assertEqual(cursor.executed[-1][1][-1], "job-4")
        self.assertIn("old log\n", cursor.executed[-1][1][1])
        self.assertIn("new event", cursor.executed[-1][1][1])

    def test_touch_training_job_updated_at_updates_only_updated_at_without_log(self):
        cursor = FakeCursor(rows=[{"id": "job-touch"}])

        with (
            patch.object(api_training, "TRAINING_TABLE_NAME", "Training"),
            patch.object(api_training, "get_db_connection", return_value=FakeConnection(cursor)),
        ):
            api_training.touch_training_job_updated_at("job-touch")

        self.assertEqual(len(cursor.executed), 1)
        query, params = cursor.executed[0]
        self.assertEqual(params, ("job-touch",))
        self.assertIn("updatedAt", str(query))
        self.assertNotIn("log", str(query))

    def test_long_running_subprocess_heartbeats_updated_at_while_running(self):
        process = LongRunningFakeProcess(timeouts_before_exit=2, return_code=0)
        calls = []

        with (
            patch.object(api_training, "TRAINING_HEARTBEAT_INTERVAL_SECONDS", 1),
            patch.object(api_training, "TRAINING_HEARTBEAT_LOG_INTERVAL_SECONDS", 0),
            patch.object(api_training, "touch_training_job_updated_at", side_effect=lambda *args, **kwargs: calls.append((args, kwargs))),
        ):
            return_code = api_training.wait_for_child_with_heartbeat(process, "job-heartbeat")

        self.assertEqual(return_code, 0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[0] == ("job-heartbeat",) for call in calls))
        self.assertTrue(all(call[1]["log_message"] is None for call in calls))

    def test_heartbeat_can_log_without_logging_every_heartbeat(self):
        process = LongRunningFakeProcess(timeouts_before_exit=3, return_code=0)
        calls = []

        with (
            patch.object(api_training, "TRAINING_HEARTBEAT_INTERVAL_SECONDS", 10),
            patch.object(api_training, "TRAINING_HEARTBEAT_LOG_INTERVAL_SECONDS", 20),
            patch.object(api_training, "touch_training_job_updated_at", side_effect=lambda *args, **kwargs: calls.append((args, kwargs))),
        ):
            api_training.wait_for_child_with_heartbeat(process, "job-heartbeat-log")

        log_messages = [kwargs["log_message"] for _, kwargs in calls]
        self.assertEqual(log_messages, [None, "Training heartbeat: child process is still running.", None])

    def test_append_log_appends_timestamped_lines(self):
        first = api_training.append_log(None, "first")
        second = api_training.append_log(first, "second")

        self.assertIn("first", first)
        self.assertIn("first", second)
        self.assertIn("second", second)
        self.assertEqual(len(second.splitlines()), 2)

    def test_failed_subprocess_updates_status_failed(self):
        calls = []
        process = FakeProcess(return_code=1, stdout="", stderr="boom\n")

        with (
            patch.object(api_training, "resolve_dataset_url_to_local_file", return_value=Path("/tmp/data.csv")),
            patch.object(api_training.subprocess, "Popen", return_value=process),
            patch.object(api_training, "Thread", ImmediateThread),
            patch.object(api_training, "fetch_training_job", return_value={"id": "job-5", "status": "in_progress"}),
            patch.object(api_training, "update_training_job", side_effect=lambda *args, **kwargs: calls.append((args, kwargs))),
        ):
            api_training.training_child_process("s3://bucket/data.csv", "job-5")

        statuses = [kwargs.get("status") for _, kwargs in calls]
        self.assertIn("failed", statuses)
        self.assertTrue(any("return_code=1" in kwargs.get("log_message", "") for _, kwargs in calls))
        self.assertTrue(any("stderr" in kwargs.get("extra", {}).get("error", "") for _, kwargs in calls))

    def test_successful_subprocess_updates_status_completed(self):
        calls = []
        process = FakeProcess(return_code=0, stdout="", stderr="")

        with (
            patch.object(api_training, "resolve_dataset_url_to_local_file", return_value=Path("/tmp/data.csv")),
            patch.object(api_training.subprocess, "Popen", return_value=process),
            patch.object(api_training, "Thread", ImmediateThread),
            patch.object(api_training, "fetch_training_job", return_value={"id": "job-6", "status": "in_progress"}),
            patch.object(api_training, "get_model_artifact_url", return_value="models/project/best_model.pth"),
            patch.object(api_training, "update_training_job", side_effect=lambda *args, **kwargs: calls.append((args, kwargs))),
        ):
            api_training.training_child_process("s3://bucket/data.csv", "job-6")

        self.assertTrue(any(kwargs.get("status") == "completed" for _, kwargs in calls))
        self.assertTrue(any(kwargs.get("model_url") == "models/project/best_model.pth" for _, kwargs in calls))

    def test_second_job_cannot_be_accepted_while_first_child_process_is_alive(self):
        active_job = {"id": "alive-job", "status": "in_progress", "updatedAt": "fresh"}
        cursor = FakeCursor(rows=[active_job])

        with (
            patch.object(api_training, "TRAINING_TABLE_NAME", "Training"),
            patch.object(api_training, "TRAINING_STALE_TIMEOUT_MINUTES", 120),
            patch.object(api_training, "get_db_connection", return_value=FakeConnection(cursor)),
        ):
            with self.assertRaises(HTTPException) as caught:
                api_training.accept_training_job(
                    api_training.TrainingRequest(dataset_url="s3://bucket/second.csv")
                )

        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.detail["active_job"]["id"], "alive-job")

    def test_dataset_url_resolver_rejects_local_paths_unless_allowed(self):
        with (
            patch.object(api_training, "ALLOW_LOCAL_DATASET_PATHS", False),
            patch.object(api_training, "update_training_job"),
        ):
            with self.assertRaises(RuntimeError):
                api_training.resolve_dataset_url_to_local_file("/tmp/data.csv", "job-7")

    def test_dataset_url_resolver_allows_local_paths_when_enabled(self):
        with (
            patch.object(api_training, "ALLOW_LOCAL_DATASET_PATHS", True),
            patch.object(api_training, "update_training_job"),
            patch.object(api_training.shutil, "copyfile") as copy_file,
            patch.object(api_training, "validate_resolved_dataset_file") as validate_file,
            patch.object(Path, "exists", return_value=True),
        ):
            result = api_training.resolve_dataset_url_to_local_file("/tmp/data.csv", "job-8")

        self.assertEqual(result, api_training.DOWNLOAD_DIR / "job-8.csv")
        copy_file.assert_called_once()
        validate_file.assert_called_once_with(result)

    def test_dataset_url_resolver_downloads_s3_without_real_s3(self):
        s3_client = MagicMock()

        with (
            patch.object(api_training, "update_training_job"),
            patch.object(api_training.boto3, "client", return_value=s3_client) as boto_client,
            patch.object(api_training, "validate_resolved_dataset_file") as validate_file,
        ):
            result = api_training.resolve_dataset_url_to_local_file("s3://bucket/key.csv", "job-9")

        boto_client.assert_called_once_with("s3")
        s3_client.download_file.assert_called_once_with("bucket", "key.csv", str(result))
        validate_file.assert_called_once_with(result)


if __name__ == "__main__":
    unittest.main()
