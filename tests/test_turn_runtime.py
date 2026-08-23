import asyncio
import unittest

from turn_runtime import TurnCoordinator, TurnKind


class TurnCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupt_releases_cancelled_turn(self) -> None:
        coordinator = TurnCoordinator()
        task = coordinator.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
            goal_id="goal-1",
        )

        result = await coordinator.interrupt()

        self.assertTrue(result.interrupted)
        self.assertTrue(result.completed)
        self.assertTrue(task.cancelled())

    async def test_stale_interrupt_does_not_cancel_new_owner(self) -> None:
        coordinator = TurnCoordinator()
        task = coordinator.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
        )

        result = await coordinator.interrupt(expected_turn_id="stale")

        self.assertFalse(result.interrupted)
        self.assertFalse(result.completed)
        self.assertFalse(task.cancelled())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_second_turn_is_rejected_without_leaking_coroutine(self) -> None:
        coordinator = TurnCoordinator()
        task = coordinator.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
        )

        with self.assertRaisesRegex(RuntimeError, "already running"):
            coordinator.start(
                asyncio.sleep(0),
                kind=TurnKind.REGULAR,
            )

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_failed_prestart_does_not_claim_turn(self) -> None:
        coordinator = TurnCoordinator()

        def fail(_snapshot):
            raise OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            coordinator.start(
                asyncio.sleep(0),
                kind=TurnKind.REGULAR,
                before_start=fail,
            )

        self.assertFalse(coordinator.is_running())
        self.assertIsNone(coordinator.snapshot)


if __name__ == "__main__":
    unittest.main()
