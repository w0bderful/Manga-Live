"""Short blocking service requests without keeping the application alive."""
from concurrent.futures import Future
import threading


def submit_background(function, *args, name):
    future = Future()

    def run():
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = function(*args)
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    threading.Thread(target=run, daemon=True, name=name).start()
    return future
