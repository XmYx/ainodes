import threading


class Singleton:
    _instance = None
    _lock = threading.Lock()

    flask_running = False


    def __new__(cls, *args: object, **kwargs: object) -> object:
        if not cls._instance:
            with cls._lock:
                # another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if not cls._instance:
                    cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance