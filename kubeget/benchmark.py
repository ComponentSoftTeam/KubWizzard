import threading
import time


class Benchmarks:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._lock = threading.RLock()
                    cls._instance = super(Benchmarks, cls).__new__(cls, *args, **kwargs)
                    cls._instance._data = {}
                    cls._instance._lock = threading.RLock()
        return cls._instance

    def __getitem__(self, key):
        with self._lock:
            return self._data[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._data[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._data

    def __len__(self):
        with self._lock:
            return len(self._data)

    def keys(self):
        with self._lock:
            return self._data.keys()

    def values(self):
        with self._lock:
            return self._data.values()

    def items(self):
        with self._lock:
            return self._data.items()

    def pop(self, *args, **kwargs):
        with self._lock:
            return self._data.pop(*args, **kwargs)

    def clear(self, *args, **kwargs):
        with self._lock:
            return self._data.clear(*args, **kwargs)
        
def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        fn = func.__name__
        bench = Benchmarks()
        with bench._lock:
            
            if fn not in bench:
                total = 0
                n = 0
                bench[fn] = {}
            else:
                record = bench[fn]
                total = record['total']
                n = record['n']

            total += end - start
            n += 1

            bench[fn]['total'] = total
            bench[fn]['n'] = n

        return result
    return wrapper