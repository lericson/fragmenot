import threading
import logging


log = logging.getLogger(__name__)


class Thread(threading.Thread):
    def run(self):
        log.debug('[%s] thread started', self.name)
        try:
            self._return = self._target(*self._args, **self._kwargs)
        except Exception:
            log.exception('[%s] thread aborted by exception', self.name)
        else:
            log.debug('[%s] thread finished', self.name)
        finally:
            del self._target, self._args, self._kwargs


def threadable(f):
    fn = f'{f.__module__}.{f.__name__}'
    def thread(*args, daemon=False, **kwds):
        name = threading._newname(template=f'{fn}:%d')
        return Thread(target=f, args=args, kwargs=kwds,
                      name=name, daemon=daemon)
    f.thread = thread
    return f
