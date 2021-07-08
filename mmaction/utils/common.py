import os, tempfile
from contextlib import contextmanager

# only support UNIX-like os
@contextmanager
def tempFileName():
    with tempfile.NamedTemporaryFile() as f:
        yield f.name
