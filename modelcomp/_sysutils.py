import os


def check_dir_read(path, raise_errors=True):
    if not os.path.exists(path):
        if raise_errors:
            raise FileNotFoundError(f"Path {path} not found")
        return False
    elif not os.path.isdir(path):
        if raise_errors:
            raise ValueError(f"Path {path} is not a directory")
        return False
    elif not os.access(path, os.R_OK):
        if raise_errors:
            raise PermissionError(f"Permission denied to read from {path}")
        return False
    return True


def check_file_read(path, raise_errors=True):
    if not os.path.exists(path):
        if raise_errors:
            raise FileNotFoundError(f"Path {path} not found")
        return False
    elif not os.path.isfile(path):
        if raise_errors:
            raise ValueError(f"Path {path} is not a file")
        return False
    elif not os.access(path, os.R_OK):
        if raise_errors:
            raise PermissionError(f"Permission denied to read from {path}")
        return False
    return True


def check_dir_write(path, force_create=False, raise_errors=True):
    if force_create:
        os.makedirs(path, exist_ok=True)
    if not force_create and not os.path.exists(path):
        if raise_errors:
            raise FileNotFoundError(f"Path {path} not found")
        return False
    if not os.access(path, os.W_OK):
        if raise_errors:
            raise PermissionError(f"Permission denied to write to {path}")
        return False
    return True


def check_file_write(path, force_create=False, raise_errors=True):
    if force_create:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if not force_create and not os.path.exists(path):
        if raise_errors:
            raise FileNotFoundError(f"Path {path} not found")
        return False
    if not os.access(path, os.W_OK):
        if raise_errors:
            raise PermissionError(f"Permission denied to write to {path}")
        return False
    return True
