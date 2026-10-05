# -*- coding: utf-8 -*-
"""
Pyomo pseudo-solver 'local.gjh': evaluates the objective gradient and the
constraint Jacobian of a model at its current point with AMPL's gjh, and
installs gjh automatically (from AMPL's package index) when it is missing.

@author: Gul Hameed
"""
import logging
import os
import shutil
import stat
from pyomo.opt.base import SolverFactory
from pyomo.solvers.plugins.solvers.ASL import ASL
from pyomo.common.tempfiles import TempfileManager
from readgjh import readgjh

logger = logging.getLogger('local.gjh')

# Name of gjh executable
GJH_FILENAME = "gjh.exe" if os.name == "nt" else "gjh"
# Directory of THIS module: a downloaded gjh is placed next to it, so the
# lookup must not depend on the user's current working directory.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# AMPL now distributes gjh via its PyPI index as 'ampl-module-gjh' wheels
# (a wheel is a zip containing ampl_module_gjh/bin/gjh[.exe]). The legacy
# ampl.com/dl/open and portal.ampl.com zip paths are dead (they soft-404 to
# HTML docs pages; verified 2026-06).
_GJH_PYPI_INDEX = 'https://pypi.ampl.com/ampl-module-gjh'
# Pinned fallback in case index parsing ever breaks:
_GJH_FALLBACK_WHEEL = {
    'win_amd64': 'ampl_module_gjh-20241202-py3-none-win_amd64.whl',
    'manylinux2010_x86_64':
        'ampl_module_gjh-20241202-py3-none-manylinux2010_x86_64.whl',
    'manylinux2014_aarch64':
        'ampl_module_gjh-20241202-py3-none-manylinux2014_aarch64.whl',
    'macosx_10_9_universal2':
        'ampl_module_gjh-20241202-py3-none-macosx_10_9_universal2.whl',
}
# AMPL's portal rejects the default Python urllib User-Agent with 403
_HTTP_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0 Safari/537.36')}


def _is_valid_exe(path):
    """A usable executable exists and is non-empty (a zero-byte file can be
    left behind by an interrupted download -- treat it as missing)."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def find_gjh():
    """Locate a gjh executable. Search order:
    1. this module's directory (a copy shipped with the repository, or the
       one placed there by a download), so that a run uses the same gjh as
       the results it is meant to reproduce;
    2. the system PATH;
    3. the Pyomo configuration bin directory (where a download lands when
       the module directory is not writable).
    Returns the absolute path, or None.
    """
    local = os.path.join(_MODULE_DIR, GJH_FILENAME)
    if _is_valid_exe(local):
        return local

    exe = shutil.which("gjh")
    if exe and _is_valid_exe(exe):
        return os.path.abspath(exe)

    try:
        from pyomo.common.envvar import PYOMO_CONFIG_DIR
        cfg = os.path.join(PYOMO_CONFIG_DIR, 'bin', GJH_FILENAME)
        if _is_valid_exe(cfg):
            return cfg
    except ImportError:
        pass

    return None


def _platform_wheel_tag():
    """Wheel platform tag for this machine."""
    import platform
    import sys
    if sys.platform.startswith('win') or sys.platform.startswith('cygwin'):
        return 'win_amd64'
    if sys.platform.startswith('darwin'):
        return 'macosx_10_9_universal2'
    if platform.machine().lower() in ('aarch64', 'arm64'):
        return 'manylinux2014_aarch64'
    return 'manylinux2010_x86_64'


def _http_get(url, timeout=120):
    from urllib.request import Request, urlopen
    with urlopen(Request(url, headers=_HTTP_HEADERS), timeout=timeout) as r:
        return r.read()


def _resolve_gjh_wheel_url():
    """Find the newest ampl-module-gjh wheel for this platform on the AMPL
    PyPI index; fall back to a pinned known-good wheel name."""
    import re
    tag = _platform_wheel_tag()
    try:
        page = _http_get(_GJH_PYPI_INDEX, timeout=60).decode(
            'utf-8', errors='replace')
        wheels = sorted(set(re.findall(
            r"href='([^']*ampl_module_gjh-[^']*%s\.whl)'" % re.escape(tag),
            page)))
        if wheels:
            # filenames embed YYYYMMDD versions, so lexicographic max = newest
            return _GJH_PYPI_INDEX + '/' + wheels[-1].lstrip('/')
    except Exception as e:
        logger.warning("gjh index lookup failed (%s); using pinned wheel", e)
    return _GJH_PYPI_INDEX + '/' + _GJH_FALLBACK_WHEEL[tag]


def _fetch_gjh_from_zip(url, member_name):
    """Download the zip/wheel at `url` and return the raw bytes of the
    archive member ending in `member_name` (wheels nest it under
    ampl_module_gjh/bin/)."""
    import io
    import zipfile

    payload = _http_get(url)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        # exact name, else any entry ending with it (archives may nest dirs)
        if member_name in names:
            return zf.read(member_name)
        for n in names:
            if n.endswith('/' + member_name) or n.endswith(member_name):
                return zf.read(n)
    raise RuntimeError(f"'{member_name}' not found in archive ({names})")


def download_gjh(dest_dir=None):
    """Download the precompiled AMPL gjh binary for this platform.

    Tries the repo directory first (keeps the project self-contained), then
    the Pyomo config bin directory. The archive is fetched fully into memory
    and the executable written to a temp name, moved into place only on
    success -- so a failed download can NEVER truncate an existing gjh.
    Returns the absolute path of the installed executable.
    """
    import sys

    candidates = ([dest_dir] if dest_dir is not None else [])
    candidates.append(_MODULE_DIR)
    try:
        from pyomo.common.envvar import PYOMO_CONFIG_DIR
        candidates.append(os.path.join(PYOMO_CONFIG_DIR, 'bin'))
    except ImportError:
        pass

    url = _resolve_gjh_wheel_url()
    member = 'gjh.exe' if (sys.platform.startswith('win')
                           or sys.platform.startswith('cygwin')) else 'gjh'

    logger.info("Fetching GJH from %s", url)
    print(f"[gjh] downloading from {url}")
    exe_bytes = _fetch_gjh_from_zip(url, member)
    if not exe_bytes:
        raise RuntimeError("downloaded gjh archive contained an empty file")

    last_err = None
    for d in candidates:
        tmp_dest = None
        try:
            os.makedirs(d, exist_ok=True)
            dest = os.path.join(d, GJH_FILENAME)
            tmp_dest = dest + '.download'
            with open(tmp_dest, 'wb') as f:
                f.write(exe_bytes)
            mode = os.stat(tmp_dest).st_mode
            os.chmod(tmp_dest, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            os.replace(tmp_dest, dest)
            print(f"[gjh] installed to {dest} ({len(exe_bytes)} bytes)")
            return dest
        except Exception as e:           # destination not writable: try next
            last_err = e
            if tmp_dest and os.path.isfile(tmp_dest):
                try:
                    os.remove(tmp_dest)
                except OSError:
                    pass
            continue

    raise RuntimeError(
        "Could not install the GJH solver automatically "
        f"(last error: {last_err}). Please install it manually, e.g. "
        "'pip install ampl-module-gjh --index-url https://pypi.ampl.com', "
        "or place a gjh executable on your PATH or next to this file "
        f"({_MODULE_DIR}).")


def ensure_gjh(download_if_missing=True, verbose=False):
    """Make sure a gjh executable is available BEFORE a solver run starts.

    Resolution order: copy next to this module > PATH installation >
    Pyomo config dir > fresh download. The executable's directory is then
    prepended to PATH for this process, so the ASL 'gjh' lookup always
    succeeds regardless of how/where the run was started.
    """
    exe = find_gjh()
    if exe is None:
        if not download_if_missing:
            raise RuntimeError(
                "GJH solver not found on PATH, in the repo directory, or in "
                "the Pyomo config dir, and download_if_missing=False.")
        exe = download_gjh()

    exe_dir = os.path.dirname(exe)
    path_entries = os.environ.get('PATH', '').split(os.pathsep)
    if exe_dir and exe_dir not in path_entries:
        os.environ['PATH'] = exe_dir + os.pathsep + os.environ.get('PATH', '')

    if verbose:
        print(f"[gjh] using: {exe}")
    return exe


@SolverFactory.register('local.gjh', doc='Local GJH solver (PATH, repo dir, '
                                          'Pyomo config dir, or downloaded)')
class LocalGJHSolver(ASL):
    def __init__(self, **kwds):
        kwds['type'] = 'gjh'
        kwds['symbolic_solver_labels'] = True
        super().__init__(**kwds)
        self.options.solver = 'gjh'
        self._metasolver = False

    def available(self, exception_flag=True):
        """Resolve gjh via find_gjh() (module dir > PATH > Pyomo config dir)."""
        exe = find_gjh()
        if exe:
            self.options.solver = exe
            return super().available(exception_flag=exception_flag)

        # Not found anywhere
        if exception_flag:
            raise RuntimeError(
                f"GJH solver not found! Tried the module directory ({_MODULE_DIR}), "
                "the system PATH and the Pyomo config dir. Run "
                "GJHPseudoSolver.ensure_gjh() to download it automatically.")
        return False

    def _initialize_callbacks(self, model):
        self._model = model
        self._model._gjh_info = None
        super()._initialize_callbacks(model)

    def _presolve(self, *args, **kwds):
        super()._presolve(*args, **kwds)
        self._gjh_file = self._soln_file[:-3] + 'gjh'
        TempfileManager.add_tempfile(self._gjh_file, exists=False)

    def _postsolve(self):
        if not os.path.exists(self._gjh_file) or \
           not os.path.exists(self._gjh_file[:-3] + 'col') or \
           not os.path.exists(self._gjh_file[:-3] + 'row'):
            raise RuntimeError(
                "GJH failed to produce .gjh, .col, or .row files. "
                "Check the solver log."
            )
        self._model._gjh_info = readgjh(self._gjh_file)
        self._model = None
        return super()._postsolve()

