#!/usr/bin/env python3
"""Bundle the AMD ROCm userspace runtime into an installed Gufo tree.

The no-Nix Ubuntu build links against /opt/rocm. This script copies the
transitive ROCm library closure into the install tree, preserving the ROCm
relative layout — the shipped libraries resolve their own dependencies
through $ORIGIN-based RPATHs — repoints the Gufo RUNPATH at the bundle, and
verifies that nothing resolves into the system ROCm anymore. Deployment
machines then only need the amdgpu kernel driver.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

ROCM_PREFIX = Path('/opt/rocm')
BUNDLE_RPATH = '$ORIGIN/../lib'

# Library trees the shipped RPATHs reference beside the libraries. The
# libraries themselves arrive through the ldd closure; only their data is
# copied here (most importantly the amdgpu.ids table the sysdeps DRM library
# reads next to itself).
DATA_TREES = ('rocm_sysdeps/share/libdrm', 'rocblas/library', 'hipblaslt/library')

LDD_RESOLVED = re.compile(r'^\s*\S+\s+=>\s+(/\S+?)\s+\(0x')
LDD_DIRECT = re.compile(r'^\s*(/\S+?)\s+\(0x')
LDD_NOT_FOUND = re.compile(r'^\s*(\S+)\s+=>\s+not found')

# AMD license directories to carry into the bundle, keyed by bundled library
# name; glob entries match the versioned Debian doc directory names.
LICENSE_PACKAGES = {
    'libamd_comgr.so': ['amd_comgr'],
    'libamdhip64.so': ['hip', 'rocr'],
    'libclang-cpp.so': ['amdrocm-llvm*'],
    'libhipblas.so': ['hipblas'],
    'libhipblaslt.so': ['hipblaslt'],
    'libhsa-runtime64.so': ['rocr'],
    'libLLVM.so': ['amdrocm-llvm*'],
    'liborigami.so': ['origami'],
    'librocblas.so': ['rocblas'],
    'librocm_kpack.so': ['rocm-core'],
    'librocprofiler-register.so': ['rocprofiler-register'],
    'librocroller.so': ['rocroller'],
    'librocm_sysdeps.so': ['amdrocm-sysdeps*'],
    'libroctx64.so': ['roctracer'],
    'librocsolver.so': ['rocsolver'],
}
REQUIRED_LICENSES = {'rocm-core', 'rocr', 'rocblas', 'hipblas', 'hipblaslt', 'amd_comgr'}


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def ldd_dependencies(elf):
    """Return (resolved paths, not-found sonames) of an ELF per ldd."""
    resolved, not_found = [], []
    for line in subprocess.check_output(['ldd', str(elf)], text=True).splitlines():
        missing = LDD_NOT_FOUND.match(line)
        entry = LDD_RESOLVED.match(line) or LDD_DIRECT.match(line)
        if missing:
            not_found.append(missing.group(1))
        elif entry:
            resolved.append(Path(entry.group(1)))
    return resolved, not_found


def rocm_candidate(soname):
    for directory in ('lib', 'lib/llvm/lib', 'lib/rocm_sysdeps/lib', 'lib/host-math/lib'):
        path = ROCM_PREFIX / directory / soname
        if path.exists():
            return path
    return None


def bundle_relative(path):
    """Map a /opt/rocm path to its install-tree location, keeping the layout."""
    relative = path.relative_to(ROCM_PREFIX).as_posix()
    if re.match(r'core-[^/]+/', relative):
        relative = relative.split('/', 1)[1]
    if not relative.startswith('lib/'):
        raise ValueError(f'unexpected ROCm path outside lib/: {path}')
    return relative


def copy_rocm_file(source, root):
    """Copy one ROCm file (and, for symlinks, its target) keeping the layout."""
    items = [source.resolve(strict=True)]
    if source.is_symlink():
        items.append(source)
    for item in items:
        destination = root / bundle_relative(item)
        if destination.exists() or destination.is_symlink():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if item.is_symlink():
            destination.symlink_to(os.readlink(item))
        else:
            shutil.copy2(item, destination)


def collect_closure(executable, root):
    """BFS the ROCm library closure from the installed executable."""
    queue, visited = [], set()
    resolved, not_found = ldd_dependencies(executable)
    for soname in not_found:
        candidate = rocm_candidate(soname)
        if candidate is None:
            raise ValueError(f'{executable}: {soname} resolves nowhere; cannot bundle')
        resolved.append(candidate)
    queue.extend(path for path in resolved if path.is_relative_to(ROCM_PREFIX))
    while queue:
        source = queue.pop()
        real = source.resolve(strict=True)
        if real in visited:
            continue
        visited.add(real)
        copy_rocm_file(source, root)
        resolved, not_found = ldd_dependencies(real)
        if not_found:
            raise ValueError(f'{real}: unbundled dependencies {not_found}')
        queue.extend(path for path in resolved if path.is_relative_to(ROCM_PREFIX))
    return len(visited)


def copy_data_trees(root):
    tensile = {'lib/rocblas/library', 'lib/hipblaslt/library'}
    for tree in [f'lib/{name}' for name in DATA_TREES]:
        source = ROCM_PREFIX / tree
        if not source.exists():
            if tree in tensile:
                raise ValueError(f'missing required Tensile data directory: {source}')
            continue
        # The closure pass may already have created parent directories.
        shutil.copytree(source, root / tree, symlinks=True, dirs_exist_ok=True)


def copy_licenses(root):
    destination = root / 'share/licenses/gufo/rocm'
    destination.mkdir(parents=True, exist_ok=True)
    doc = ROCM_PREFIX / 'share/doc'
    notices = doc / 'NOTICES.txt'
    if not notices.is_file():
        raise ValueError(f'missing AMD notices file: {notices}')
    shutil.copy2(notices, destination / 'NOTICES.txt')
    bundled = {path.name for path in (root / 'lib').rglob('*')}
    missing = set(REQUIRED_LICENSES)
    for prefix, packages in LICENSE_PACKAGES.items():
        if not any(name == prefix or name.startswith(prefix + '.') for name in bundled):
            continue
        for package in packages:
            licenses = sorted(doc.glob(f'{package}/LICENSE*'))
            if not licenses:
                continue
            missing.discard(package)
            for license_file in licenses:
                shutil.copy2(license_file, destination / f'{package}-{license_file.name}')
    if missing:
        raise ValueError(f'missing required AMD license files for {sorted(missing)}')


def verify(executable, root):
    rpath = output('patchelf', '--print-rpath', str(executable))
    if rpath != BUNDLE_RPATH:
        raise ValueError(f'unexpected RUNPATH {rpath!r}; expected {BUNDLE_RPATH!r}')
    sonames = re.compile(r'\.so(\.[0-9]+)?$')
    elks = [executable]
    for path in sorted((root / 'lib').rglob('*')):
        if not sonames.search(path.name) or path.is_symlink() or not path.is_file():
            continue
        with path.open('rb') as file:
            if file.read(4) != b'\x7fELF':
                continue
        elks.append(path)
    for elf in elks:
        linked = subprocess.check_output(
            ['env', '-u', 'LD_LIBRARY_PATH', 'ldd', str(elf)],
            text=True, stderr=subprocess.DEVNULL)
        for line in linked.splitlines():
            if 'not found' in line:
                raise ValueError(f'{elf}: {line.strip()}')
            for path in re.findall(r'(/\S+?)\s+\(0x', line):
                if path.startswith(str(ROCM_PREFIX)):
                    raise ValueError(f'{elf} still resolves into the system ROCm: {line.strip()}')
                if not (path.startswith(str(root))
                        or path.startswith(('/lib', '/usr/lib', '/lib64', '/etc'))):
                    raise ValueError(f'{elf} resolves outside the bundle and base system: {line.strip()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', type=Path, required=True,
                        help='installed Gufo tree (the gufo/ directory)')
    args = parser.parse_args()
    root = args.install.resolve()
    executable = root / 'bin/gufo'
    if not executable.is_file():
        parser.error(f'{executable} must be the ELF binary of an installed tree')

    libraries = collect_closure(executable, root)
    copy_data_trees(root)
    copy_licenses(root)
    subprocess.run(['patchelf', '--set-rpath', BUNDLE_RPATH, str(executable)], check=True)
    verify(executable, root)

    size = sum(path.stat().st_size for path in (root / 'lib').rglob('*') if path.is_file())
    print(f'Bundled ROCm userspace: {libraries} libraries, '
          f'{size / (1 << 20):.0f} MiB in {root / "lib"}', flush=True)


if __name__ == '__main__':
    main()
