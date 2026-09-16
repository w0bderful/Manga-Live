"""Package external libraries for the single Manga Live executable."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
LIMIT = 1250 * 1024**2


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launcher-only', action='store_true', help='기존 매니페스트로 실행기만 다시 빌드')
    parser.add_argument('--source', default='output/integrated-runtime/Manga Live')
    parser.add_argument('--asset-base-url', required=True, help='공개할 런타임 릴리스의 /download/<tag> 주소')
    args = parser.parse_args()
    source = (ROOT/args.source).resolve()
    output = ROOT/'output/downloader'
    work = ROOT/'build/downloader'
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if args.launcher_only:
        build_launcher(work/'runtime-manifest.json',output,work)
        return
    if not (source/'Manga Live.exe').is_file():
        raise SystemExit('먼저 build_exe.py --runtime --output output/integrated-runtime 를 실행하세요.')
    if not (source/'_internal'/'main.py').is_file() or not (source/'_internal'/'runtime-hooks').is_dir():
        raise SystemExit('build_exe.py --runtime 옵션으로 라이브러리를 먼저 생성하세요.')
    # Installer/update modules are bundled in the integrated EXE. Do not ship a
    # second copy that could shadow the EXE's current implementation.
    embedded = {'bootstrap.pyc', 'self_update.pyc'}
    files = [(p, p.relative_to(source).as_posix()) for p in sorted((source/'_internal').rglob('*'))
             if p.is_file() and not (p.parent == source/'_internal' and p.name in embedded)]
    forbidden = {'api-keys.json','settings.json','.env','AGENTS.md'}
    if any(p.name in forbidden or p.suffix == '.log' for p,_ in files):
        raise ValueError('배포에 포함할 수 없는 파일이 있습니다.')
    groups, current, total = [], [], 0
    for path, name in files:
        size = path.stat().st_size
        if size >= LIMIT:
            raise ValueError(f'개별 파일이 너무 큽니다: {name}')
        if current and total+size > LIMIT:
            groups.append(current); current=[]; total=0
        current.append((path,name)); total += size
    if current:
        groups.append(current)

    def package(index):
        name = f'Manga-Live-runtime-{index+1:03}.zip'
        target = output/name
        records = {}
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path, member in groups[index]:
                records[member] = {'size':path.stat().st_size, 'sha256':digest(path)}
                archive.write(path,member)
        if target.stat().st_size >= 2*1024**3:
            raise ValueError('GitHub 릴리스 파일 크기 제한 초과')
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP 검증 실패')
        print(f'검증 완료: {name} ({target.stat().st_size} bytes)',flush=True)
        return {'name':name,'url':args.asset_base_url.rstrip('/')+'/'+name,
                'size':target.stat().st_size,'sha256':digest(target)}, records

    manifest = {'entrypoint':'_internal/main.py','assets':[],'files':{}}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for asset, records in pool.map(package,range(len(groups))):
            manifest['assets'].append(asset)
            manifest['files'].update(records)
    manifest_path = work/'runtime-manifest.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    build_launcher(manifest_path,output,work)


def build_launcher(manifest_path, output, work):
    if not manifest_path.is_file():
        raise SystemExit('먼저 런타임 패키지를 생성하세요.')
    windows = Path(os.environ.get('SystemRoot',r'C:\Windows'))
    os.environ['PATH'] = os.pathsep.join(map(str,(Path(sys.executable).parent,
        Path(sys.base_prefix),windows/'System32',windows)))
    from auto_py_to_exe import config, packaging
    config.temporary_directory = str(work/'packaging')
    Path(config.temporary_directory).mkdir(exist_ok=True)
    options = {'outputDirectory':str(output),'increaseRecursionLimit':True,'manualArguments':''}
    command = ['pyinstaller',str(ROOT/'bootstrap.py'),'--onefile','--windowed','--clean','--noconfirm',
        '--noupx','--name','Manga Live','--icon',str(ROOT/'assets/manga-live.ico'),
        '--add-data',manifest_path.as_posix()+':.']
    if not packaging.package(shlex.join(command),options):
        raise SystemExit('실행기 빌드 실패')
    executable = output/'Manga Live.exe'
    if not executable.is_file():
        raise SystemExit('실행 파일이 생성되지 않았습니다.')
    print(f'완료: {executable} ({executable.stat().st_size} bytes)',flush=True)


if __name__ == '__main__':
    main()
