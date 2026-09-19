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
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
LIMIT = 1250 * 1024**2


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def reusable_runtime(manifest_path, records):
    if not manifest_path.is_file():
        return None
    try:
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if (isinstance(previous, dict) and previous.get('entrypoint') == '_internal/main.py'
            and previous.get('files') == records and previous.get('assets')):
        return previous
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launcher-only', action='store_true', help='기존 매니페스트로 실행기만 다시 빌드')
    parser.add_argument('--relocate-runtime', action='store_true', help='기존 런타임 내용은 유지하고 다운로드 주소만 변경')
    parser.add_argument('--source', default='output/integrated-runtime/Manga Live')
    parser.add_argument('--asset-base-url', required=True, help='런타임 ZIP을 제공하는 HTTP(S) 디렉터리 주소')
    args = parser.parse_args()
    source = (ROOT/args.source).resolve()
    output = ROOT/'output/downloader'
    work = ROOT/'build/downloader'
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if args.relocate_runtime:
        base = args.asset_base_url.rstrip('/')
        parsed = urlsplit(base)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise SystemExit('인증 정보·쿼리 없는 HTTP(S) 런타임 주소를 지정하세요.')
        manifest_path = work/'runtime-manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for asset in manifest['assets']:
            asset['url'] = base + '/' + asset['name']
        from json_storage import write_object
        write_object(manifest_path, manifest)
        build_launcher(manifest_path, output, work)
        return
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
    records = {name: {'size':path.stat().st_size, 'sha256':digest(path)} for path,name in files}
    manifest_path = work/'runtime-manifest.json'
    if reusable_runtime(manifest_path, records) is not None:
        print('런타임 내용 변경 없음: 기존 다운로드 주소를 재사용합니다. 런타임 재업로드는 필요하지 않습니다.', flush=True)
        build_launcher(manifest_path, output, work)
        return
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
        group_records = {}
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path, member in groups[index]:
                group_records[member] = records[member]
                archive.write(path,member)
        if target.stat().st_size >= 2*1024**3:
            raise ValueError('GitHub 릴리스 파일 크기 제한 초과')
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP 검증 실패')
        print(f'검증 완료: {name} ({target.stat().st_size} bytes)',flush=True)
        return {'name':name,'url':args.asset_base_url.rstrip('/')+'/'+name,
                'size':target.stat().st_size,'sha256':digest(target)}, group_records

    manifest = {'entrypoint':'_internal/main.py','assets':[],'files':{}}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for asset, group_records in pool.map(package,range(len(groups))):
            manifest['assets'].append(asset)
            manifest['files'].update(group_records)
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
