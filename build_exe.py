"""Reproduce the auto-py-to-exe folder build using its packaging backend."""
import json
import logging
import os
from pathlib import Path
import shlex
import sys
import argparse
import ast
import shutil

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='프로젝트 내부의 빌드 출력 폴더')
    parser.add_argument('--runtime', action='store_true', help='통합 EXE가 불러올 라이브러리 패키지 생성')
    args = parser.parse_args()
    # Avoid bundling unrelated DLLs from tools added to the calling shell's PATH.
    windows = Path(os.environ.get('SystemRoot', r'C:\Windows'))
    os.environ['PATH'] = os.pathsep.join(map(str, (
        Path(sys.executable).parent, Path(sys.base_prefix), windows/'System32', windows,
    )))
    from auto_py_to_exe import config, packaging
    os.chdir(ROOT)
    settings = json.loads((ROOT/'auto-py-to-exe.json').read_text(encoding='utf-8'))
    options = settings['nonPyinstallerOptions']
    if args.output:
        options['outputDirectory'] = args.output
    values = {entry['optionDest']:entry['value'] for entry in settings['pyinstallerOptions']}
    output = (ROOT/options['outputDirectory']).resolve()
    target = (output/values['name']).resolve()
    if not output.is_relative_to(ROOT) or not target.is_relative_to(output):
        raise ValueError('빌드 출력은 프로젝트 폴더 안에 있어야 합니다.')
    if any((target/name).exists() for name in ('api-keys.json','settings.json','logs','.models')):
        raise RuntimeError('기존 실행 파일 폴더에 설정·로그·모델이 있습니다. 폴더를 별도로 보관한 뒤 빌드하세요.')
    work = ROOT/'build/auto-py-to-exe'
    work.mkdir(parents=True,exist_ok=True)
    config.temporary_directory = str(work)
    for entry in settings['pyinstallerOptions']:
        if entry['optionDest'] in ('filenames','icon_file'):
            entry['value'] = (ROOT/entry['value']).as_posix()
            values[entry['optionDest']] = entry['value']
    extra = shlex.split(options.get('manualArguments',''))
    if args.runtime:
        extra.extend(['--debug', 'noarchive', '--add-data', 'main.py:.'])
    for index,argument in enumerate(extra):
        if argument in ('--add-data','--add-binary'):
            source,destination = extra[index+1].rsplit(':',1)
            extra[index+1] = (ROOT/source).as_posix()+':'+destination
    options['manualArguments'] = shlex.join(extra)
    options['outputDirectory'] = str(output)
    (ROOT/'build/auto-py-to-exe.config.json').write_text(
        json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    command = ['pyinstaller',values['filenames'],'--onefile' if values['onefile'] else '--onedir',
               '--console' if values['console'] else '--windowed','--name',values['name'],
               '--icon',values['icon_file']]
    for name in ('noconfirm','noupx','clean'):
        if values.get(name): command.append('--'+name)
    command.extend(extra)
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    if not packaging.package(shlex.join(command),options) or not (target/(values['name']+'.exe')).is_file():
        raise SystemExit('EXE 빌드에 실패했습니다. 빌드 로그를 확인하세요.')
    print('EXE:',target/(values['name']+'.exe'))
    if args.runtime:
        analysis = ast.literal_eval((work/'build'/values['name']/'Analysis-00.toc').read_text(encoding='utf-8'))
        hooks = []
        for group in analysis:
            if not isinstance(group, list):
                continue
            for record in group:
                if (isinstance(record, tuple) and len(record) == 3
                        and record[2] == 'PYSOURCE' and record[0].startswith('pyi_rth_')):
                    hooks.append(record)
        hook_dir = target/'_internal'/'runtime-hooks'
        hook_dir.mkdir(exist_ok=True)
        for index, (name, path, _) in enumerate(hooks):
            # The integrated entrypoint already initializes Tk and multiprocessing.
            if name not in ('pyi_rth__tkinter', 'pyi_rth_multiprocessing'):
                shutil.copyfile(path, hook_dir/f'{index:02}-{name}.py')


if __name__ == '__main__':
    main()
