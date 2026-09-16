"""Release checks and verified launcher replacement; user data stays in place."""
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import quote
import urllib.request

from app_version import VERSION
from app_settings import read_settings, update_settings

INTERVAL = 7 * 24 * 60 * 60
REPOSITORY = 'https://github.com/w0bderful/Manga-Live'
LATEST_API = 'https://api.github.com/repos/w0bderful/Manga-Live/releases/latest'


def version_key(value):
    if not isinstance(value,str) or not re.fullmatch(r'v?\d+(?:\.\d+){1,3}',value):
        raise ValueError('릴리스 버전 형식이 올바르지 않습니다.')
    parts = tuple(int(p) for p in value.removeprefix('v').split('.'))
    return parts + (0,) * (4-len(parts))


def load_state(home):
    value = read_settings(Path(home)/'settings.json').get('updates',{})
    return value if isinstance(value,dict) else {}


def save_state(home, changes):
    # Runs on the UI thread alongside the application's other preference writes.
    state = load_state(home)
    state.update(changes)
    update_settings({'updates':state},Path(home)/'settings.json')


def check_due(state, now=None):
    now = time.time() if now is None else now
    value = state.get('last_checked')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        return True
    return value <= 0 or value > now or now-value >= INTERVAL


def parse_release(release, current=VERSION):
    if not isinstance(release,dict):
        raise ValueError('릴리스 응답이 올바르지 않습니다.')
    tag = release.get('tag_name','')
    if not isinstance(tag,str):
        raise ValueError('릴리스 버전 형식이 올바르지 않습니다.')
    if release.get('draft') or release.get('prerelease') or tag.startswith('runtime-'):
        return None
    if version_key(tag) <= version_key(current):
        return None
    expected_prefix = REPOSITORY+'/releases/download/'+quote(tag,safe='')+'/'
    for asset in release.get('assets',[]):
        if not isinstance(asset,dict) or asset.get('name') not in ('Manga.Live.exe','Manga Live.exe'):
            continue
        digest = asset.get('digest','')
        size = asset.get('size')
        url = asset.get('browser_download_url','')
        expected_url = expected_prefix+quote(asset['name'],safe='')
        if (asset.get('state') != 'uploaded' or not isinstance(digest,str)
                or not re.fullmatch(r'sha256:[0-9a-f]{64}',digest)
                or isinstance(size,bool) or not isinstance(size,int) or not 0 < size <= 128*1024**2
                or url != expected_url):
            raise ValueError('업데이트 파일의 주소·크기·검증 정보를 확인할 수 없습니다.')
        return {'version':tag.removeprefix('v'),'tag':tag,'name':'Manga Live.exe',
                'url':url,'size':size,'sha256':digest[7:], 'release_url':REPOSITORY+'/releases/tag/'+quote(tag,safe='')}
    raise ValueError('새 릴리스에 Manga Live.exe 파일이 없습니다.')


def fetch_latest(current=VERSION, opener=None):
    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(LATEST_API,headers={
        'Accept':'application/vnd.github+json','User-Agent':'MangaLive-Update/1.0',
        'X-GitHub-Api-Version':'2022-11-28'})
    with opener(request,timeout=10) as response:
        data = response.read(2*1024**2+1)
    if len(data) > 2*1024**2:
        raise ValueError('릴리스 응답이 너무 큽니다.')
    return parse_release(json.loads(data),current)


def launcher_path(home):
    value = os.environ.get('MANGA_LIVE_LAUNCHER')
    if not value:
        return None
    target = Path(value)
    if not target.is_absolute() or target.suffix.lower() != '.exe' or target.resolve().parent != Path(home).resolve():
        return None
    if target.is_symlink():
        return None
    return target


def apply_update(info, home, target, cancel, progress):
    from bootstrap import download, checksum, check_cancel
    home = Path(home).resolve()
    target = Path(target)
    if target.resolve().parent != home or target.is_symlink() or target.suffix.lower() != '.exe':
        raise ValueError('업데이트할 실행 파일 경로가 올바르지 않습니다.')
    if not re.fullmatch('[0-9a-f]{64}',info['sha256']):
        raise ValueError('업데이트 검증 정보가 올바르지 않습니다.')
    cache = home/'.manga-live-runtime'/'updates'/info['sha256']
    if not cache.resolve().is_relative_to(home):
        raise ValueError('업데이트 저장 경로가 올바르지 않습니다.')
    cache.mkdir(parents=True,exist_ok=True)
    if target.is_file() and checksum(target,cancel)==info['sha256']:
        return
    candidate = download(info,cache,cancel,progress)
    check_cancel(cancel)
    from self_update import pending_path
    receipt = pending_path(home)
    temporary = receipt.with_suffix('.tmp')
    try:
        if checksum(candidate,cancel)!=info['sha256']:
            raise ValueError('실행 파일 검증에 실패했습니다.')
        check_cancel(cancel)
        temporary.write_text(json.dumps({'sha256':info['sha256'], 'target':target.name,
            'version':info['version']}, ensure_ascii=False), encoding='utf-8')
        temporary.replace(receipt)
    finally:
        temporary.unlink(missing_ok=True)
