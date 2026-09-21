"""Release checks and verified launcher replacement; user data stays in place."""
import json
import math
import os
from pathlib import Path
import re
import time
import socket
import ssl
from urllib.parse import quote
import urllib.error
import urllib.request

from app_version import VERSION
from app_settings import read_settings, update_settings

INTERVAL = 24 * 60 * 60
REPOSITORY = 'https://github.com/w0bderful/Manga-Live'
RELEASES_API = 'https://api.github.com/repos/w0bderful/Manga-Live/releases?per_page=100'


class NoReleaseError(ValueError):
    pass


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
    if release.get('draft') or tag.startswith('runtime-'):
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
    def fetch(url):
        request = urllib.request.Request(url,headers={
            'Accept':'application/vnd.github+json','User-Agent':'MangaLive-Update/1.0',
            'X-GitHub-Api-Version':'2022-11-28'})
        with opener(request,timeout=10) as response:
            data = response.read(2*1024**2+1)
        if len(data) > 2*1024**2:
            raise ValueError('릴리스 응답이 너무 큽니다.')
        return json.loads(data)
    # /latest excludes prereleases and returns 404 when every app release is
    # marked prerelease. The project publishes numeric app versions in both forms.
    releases = fetch(RELEASES_API)
    if not isinstance(releases, list):
        raise ValueError('릴리스 목록 응답이 올바르지 않습니다.')
    candidates = []
    for item in releases:
        if not isinstance(item, dict) or item.get('draft'):
            continue
        try:
            key = version_key(item.get('tag_name'))
        except ValueError:
            continue
        candidates.append((key, item))
    if not candidates:
        raise NoReleaseError('조회할 수 있는 프로그램 릴리스가 없습니다.')
    release = max(candidates, key=lambda candidate: candidate[0])[1]
    return parse_release(release,current)



def error_detail(error):
    """Describe network/verification failures without echoing URLs or secrets."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, urllib.error.HTTPError):
            if error.code in (403, 429):
                return f'GitHub 요청이 제한되거나 차단되었습니다(HTTP {error.code}). 잠시 후 다시 시도하세요.'
            if error.code == 404:
                return 'GitHub에서 릴리스 또는 다운로드 파일을 찾지 못했습니다(HTTP 404).'
            return f'GitHub 서버가 오류를 반환했습니다(HTTP {error.code}).'
        if isinstance(error, (TimeoutError, socket.timeout)):
            return 'GitHub 응답 시간이 초과되었습니다. 잠시 후 다시 시도하세요.'
        if isinstance(error, ssl.SSLError):
            return '보안 연결을 확인하지 못했습니다. PC 시간과 HTTPS 검사 설정을 확인하세요.'
        if isinstance(error, urllib.error.URLError):
            if isinstance(error.reason, BaseException):
                error = error.reason
                continue
            return 'GitHub에 연결하지 못했습니다. 인터넷 연결을 확인하세요.'
        if isinstance(error, PermissionError):
            return '업데이트 폴더에 쓸 수 없습니다. 폴더 권한과 파일 잠금을 확인하세요.'
        if isinstance(error, NoReleaseError):
            return '공개된 프로그램 릴리스가 없습니다. 릴리스 게시 상태를 확인하세요.'
        if isinstance(error, ValueError):
            return '릴리스 응답이나 다운로드 파일의 검증 정보가 올바르지 않습니다.'
        error = error.__cause__ or error.__context__
    return '인터넷 연결·GitHub 상태·저장 공간을 확인하고 다시 시도하세요.'


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
