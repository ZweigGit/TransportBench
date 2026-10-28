import urllib.request, struct, zlib, re, json, time, os, sys
OUT = 'E:/PROJECT/TransportBench/nml_table.json'

def fetch(url, start, end, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'Mozilla/5.0'})
            return urllib.request.urlopen(req, timeout=90).read()
        except Exception as e:
            time.sleep(2 + 2 * t)
    raise RuntimeError(f'fetch failed {url} {start}-{end}')

def get_nml(case):
    url = f'https://huggingface.co/datasets/CFDML/TransportBench/resolve/main/Doublecone/case{case}.zip'
    r = urllib.request.urlopen(urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'}), timeout=60)
    total = int(r.headers['Content-Length'])
    tail = fetch(url, total - 66000, total - 1)
    i = tail.rfind(b'PK\x05\x06')
    cd_size, cd_off = struct.unpack('<I', tail[i+12:i+16])[0], struct.unpack('<I', tail[i+16:i+20])[0]
    cd = fetch(url, cd_off, cd_off + cd_size - 1)
    j = 0
    while j < len(cd):
        if cd[j:j+4] != b'PK\x01\x02': break
        method, = struct.unpack('<H', cd[j+10:j+12])
        csize, = struct.unpack('<I', cd[j+20:j+24])
        nlen, elen, clen = struct.unpack('<HHH', cd[j+28:j+34])
        lho, = struct.unpack('<I', cd[j+42:j+46])
        fn = cd[j+46:j+46+nlen].decode('utf-8', 'replace')
        if fn == 'artist-sts.nml':
            lh = fetch(url, lho, lho + 29)
            nl, el = struct.unpack('<HH', lh[26:30])
            data = fetch(url, lho + 30 + nl + el, lho + 30 + nl + el + csize - 1)
            return zlib.decompress(data, -15).decode('utf-8', 'replace') if method == 8 else data.decode()
        j += 46 + nlen + elen + clen
    return None

def parse(nml):
    d = {}
    for k in ['vel', 'tem', 'temv', 'rho', 'n2', 'n', 'twall']:
        m = re.search(rf'{k}\s*=\s*([-\d.Ee+]+)', nml)
        if m: d[k] = float(m.group(1))
    return d

table = json.load(open(OUT)) if os.path.exists(OUT) else {}
cases = [str(i) for i in range(1, 51)] + ['10-1']
for c in cases:
    key = f'case{c}'
    if key in table:
        print(f'{key}: cached', flush=True); continue
    try:
        table[key] = parse(get_nml(c))
        print(f'{key}: u={table[key].get("vel")} T={table[key].get("tem")} Tv={table[key].get("temv")} '
              f'rho={table[key].get("rho")} tw={table[key].get("twall")}', flush=True)
    except Exception as e:
        print(f'{key}: FAILED {type(e).__name__} {e}', flush=True)
    json.dump(table, open(OUT, 'w'), indent=1)
print(f'HARVEST_DONE {len(table)}/51', flush=True)
