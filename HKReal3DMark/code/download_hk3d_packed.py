#!/usr/bin/env python3
"""Space-efficient resumable downloader for the full Hong Kong Cesium tree.

JSON hierarchy documents are retained compactly in SQLite and B3DM payloads are
written to ZIP64 shards. This avoids the multi-megabyte allocation cost of every
small file on exFAT volumes while preserving original relative paths.
"""
from __future__ import annotations
import argparse, json, os, queue, sqlite3, threading, time, urllib.parse, urllib.request, zipfile
from pathlib import Path

ROOT="https://data.map.gov.hk/api/3d-data/3dtiles/f2/tileset.json"

def env(path):
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            k,v=line.split('=',1);os.environ.setdefault(k.strip(),v.strip())

class Packed:
    def __init__(self,key,out,workers,shard_size):
        self.key=key;self.out=out;self.workers=workers;self.shard_size=shard_size
        out.mkdir(parents=True,exist_ok=True);self.db=sqlite3.connect(out/'inventory.sqlite',check_same_thread=False)
        self.db.execute('pragma journal_mode=WAL');self.db.execute('create table if not exists urls(url text primary key,status text,type text,size integer,error text)')
        self.db.commit();self.lock=threading.Lock();self.q=queue.Queue();self.payload=queue.Queue(maxsize=workers*4);self.stop=False
    def auth(self,u):
        s=urllib.parse.urlsplit(u);q=urllib.parse.parse_qs(s.query);q['key']=[self.key];return urllib.parse.urlunsplit((s.scheme,s.netloc,s.path,urllib.parse.urlencode(q,doseq=True),''))
    def add(self,u):
        u=urllib.parse.urlunsplit((*urllib.parse.urlsplit(u)[:3],'',''))
        with self.lock:
            cur=self.db.execute('insert or ignore into urls values(?,?,?,?,?)',(u,'queued','json' if u.lower().endswith('.json') else 'payload',None,None));self.db.commit()
        if cur.rowcount:self.q.put(u)
    def fetch(self):
        while True:
            u=self.q.get()
            if u is None:self.q.task_done();return
            try:
                req=urllib.request.Request(self.auth(u),headers={'User-Agent':'HKReal3DMark-academic-packed/1.0'})
                with urllib.request.urlopen(req,timeout=90) as r:data=r.read()
                if u.lower().endswith('.json'):
                    doc=json.loads(data);stack=[doc.get('root',doc)]
                    while stack:
                        n=stack.pop();stack.extend(n.get('children',[]));c=n.get('content') or {};v=c.get('uri') or c.get('url')
                        if v:self.add(urllib.parse.urljoin(u,v))
                    with self.lock:self.db.execute('update urls set status=?,size=? where url=?',('done',len(data),u));self.db.commit()
                else:self.payload.put((u,data))
            except Exception as e:
                with self.lock:self.db.execute('update urls set status=?,error=? where url=?',('queued',str(e)[:500],u));self.db.commit()
            self.q.task_done()
    def write(self):
        existing=[int(p.stem.rsplit('_',1)[-1]) for p in self.out.glob('payload_*.zip')]
        shard=max(existing,default=0);zf=None;count=0
        while True:
            item=self.payload.get()
            if item is None:break
            u,data=item
            if zf is None or count>=self.shard_size:
                if zf:zf.close()
                shard+=1;zf=zipfile.ZipFile(self.out/f'payload_{shard:04d}.zip','w',zipfile.ZIP_STORED,allowZip64=True);count=0
            name=urllib.parse.unquote(urllib.parse.urlsplit(u).path).split('/f2/',1)[-1]
            zf.writestr(name,data);count+=1
            with self.lock:self.db.execute('update urls set status=?,size=? where url=?',('done',len(data),u));self.db.commit()
            self.payload.task_done()
        if zf:zf.close()
    def run(self):
        # Recover queued work before adding the root.
        old=[r[0] for r in self.db.execute("select url from urls where status='queued'")];threads=[threading.Thread(target=self.fetch,daemon=True) for _ in range(self.workers)]
        writer=threading.Thread(target=self.write,daemon=True);writer.start()
        for t in threads:t.start()
        if old:
            for u in old:self.q.put(u)
        else:self.add(ROOT)
        self.q.join()
        for _ in threads:self.q.put(None)
        for t in threads:t.join()
        self.payload.join();self.payload.put(None);writer.join()
        total,done,bytes_=self.db.execute("select count(*),sum(status='done'),coalesce(sum(size),0) from urls").fetchone()
        print(json.dumps({'discovered':total,'done':done,'bytes':bytes_,'complete':total==done}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--env',default=str(Path(__file__).resolve().parents[1]/'.env'));p.add_argument('--workers',type=int,default=8);p.add_argument('--shard-size',type=int,default=10000);a=p.parse_args();env(Path(a.env));root=Path(os.environ['HK3D_DATA_ROOT'])/'packed';Packed(os.environ['HK3D_API_KEY'],root,a.workers,a.shard_size).run()
if __name__=='__main__':main()
