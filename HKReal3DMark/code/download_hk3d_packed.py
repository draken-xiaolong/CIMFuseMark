#!/usr/bin/env python3
"""Space-efficient resumable downloader for the full Hong Kong Cesium tree.

JSON hierarchy documents are retained compactly in SQLite and B3DM payloads are
written to ZIP64 shards. This avoids the multi-megabyte allocation cost of every
small file on exFAT volumes while preserving original relative paths.
"""
from __future__ import annotations
import argparse, json, os, queue, sqlite3, threading, time, urllib.error, urllib.parse, urllib.request, zipfile, zlib
from pathlib import Path
from packed_paths import inventory_path, write_pointer

ROOT="https://data.map.gov.hk/api/3d-data/3dtiles/f2/tileset.json"

def env(path):
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            k,v=line.split('=',1);os.environ.setdefault(k.strip(),v.strip())

class Packed:
    def __init__(self,key,out,workers,shard_size,db_path=None):
        self.key=key;self.out=out;self.workers=workers;self.shard_size=shard_size
        out.mkdir(parents=True,exist_ok=True);db_path=inventory_path(out,db_path);db_path.parent.mkdir(parents=True,exist_ok=True);write_pointer(out,db_path);self.db=sqlite3.connect(db_path,check_same_thread=False)
        self.db.execute('pragma journal_mode=WAL');self.db.execute('create table if not exists urls(url text primary key,status text,type text,size integer,error text,attempts integer default 0,content blob,archive text,member text)')
        columns={row[1] for row in self.db.execute('pragma table_info(urls)')}
        if 'attempts' not in columns:self.db.execute('alter table urls add column attempts integer default 0')
        if 'content' not in columns:self.db.execute('alter table urls add column content blob')
        if 'archive' not in columns:self.db.execute('alter table urls add column archive text')
        if 'member' not in columns:self.db.execute('alter table urls add column member text')
        # Earlier versions discarded JSON bodies. Re-fetch them once so the
        # hierarchy, transforms and spatial metadata remain reconstructable.
        self.db.execute("update urls set status='queued',size=null,error='requeued to retain JSON content' where type='json' and status='done' and content is null")
        self.db.commit();self.lock=threading.Lock();self.q=queue.PriorityQueue();self.payload=queue.Queue(maxsize=workers*4);self.stop=False;self.sequence=0;self.pending_db=0
    def checkpoint(self,force=False):
        """Commit a bounded batch; caller must hold ``self.lock``."""
        if force or self.pending_db>=100:
            self.db.commit();self.pending_db=0
    def auth(self,u):
        s=urllib.parse.urlsplit(u);q=urllib.parse.parse_qs(s.query);q['key']=[self.key];return urllib.parse.urlunsplit((s.scheme,s.netloc,s.path,urllib.parse.urlencode(q,doseq=True),''))
    def add(self,u):
        u=urllib.parse.urlunsplit((*urllib.parse.urlsplit(u)[:3],'',''))
        kind='json' if u.lower().endswith('.json') else 'payload'
        with self.lock:
            cur=self.db.execute('insert or ignore into urls(url,status,type,size,error,attempts) values(?,?,?,?,?,?)',(u,'queued',kind,None,None,0))
            if cur.rowcount:
                self.sequence+=1;item=(0 if kind=='json' else 1,self.sequence,u)
            else:item=None
        if item:self.q.put(item)
    def fetch(self):
        while True:
            _,_,u=self.q.get()
            if u is None:self.q.task_done();return
            try:
                req=urllib.request.Request(self.auth(u),headers={'User-Agent':'HKReal3DMark-academic-packed/1.0'})
                with urllib.request.urlopen(req,timeout=90) as r:data=r.read()
                if u.lower().endswith('.json'):
                    doc=json.loads(data);stack=[doc.get('root',doc)]
                    while stack:
                        n=stack.pop();stack.extend(n.get('children',[]));c=n.get('content') or {};v=c.get('uri') or c.get('url')
                        if v:self.add(urllib.parse.urljoin(u,v))
                    packed_json=zlib.compress(data,6)
                    with self.lock:
                        self.db.execute('update urls set status=?,size=?,content=?,error=null where url=?',('done',len(data),packed_json,u));self.pending_db+=1;self.checkpoint()
                else:self.payload.put((u,data))
            except Exception as e:
                permanent=isinstance(e,urllib.error.HTTPError) and e.code in {404,410}
                with self.lock:
                    self.db.execute('update urls set status=?,error=?,attempts=attempts+1 where url=?',('missing' if permanent else 'queued',str(e)[:500],u));self.pending_db+=1;self.checkpoint()
            self.q.task_done()
    def write(self):
        existing=[int(p.stem.rsplit('_',1)[-1]) for p in self.out.glob('payload_*.zip')]
        shard=max(existing,default=0);zf=None;count=0;archive_name=None
        while True:
            item=self.payload.get()
            if item is None:break
            u,data=item
            if zf is None or count>=self.shard_size:
                if zf:
                    with self.lock:self.checkpoint(force=True)
                    zf.close()
                shard+=1;archive_name=f'payload_{shard:04d}.zip';zf=zipfile.ZipFile(self.out/archive_name,'w',zipfile.ZIP_STORED,allowZip64=True);count=0
            name=urllib.parse.unquote(urllib.parse.urlsplit(u).path).split('/f2/',1)[-1]
            zf.writestr(name,data);count+=1
            with self.lock:
                self.db.execute('update urls set status=?,size=?,archive=?,member=?,error=null where url=?',('done',len(data),archive_name,name,u));self.pending_db+=1;self.checkpoint()
            self.payload.task_done()
        if zf:
            with self.lock:self.checkpoint(force=True)
            zf.close()
    def run(self):
        # Recover queued work before adding the root.
        old=self.db.execute("select url,type from urls where status='queued' order by type='json' desc,url").fetchall();threads=[threading.Thread(target=self.fetch,daemon=True) for _ in range(self.workers)]
        writer=threading.Thread(target=self.write,daemon=True);writer.start()
        if old:
            for u,kind in old:
                self.sequence+=1;self.q.put((0 if kind=='json' else 1,self.sequence,u))
            del old
        else:self.add(ROOT)
        for t in threads:t.start()
        self.q.join()
        for _ in threads:
            self.sequence+=1;self.q.put((2,self.sequence,None))
        for t in threads:t.join()
        self.payload.join();self.payload.put(None);writer.join()
        with self.lock:self.checkpoint(force=True)
        total,done,missing,bytes_=self.db.execute("select count(*),sum(status='done'),sum(status='missing'),coalesce(sum(size),0) from urls").fetchone()
        print(json.dumps({'discovered':total,'done':done,'missing':missing,'bytes':bytes_,'complete':total==done+missing}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--env',default=str(Path(__file__).resolve().parents[1]/'.env'));p.add_argument('--workers',type=int,default=8);p.add_argument('--shard-size',type=int,default=10000);a=p.parse_args();env(Path(a.env));root=Path(os.environ['HK3D_DATA_ROOT'])/'packed';Packed(os.environ['HK3D_API_KEY'],root,a.workers,a.shard_size,os.environ.get('HK3D_DB_PATH')).run()
if __name__=='__main__':main()
