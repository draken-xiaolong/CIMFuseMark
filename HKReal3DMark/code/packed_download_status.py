#!/usr/bin/env python3
"""Print a compact status snapshot for the packed full-territory download."""
import argparse, glob, json, os, sqlite3, zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/Volumes/SANDISK-ELE/HKReal3DMarkData/packed');a=p.parse_args();root=Path(a.root)
    db=sqlite3.connect(root/'inventory.sqlite',timeout=30)
    total,done,missing,json_done,json_stored,payload_done,json_queued,payload_queued,size=db.execute("select count(*),sum(status='done'),sum(status='missing'),sum(type='json' and status='done'),sum(type='json' and status='done' and content is not null),sum(type='payload' and status='done'),sum(type='json' and status='queued'),sum(type='payload' and status='queued'),coalesce(sum(size),0) from urls").fetchone()
    shards=sorted(glob.glob(str(root/'payload_*.zip')))
    # A closed ZIP has a readable central directory; an actively written shard does not.
    closed=len(shards) if not shards or zipfile.is_zipfile(shards[-1]) else len(shards)-1
    result={'total':total,'done':done or 0,'missing':missing or 0,'queued':total-(done or 0)-(missing or 0),'progress_percent':round(100*((done or 0)+(missing or 0))/max(total,1),3),'json_done':json_done or 0,'json_stored':json_stored or 0,'json_queued':json_queued or 0,'payload_done':payload_done or 0,'payload_queued':payload_queued or 0,'content_gib':round(size/2**30,3),'zip_shards':len(shards),'closed_shards':closed,'allocated_gib':round(sum(os.path.getsize(x) for x in shards)/2**30,3),'complete':total==(done or 0)+(missing or 0)}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
