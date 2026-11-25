"""Read selected members of public ZIPs using HTTP Range (no full download)."""
import io
import re
import requests

class RemoteZipReader(io.RawIOBase):
    def __init__(self,url,max_bytes=100*1024*1024):
        self.url=url; self.pos=0; self.downloaded=0; self.limit=max_bytes
        self.session=requests.Session()
        with self.session.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=60) as r:
            r.raise_for_status()
            if r.status_code!=206: raise IOError('Server does not honor HTTP Range; full download refused')
            self.size=int(r.headers['Content-Range'].rsplit('/',1)[1]); self.url=r.url
            r.content
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0: raise ValueError('Negative seek')
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n<=0: return b''
        if self.downloaded+n>self.limit: raise IOError('HTTP Range byte budget exceeded')
        with self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},stream=True,timeout=60) as r:
            r.raise_for_status()
            if r.status_code!=206: raise IOError('Range response required')
            data=r.content
            if len(data)!=n: raise IOError(f'Invalid range length {len(data)} != {n}')
        self.pos+=len(data); self.downloaded+=len(data)
        return data
