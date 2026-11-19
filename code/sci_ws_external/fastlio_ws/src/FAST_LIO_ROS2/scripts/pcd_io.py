#!/usr/bin/env python3
# FLIO-15 공용 PCD/PLY I/O (numpy only, open3d 불필요). ascii/binary PCD 지원.
import numpy as np

_PCD_TYPE = {('F',4):'f4',('F',8):'f8',('I',1):'i1',('I',2):'i2',('I',4):'i4',('I',8):'i8',
             ('U',1):'u1',('U',2):'u2',('U',4):'u4',('U',8):'u8'}

def read_pcd(path):
    """returns (dict field->np.array, header dict)"""
    with open(path,'rb') as f: raw=f.read()
    nl=raw.find(b'DATA')
    eol=raw.find(b'\n',nl)
    head_txt=raw[:eol].decode('ascii','replace')
    data_kind=raw[nl:eol].decode().split()[1]
    H={}
    for ln in head_txt.splitlines():
        p=ln.split()
        if not p or p[0].startswith('#'): continue
        H[p[0]]=p[1:]
    fields=H['FIELDS']; sizes=[int(x) for x in H['SIZE']]; types=H['TYPE']
    counts=[int(x) for x in H.get('COUNT',['1']*len(fields))]
    npts=int(H['POINTS'][0]) if 'POINTS' in H else int(H['WIDTH'][0])*int(H['HEIGHT'][0])
    body=raw[eol+1:]
    cols={}
    if data_kind=='ascii':
        arr=np.array([[float(x) for x in ln.split()] for ln in body.decode('ascii','replace').split('\n') if ln.strip()][:npts])
        ci=0
        for i,fn in enumerate(fields):
            cols[fn]=arr[:,ci].astype(_PCD_TYPE.get((types[i],sizes[i]),'f4')); ci+=counts[i]
    else:  # binary
        dt=[]
        for i,fn in enumerate(fields):
            dt.append((fn,_PCD_TYPE.get((types[i],sizes[i]),'f4')))
        stride=sum(sizes[i]*counts[i] for i in range(len(fields)))
        rec=np.frombuffer(body[:npts*stride],dtype=np.dtype(dt))
        for fn in fields: cols[fn]=rec[fn].astype(np.float64)
    return cols, H

def get_xyz(cols):
    return np.stack([cols['x'],cols['y'],cols['z']],axis=1).astype(np.float64)

def write_ply(path, xyz, extra=None):
    n=xyz.shape[0]
    with open(path,'w') as f:
        f.write("ply\nformat ascii 1.0\ncomment flio15 cleaned\n")
        f.write(f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for p in xyz: f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

def write_pcd_binary(path, xyz):
    n=xyz.shape[0]
    hdr=("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
         f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path,'wb') as f:
        f.write(hdr.encode('ascii'))
        f.write(xyz.astype(np.float32).tobytes())
