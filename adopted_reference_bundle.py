"""Selected producer I/O. Independent SPL4 decoder; no Viewer schema claim."""
import ctypes
import errno
import importlib.util
import os
from pathlib import Path
import struct

from formal_checkpoint_identity import (canonical, file_identity, require, keys,
    open_directory, checked_file, stream_identity, strict_json)
from formal_checkpoint import _write
from formal_entry import OutputClaim

SCHEMA='adopted-b-reference-bundle-v1'
RAW=('render','alpha','depth','radii','visibility_filter','gt')
PNG=('render','alpha','depth','gt')
ENCODING=dict(raw='float32 CHW; radii int32; visibility bool; npy without pickle',
    png='clamp01 CHW-to-HWC multiply255 uint8-truncate RGB top-down no-gamma',
    depth='raw accumulated depth; PNG per-view minmax; constant black')
LIMITATIONS=dict(viewer_compatibility_verified=False,gate_c_passed=False,
    directRasterizerEvidence='not-generated diagnostic-only not-formal-acceptance')


def write(path,writer):
    path=Path(path);fd=open_directory(str(path.parent))
    try:_write(fd,path.name,writer)
    finally:os.close(fd)


def write_json(path,value):
    data=canonical(value)
    write(path,lambda f:f.write(data))
    require(read_json(path)==value,'bundle_json_readback')


def read_json(path):
    from staged_comparison_checkpoint import _read_record
    return _read_record(path)


class Claim(OutputClaim):
    """Reuse identity checking/close; source config's output is never rewritten."""
    def __init__(self,path):
        self.path=str(path);self.fd=None
        p=Path(path);parent=open_directory(str(p.parent))
        try:
            os.mkdir(p.name,mode=0o700,dir_fd=parent)
            self.fd=os.open(p.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            st=os.fstat(self.fd);self.identity=(st.st_dev,st.st_ino);self.check()
        except BaseException:
            self.close();raise
        finally:os.close(parent)


def rename_new(source,destination):
    """Linux same-filesystem non-replacing publication. No unsafe fallback."""
    source,destination=Path(source),Path(destination)
    require(source.parent==destination.parent,'bundle_same_filesystem')
    fd=open_directory(str(source.parent))
    try:
        libc=ctypes.CDLL(None,use_errno=True)
        rename=getattr(libc,'renameat2',None)
        require(rename is not None,'rename_noreplace_unavailable')
        rename.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint)
        rename.restype=ctypes.c_int
        if rename(fd,os.fsencode(source.name),fd,os.fsencode(destination.name),1):
            err=ctypes.get_errno();raise OSError(err,os.strerror(err))
        os.fsync(fd)
    finally:os.close(fd)


def serializer(path,expected):
    require(file_identity(path)==expected,'serializer_identity')
    spec=importlib.util.spec_from_file_location('_adopted_spl4_serializer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(file_identity(path)==expected,'serializer_changed')
    return module


def export_spl4(path,capture,module):
    n=len(capture[1])
    fields=dict(xyz=capture[1],rotation=capture[5],rotation_r=capture[15],scale_xyz=capture[4],
        f_dc=capture[2],f_rest=capture[3],opacity=capture[6],t=capture[13],scale_t=capture[14])
    def emit(stream):
        module.write_header_v2(stream,n,0,0,True,True,3,4,4,3,3,45,1,1,1)
        for chunk in module.iter_v2_serialized_range(fields,0,n,True,chunk_records=8192):stream.write(chunk)
    write(path,emit)
    return check_spl4(path,capture)


def check_spl4(path,capture):
    """Independent header/offset/row oracle; no serializer constants/functions."""
    import numpy as np
    n=len(capture[1]);path=Path(path)
    fd=open_directory(str(path.parent))
    try:
        stream,before=checked_file(fd,path.name,128+260*n)
        with stream:
            identity=stream_identity(stream,before)
            header=stream.read(128)
            require(len(header)==128 and struct.unpack('<4sIIiiBBBB10I',header[:64])==
                (b'SPL4',2,n,0,0,1,1,1,1,3,4,4,3,3,45,1,1,1,0) and not any(header[64:]),'spl4_header')
            for start in range(0,n,8192):
                count=min(8192,n-start);raw=stream.read(260*count)
                require(len(raw)==260*count,'spl4_short')
                rows=np.frombuffer(raw,dtype='<f4').reshape(count,65)
                require(np.isfinite(rows).all(),'spl4_nonfinite')
                for slot,offset,width in ((1,0,3),(5,3,4),(15,7,4),(4,11,3),(2,14,3),
                                          (3,17,45),(6,62,1),(13,63,1),(14,64,1)):
                    tensor=capture[slot][start:start+count]
                    require(str(tensor.dtype)=='torch.float32' and tensor.device.type=='cpu','spl4_source_type')
                    expected=tensor.detach().numpy().reshape(count,width)
                    require(rows[:,offset:offset+width].tobytes()==expected.astype('<f4',copy=False).tobytes(),
                            'spl4_source_correspondence')
                for scale in (rows[:,11:14],rows[:,64:65]):
                    with np.errstate(over='ignore',under='ignore'):
                        activated=np.exp(scale)
                    require(np.isfinite(activated).all() and (activated>0).all(),'spl4_scale_activation')
                for quat in (rows[:,3:7],rows[:,7:11]):
                    norm=np.linalg.norm(quat.astype(np.float64),axis=1)
                    require(np.isfinite(norm).all() and (norm>0).all(),'spl4_quaternion')
            require(not stream.read(1) and stream_identity(stream,before)==identity,'spl4_length_or_changed')
            ranges=[];count=min(n,8192)
            for label,start in (('head',0),('middle',(n-count)//2),('tail',n-count)):
                stream.seek(128+start*260)
                from formal_checkpoint_identity import byte_identity
                ranges.append(dict(name=label,start=start,count=count,source_index='start + localIndex',
                                   identity=byte_identity(stream.read(count*260))))
    finally:os.close(fd)
    return dict(schema='adopted-population-v1',identity=identity,rows=n,header_bytes=128,stride=260,
        physical_sh=16,active_sh=[0,0],maximum_sh_labels=[3,2],quaternion='raw wxyz',
        scales='raw log',opacity='raw logit',sh='raw coefficient-major RGB',
        mapping='sourceIndex = assetRecordIndex = CUDA localIndex',ranges=ranges)


def member_names(count):
    return {'population.splat4d','population.json','manifest.json','verification.json',
        *(f'references/{i:02d}/{n}.npy' for i in range(count) for n in RAW),
        *(f'references/{i:02d}/{n}.png' for i in range(count) for n in PNG),
        *(f'references/{i:02d}/runtime.json' for i in range(count))}


def members(root):
    result={}
    for path in Path(root).rglob('*'):
        require(not path.is_symlink(),'bundle_symlink')
        if path.is_file():result[path.relative_to(root).as_posix()]=file_identity(path)
        else:require(path.is_dir(),'bundle_special_file')
    return result


def check_manifest(value,expected):
    keys(value,'schema version input provenance population references encoding limitations verification')
    require(value['schema']==SCHEMA and value['version']==1 and value==expected and
            value['encoding']==ENCODING and value['limitations']==LIMITATIONS,'bundle_manifest')


def index_bundle(root,expected_manifest):
    root=Path(root);check_manifest(read_json(root/'manifest.json'),expected_manifest)
    found=members(root)
    require(set(found)==member_names(len(expected_manifest['references'])),'bundle_members')
    write_json(root/'index.json',dict(schema='adopted-bundle-index-v1',members=found))
    check_index(root,expected_manifest)


def check_index(root,expected_manifest):
    root=Path(root);index=read_json(root/'index.json');keys(index,'schema members')
    require(index['schema']=='adopted-bundle-index-v1','bundle_index_schema')
    actual=members(root);actual.pop('index.json')
    require(set(actual)==member_names(len(expected_manifest['references'])) and actual==index['members'],
            'bundle_index_binding')
    check_manifest(read_json(root/'manifest.json'),expected_manifest)
    return file_identity(root/'index.json')


def publish(root,expected_manifest,verification):
    root=Path(root);pending=root/'.pending-bundle'
    require(verification==read_json(pending/'verification.json') and
            verification['status']=='verified_requires_review','bundle_independent_verification')
    index=check_index(pending,expected_manifest)
    rename_new(pending,root/'bundle')
    require(check_index(root/'bundle',expected_manifest)==index,'published_readback')
    receipt=dict(schema='adopted-reference-receipt-v1',status='verified_requires_review',
        index=index,verification=file_identity(root/'bundle/verification.json'),gate_c_passed=False)
    write_json(root/'.pending-receipt.json',receipt)
    rename_new(root/'.pending-receipt.json',root/'receipt.json')
    require(read_json(root/'receipt.json')==receipt,'receipt_readback')
    return receipt


def failure(root,stage,error):
    # A failed/uncertain publication is never deleted, retried or called done.
    write_json(Path(root)/'failure.json',dict(status='incomplete',stage=stage,
        error_type=type(error).__name__,error=str(error)[:1024],retry=False))
