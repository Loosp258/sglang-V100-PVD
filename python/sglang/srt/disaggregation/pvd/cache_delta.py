"""Exact monotonic CPU cache versions bound to one request-owned channel.

Versions count installed unique rows. Only successful local installation adds
rows; search confirms a snapshot independently of delivery ACK. Missing/older
delta bases request a new fenced identity and a full snapshot, never a guess.
"""
import base64
import hashlib
import threading
import uuid
import numpy as np
from sglang.srt.disaggregation.pvd.cache_snapshot import cache_ids

PROTOCOL='pvd.cache-delta.v1'


class CacheDeltaResync(ValueError):
    pass


def delta_scratch_bytes(prompt_tokens):
    if type(prompt_tokens) is not int or not 1<=prompt_tokens<=32768:
        raise ValueError('bounded cache delta Prompt required')
    # Journals, bitmaps, cached canonical JSON and bounded per-head objects.
    # Remaining allowance covers four live frozen snapshot copies/encoding.
    return 112*(2*prompt_tokens+5*((prompt_tokens+7)//8)+1024)


def _mask(ids, prompt_tokens):
    bits=np.zeros((prompt_tokens+7)//8,dtype=np.uint8)
    ids=np.asarray(ids,dtype=np.int64)
    np.bitwise_or.at(bits,ids//8,(1<<(ids%8)).astype(np.uint8))
    return bits


def _digest(bits):
    return hashlib.sha256(bits.tobytes()).hexdigest()


def _snapshot(bits, ids):
    if len(ids)<=8:return sorted(map(int,ids))
    if len(ids)*2<=len(bits):
        encoding='u16le-base64-v1'
        raw=np.sort(np.asarray(ids,dtype='<u2')).tobytes()
    else:
        encoding='bitset-le-base64-v1';raw=bits.tobytes()
    return dict(encoding=encoding,data=base64.b64encode(raw).decode('ascii'))


def full_resync_delta(delta,cached):
    return {**delta,'states':[{**state,'base':None,'added':[],'full':snapshot}
        for state,snapshot in zip(delta['states'],cached,strict=True)]}


class CacheDeltaSender:
    def __init__(self,prompt_tokens):
        delta_scratch_bytes(prompt_tokens)
        self.prompt_tokens=prompt_tokens
        self.lock=threading.RLock()
        self.states={}

    def _state(self,layer,head):
        if type(layer) is not int or not 0<=layer<28 or type(head) is not int or not 0<=head<4:
            raise ValueError('bounded cache layer/head required')
        key=(layer,head)
        if key not in self.states:
            self.states[key]=dict(ids=np.empty(self.prompt_tokens,dtype='<u2'),
                bits=np.zeros((self.prompt_tokens+7)//8,dtype=np.uint8),
                version=0,confirmed=None,cached=None,digest=None)
        return self.states[key]

    def publish(self,layer,head,token):
        with self.lock:
            state=self._state(layer,head)
            if type(token) is not int or not 0<=token<self.prompt_tokens:
                raise ValueError('bounded installed cache token required')
            if state['bits'][token//8]&(1<<(token%8)):
                raise ValueError('cache delta row already published')
            state['ids'][state['version']]=token
            state['bits'][token//8]|=1<<(token%8)
            state['version']+=1;state['cached']=state['digest']=None

    def snapshot(self,layer,heads):
        if heads not in ([0,1],[2,3]):raise ValueError('rank head pair required')
        with self.lock:
            snapshots=[];messages=[]
            for head in heads:
                state=self._state(layer,head);version=state['version'];base=state['confirmed']
                if state['cached'] is None:
                    state['cached']=_snapshot(state['bits'],state['ids'][:version])
                    state['digest']=_digest(state['bits'])
                snapshots.append(state['cached'])
                added=state['ids'][base:version].astype('<u2').tobytes() if base is not None else b''
                messages.append(dict(base=base,version=version,digest=state['digest'],
                    added=dict(encoding='u16le-base64-v1',data=base64.b64encode(np.sort(np.frombuffer(added,dtype='<u2')).tobytes()).decode())
                        if added else [],full=state['cached'] if base is None else None))
            return snapshots,dict(protocol=PROTOCOL,layer=layer,heads=heads,prompt_tokens=self.prompt_tokens,states=messages)

    def confirm(self,delta,proof):
        expected=[dict(version=s['version'],digest=s['digest']) for s in delta['states']]
        if (proof!=expected or not isinstance(proof,list)
                or any(type(item.get('version')) is not int for item in proof)):
            raise ValueError('cache version confirmation mismatch')
        with self.lock:
            for head,item in zip(delta['heads'],proof,strict=True):
                state=self._state(delta['layer'],head)
                if item['version']>state['version']:raise ValueError('uninstalled cache version')
                state['confirmed']=max(state['confirmed'] or 0,item['version'])


class CacheDeltaReceiver:
    """At most 28x2 packed bitmaps, charged until all channel work joins."""
    def __init__(self,prompt_tokens,heads,budget):
        delta_scratch_bytes(prompt_tokens)
        if heads not in ([0,1],[2,3]):raise ValueError('rank head pair required')
        self.prompt_tokens,self.heads,self.budget=prompt_tokens,heads,budget
        self.owner='cache-delta:'+uuid.uuid4().hex
        # Persistent masks/metadata plus two rank jobs' temporary decoded sets.
        # Host objects are conservatively included, not just bitmap storage.
        bit_bytes=(prompt_tokens+7)//8
        self.bytes=56*(bit_bytes+1024)+4*(64*prompt_tokens+8*bit_bytes+8192)
        self.states={};self.closed=False
        budget.reserve(self.owner,self.bytes,0)

    def prepare(self,scope,delta):
        if (self.closed or not isinstance(delta,dict)
                or set(delta)!={'protocol','layer','heads','prompt_tokens','states'}
                or delta['protocol']!=PROTOCOL or delta['prompt_tokens']!=self.prompt_tokens
                or delta['heads']!=self.heads or delta['heads']!=scope['heads']
                or type(delta['layer']) is not int or not 0<=delta['layer']<28
                or delta['layer']!=scope['layer'] or delta['prompt_tokens']!=scope['prompt_tokens']
                or scope['cached']!=[[],[]] or not isinstance(delta['states'],list) or len(delta['states'])!=2):
            raise ValueError('cache delta channel scope mismatch')
        planned=[];snapshots=[];proof=[]
        for head,message in zip(self.heads,delta['states'],strict=True):
            if (not isinstance(message,dict) or set(message)!={'base','version','digest','added','full'}
                    or type(message['version']) is not int or not 0<=message['version']<=self.prompt_tokens
                    or (message['base'] is not None and (type(message['base']) is not int or not 0<=message['base']<=message['version']))
                    or not isinstance(message['digest'],str) or len(message['digest'])!=64):
                raise ValueError('invalid exact cache version')
            key=(delta['layer'],head);current=self.states.get(key)
            added=cache_ids(message['added'],self.prompt_tokens)
            if message['full'] is not None:
                if message['base'] is not None or len(added):raise ValueError('full cache resync must stand alone')
                ids=cache_ids(message['full'],self.prompt_tokens)
                bits=_mask(ids,self.prompt_tokens)
                if current is not None:
                    old=current[1]
                    if message['version']>=current[0]:
                        if np.any(old&~bits):raise ValueError('cache version removed installed rows')
                    elif np.any(bits&~old):raise ValueError('old full version is not an exact subset')
            else:
                if message['base'] is None:raise ValueError('cache delta needs a base')
                if current is None or not message['base']<=current[0]<=message['version']:
                    raise CacheDeltaResync('cache delta base unavailable')
                bits=current[1]|_mask(added,self.prompt_tokens)
                ids=np.flatnonzero(np.unpackbits(bits,bitorder='little')[:self.prompt_tokens])
            if len(ids)!=message['version'] or _digest(bits)!=message['digest']:
                raise ValueError('cache delta exact set/version digest mismatch')
            snapshots.append(_snapshot(bits,ids))
            proof.append(dict(version=message['version'],digest=message['digest']))
            if current is None or message['version']>=current[0]:planned.append((key,(message['version'],bits)))
        def commit():
            if self.closed:raise RuntimeError('cache delta owner closed')
            self.states.update(planned)
        return {**scope,'cached':snapshots},proof,commit

    def close(self):
        if not self.closed:
            self.states.clear();self.budget.release(self.owner);self.closed=True
