"""Bounded multiplexed binary control frames over one owned WebSocket.

KV bytes and WRITE completion proofs stay on the ordinary Mooncake/lifecycle
path. Disconnect never proves that a published destination has no writer.
"""
import asyncio
import json
import struct
import aiohttp

PATH = '/internal/v1/indexes/search-deliver-channel'
MAX_REQUEST_BYTES = 12+262144+4*100000
MAX_RESPONSE_BYTES = 2*1024*1024
_MAGIC = b'PVDFC01\x00'
_REQUEST = struct.Struct('<8sQ')
_RESPONSE = struct.Struct('<8sQI')


class BinaryChannelError(RuntimeError):
    pass


class BinaryChannelRefused(BinaryChannelError):
    def __init__(self, status, body):
        self.status, self.body = status, body
        super().__init__(str(body.get('error', 'request refused')))


def pack_request(sequence, body):
    if type(sequence) is not int or not 1 <= sequence < 2**64 or not isinstance(body,bytes) or not 1 <= len(body) <= MAX_REQUEST_BYTES:
        raise BinaryChannelError('invalid bounded request frame')
    return _REQUEST.pack(_MAGIC,sequence)+body


def unpack_request(frame):
    if not isinstance(frame,bytes) or not _REQUEST.size < len(frame) <= _REQUEST.size+MAX_REQUEST_BYTES:
        raise BinaryChannelError('invalid request frame extent')
    magic,sequence=_REQUEST.unpack_from(frame)
    if magic != _MAGIC or sequence==0: raise BinaryChannelError('invalid request frame identity')
    return sequence,frame[_REQUEST.size:]


def pack_response(sequence,status,body):
    if (type(sequence) is not int or not 1 <= sequence < 2**64 or type(status) is not int
            or not 100 <= status <= 599 or not isinstance(body,bytes) or not 1 <= len(body) <= MAX_RESPONSE_BYTES):
        raise BinaryChannelError('invalid bounded response frame')
    return _RESPONSE.pack(_MAGIC,sequence,status)+body


def unpack_response(frame,max_bytes=MAX_RESPONSE_BYTES):
    if not isinstance(frame,bytes) or not _RESPONSE.size < len(frame) <= _RESPONSE.size+max_bytes:
        raise BinaryChannelError('invalid response frame extent')
    magic,sequence,status=_RESPONSE.unpack_from(frame)
    if magic != _MAGIC or sequence==0 or not 100 <= status <= 599:
        raise BinaryChannelError('invalid response frame identity')
    body=json.loads(frame[_RESPONSE.size:])
    if not isinstance(body,dict): raise BinaryChannelError('response must be an object')
    return sequence,status,body


class FusedBinaryChannel:
    def __init__(self,base_url,*,timeout,max_response_bytes):
        self.base_url,self.timeout,self.max_response_bytes=base_url,timeout,min(max_response_bytes,MAX_RESPONSE_BYTES)
        self._connect_lock,self._send_lock=asyncio.Lock(),asyncio.Lock()
        self._session=self._socket=self._reader_task=None
        self._owned,self._pending=set(),{}
        self._sequence=0
        self._closed,self._failure=False,None
        self.connections,self.requests,self.pending_peak=0,0,0

    def _fail(self,exc):
        if self._failure is None:self._failure=exc
        for future in tuple(self._pending.values()):
            if not future.done():future.set_exception(self._failure)

    async def _connect(self):
        async with self._connect_lock:
            if self._failure:raise self._failure
            if self._socket is None:
                self._session=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
                self._socket=await self._session.ws_connect(self.base_url+PATH,
                    max_msg_size=self.max_response_bytes+_RESPONSE.size,compress=0)
                self.connections+=1
                self._reader_task=asyncio.create_task(self._read())
            if self._socket.closed:raise BinaryChannelError('channel disconnected; no automatic retry')

    async def _read(self):
        try:
            async for message in self._socket:
                if message.type != aiohttp.WSMsgType.BINARY:
                    raise BinaryChannelError('binary response frame required')
                sequence,status,body=unpack_response(message.data,self.max_response_bytes)
                future=self._pending.get(sequence)
                if future is None or future.done():raise BinaryChannelError('unsolicited or duplicate channel response')
                if status==200:future.set_result(body)
                else:future.set_exception(BinaryChannelRefused(status,body))
        except Exception as exc:
            self._fail(BinaryChannelError(f'binary channel receive failed: {exc}'))
        finally:
            if self._pending or not self._closed:self._fail(BinaryChannelError('binary channel disconnected; writer state is unknown'))

    async def _exchange(self,body):
        async def work():
            await self._connect()
            async with self._send_lock:
                if self._failure:raise self._failure
                self._sequence+=1
                sequence=self._sequence
                frame=pack_request(sequence,body)
                future=asyncio.get_running_loop().create_future()
                self._pending[sequence]=future
                self.pending_peak=max(self.pending_peak,len(self._pending))
                try:
                    await self._socket.send_bytes(frame)
                    self.requests+=1
                except BaseException:
                    self._pending.pop(sequence,None)
                    raise
            try:
                return await future
            finally:
                self._pending.pop(sequence,None)
        try:
            return await asyncio.wait_for(work(),self.timeout)
        except BinaryChannelRefused:
            raise
        except Exception as exc:
            failure=exc if isinstance(exc,BinaryChannelError) else BinaryChannelError(str(exc))
            self._fail(failure)
            if self._socket is not None:await self._socket.close()
            raise failure

    async def exchange(self,body):
        if self._closed or self._failure:raise BinaryChannelError('binary channel closed or failed')
        # Matches the two serving workers; reject before publication rather than
        # buffering an unbounded number of authorized write requests.
        if len(self._owned)>=2:raise BinaryChannelError('binary channel inflight capacity exhausted')
        task=asyncio.create_task(self._exchange(body))
        self._owned.add(task)
        def done(future):
            self._owned.discard(future)
            if not future.cancelled():future.exception()
        task.add_done_callback(done)
        return await asyncio.shield(task)

    async def close(self):
        self._closed=True
        if self._owned:
            await asyncio.gather(*(asyncio.shield(t) for t in tuple(self._owned)),return_exceptions=True)
        if self._socket is not None:await self._socket.close()
        if self._reader_task is not None:await self._reader_task
        if self._session is not None:await self._session.close()

    def snapshot(self):
        return dict(connections=self.connections,requests=self.requests,pending_peak=self.pending_peak,
                    owned=len(self._owned),failed=self._failure is not None,closed=self._closed)
