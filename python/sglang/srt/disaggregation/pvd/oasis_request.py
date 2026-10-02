"""One live request's Oasis foreground and per-layer lookahead ownership.

Admission supplies validated sparse initial banks and initialized EAGLE state.
The model/sampler caller must acknowledge each actual token commit. This owner
has no complete-prefix target probe and cannot fall back to one.
"""

from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerLookahead


class OasisRequestDecoder:
    def __init__(self, request_id, incarnation, *, decoder, initial_banks,
                 predict_one, fetch_layer, current_token, position, max_steps,
                 workers=2, timeout=60):
        if (not callable(predict_one) or not callable(fetch_layer)
                or type(max_steps) is not int or max_steps <= 0
                or type(current_token) is not int or current_token < 0
                or type(position) is not int or position < 0
                or len(initial_banks) != decoder.layers):
            raise ValueError("validated initial sparse banks and bounded one-token draft required")
        self.decoder, self.predict_one, self.fetch_layer = decoder, predict_one, fetch_layer
        self.pipeline = LayerLookahead(request_id, incarnation, layers=decoder.layers,
            workers=workers, timeout=timeout)
        self.request_id, self.incarnation = request_id, incarnation
        self.banks = list(initial_banks)
        self.current_token, self.position = current_token, position
        self.max_steps, self.step = max_steps, 0
        self.state = "ready"
        self.features = None
        self.predicted = None
        self._published = set()

    def _bank(self, layer):
        if self.step:
            self.banks[layer] = self.pipeline.consume(self.step - 1, layer)
        return self.banks[layer]

    def _publish(self, layer, query, bank):
        self._published.add(layer)
        if self.step + 1 >= self.max_steps:
            return
        # Query preparation must snapshot/enqueue D2H on the owner CUDA stream
        # before the background thread starts. fetch_layer owns all copies and
        # returns a synchronous callback carrying an exact LayerReply ticket.
        callback = self.fetch_layer(query, bank)
        if not callable(callback):
            raise TypeError("layer transport must prepare a terminal callback")
        self.pipeline.publish(self.step, layer, callback)

    def forward(self, current_token, position):
        if (self.state != "ready" or self.step >= self.max_steps
                or current_token != self.current_token or position != self.position):
            raise RuntimeError("exact next actual token/position and unused forward required")
        self.state = "executing"
        self._published.clear()
        try:
            predicted = self.predict_one(self.current_token, self.features)
            if type(predicted) is not int or predicted < 0:
                raise ValueError("one lookahead token required")
            self.predicted = predicted
            logits, features = self.decoder.step(self.current_token, predicted,
                self.position, self._bank, publish=self._publish)
            if self._published != set(range(self.decoder.layers)):
                raise RuntimeError("target forward omitted a layer's future Q")
            self.features = features
            self.state = "awaiting_actual_commit"
            return logits
        except BaseException:
            self.state = "failed"
            raise

    def actual_committed(self, token):
        if self.state != "awaiting_actual_commit" or type(token) is not int or token < 0:
            raise RuntimeError("one authoritative sampler token commit required")
        self.current_token = token
        self.position += 1
        self.step += 1
        self.state = "ready"

    def close(self):
        if self.state == "closed":
            return ()
        if self.state == "executing":
            raise RuntimeError("cannot retire an executing target forward")
        owner = getattr(self.decoder, "owner", None)
        if owner is not None and (owner.active is not None or owner.quarantined):
            raise RuntimeError("retain request until target completion is proved")
        errors = self.pipeline.close()
        self.banks.clear()
        self.features = None
        for history in self.decoder.generated if hasattr(self.decoder, "generated") else ():
            history.clear()
        self.state = "closed"
        return errors
