"""One object that gives the pipeline both models: Muse Spark Open (reasoning, vision) and SAM 3.1."""
import asyncio

from app.clients.llm_client import LLMClient
from app.clients.sam_local import LocalSAM


class Models:
    def __init__(self, llm: LLMClient | None = None, sam: LocalSAM | None = None):
        self.llm = llm or LLMClient()
        self.sam = sam or LocalSAM(cache=self.llm.cache)

    async def spark_json(self, prompt, schema, jpeg=None, tag="llm"):
        return await self.llm.json(prompt, schema, jpeg=jpeg, tag=tag)

    async def sam_segment(self, jpeg: bytes, width: int, height: int, concept: str) -> list:
        # SAM runs on the CPU; run it in a thread so the event loop stays free
        return await asyncio.to_thread(self.sam.segment, jpeg, concept)

    @property
    def cache_hits(self) -> int:
        return self.llm.cache_hits + self.sam.cache_hits

    async def aclose(self):
        await self.llm.aclose()
