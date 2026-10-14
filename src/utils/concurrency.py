import asyncio
from typing import Awaitable, Callable, Iterable, List, TypeVar


T = TypeVar("T")


async def bounded_gather(coros: Iterable[Awaitable[T]], max_concurrent: int) -> List[T]:
    sem = asyncio.Semaphore(max_concurrent)

    async def _runner(coro: Awaitable[T]) -> T:
        async with sem:
            return await coro

    return await asyncio.gather(*[_runner(c) for c in coros], return_exceptions=False)


