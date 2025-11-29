import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import diskcache

log = logging.getLogger(__name__)

TWO_WEEKS_IN_SECONDS = 14 * 24 * 60 * 60


def _sanitize_topic(topic: str) -> str:
    return "".join(x for x in topic if x.isalnum())


@dataclass
class ItemWrapper:
    data: object
    timestamp: datetime

    @classmethod
    def from_object(cls, obj: object) -> "ItemWrapper":
        return cls(data=obj, timestamp=datetime.now())


class Cache:
    def __init__(self, cache_dir: Path):
        # topic -> diskcache.Cache
        self._parent_cache_dir = cache_dir
        self._caches: dict[str, diskcache.Cache] = {}
        self._parent_cache_dir.mkdir(parents=True, exist_ok=True)
        for d in self._parent_cache_dir.iterdir():
            if d.is_dir():
                self._caches[d.name] = diskcache.Cache(d)
                log.debug(f"Initialized cache for topic '{d.name}' at {d}")

    def add_to_topic(self, topic: str, data: object, safe: bool = True) -> bool:
        """
        Adds data to the cache under the given topic. If safe is True, it checks for duplicates before adding.
        Returns True if the item was added, False if it was skipped due to duplication.
        """
        sanitied_topic = _sanitize_topic(topic)

        if sanitied_topic not in self._caches:
            topic_cache_dir = self._parent_cache_dir / sanitied_topic
            topic_cache_dir.mkdir(parents=True, exist_ok=True)
            self._caches[sanitied_topic] = diskcache.Cache(topic_cache_dir)
            log.debug(f"Created new cache for topic '{sanitied_topic}' at {topic_cache_dir}")

        if safe:
            existing_items = self.dump_topic(topic)
            for item in existing_items:
                if item.data == data:
                    log.debug(f"Item already exists in topic '{sanitied_topic}' cache. Skipping add.")
                    return False

        # i don't think we need the key?
        key = str(uuid4())
        log.debug(f"Adding item to topic '{sanitied_topic}' cache with key '{key}': {data}")
        self._caches[sanitied_topic].add(key=key, value=ItemWrapper.from_object(data), expire=TWO_WEEKS_IN_SECONDS)
        return True

    def dump_topic(self, topic: str) -> list[ItemWrapper]:
        """
        Dumps all items in the given topic cache.
        Returns a list of ItemWrapper.
        """
        sanitied_topic = _sanitize_topic(topic)

        if sanitied_topic not in self._caches:
            log.debug(f"No cache found for topic '{sanitied_topic}'. Returning empty list.")
            return []

        topic_cache = self._caches[sanitied_topic]
        dumped_items = [topic_cache[key] for key in topic_cache.iterkeys()]
        dumped_items.sort(key=lambda item: item.timestamp, reverse=True)
        log.debug(f"Dumped {len(dumped_items)} items from topic '{sanitied_topic}' cache.")

        return dumped_items
