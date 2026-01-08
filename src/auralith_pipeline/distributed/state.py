"""State management for distributed processing."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class StateStore(ABC):
    """Abstract base class for state storage."""

    @abstractmethod
    def connect(self):
        """Connect to state store."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from state store."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None):
        """Set a value in the store."""
        pass

    @abstractmethod
    def get(self, key: str) -> Any:
        """Get a value from the store."""
        pass

    @abstractmethod
    def delete(self, key: str):
        """Delete a key from the store."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        pass

    @abstractmethod
    def push_queue(self, queue_name: str, item: Any):
        """Push an item to a queue."""
        pass

    @abstractmethod
    def pop_queue(self, queue_name: str, timeout: int = 0) -> Any:
        """Pop an item from a queue."""
        pass

    @abstractmethod
    def queue_length(self, queue_name: str) -> int:
        """Get the length of a queue."""
        pass


class RedisStateStore(StateStore):
    """Redis-based state storage implementation."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.client = None

    def connect(self):
        """Connect to Redis."""
        try:
            import redis

            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except ImportError:
            raise ImportError("Redis not installed. Install with: pip install redis") from None
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from Redis")

    def set(self, key: str, value: Any, ttl: int | None = None):
        """Set a value in Redis."""
        if not self.client:
            self.connect()

        serialized = json.dumps(value)
        if ttl:
            self.client.setex(key, ttl, serialized)
        else:
            self.client.set(key, serialized)

    def get(self, key: str) -> Any:
        """Get a value from Redis."""
        if not self.client:
            self.connect()

        value = self.client.get(key)
        if value is None:
            return None
        return json.loads(value)

    def delete(self, key: str):
        """Delete a key from Redis."""
        if not self.client:
            self.connect()
        self.client.delete(key)

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        if not self.client:
            self.connect()
        return bool(self.client.exists(key))

    def push_queue(self, queue_name: str, item: Any):
        """Push an item to a Redis list (queue)."""
        if not self.client:
            self.connect()
        serialized = json.dumps(item)
        self.client.rpush(queue_name, serialized)

    def pop_queue(self, queue_name: str, timeout: int = 0) -> Any:
        """Pop an item from a Redis list (queue)."""
        if not self.client:
            self.connect()

        if timeout > 0:
            result = self.client.blpop(queue_name, timeout=timeout)
            if result:
                _, value = result
                return json.loads(value)
        else:
            value = self.client.lpop(queue_name)
            if value:
                return json.loads(value)
        return None

    def queue_length(self, queue_name: str) -> int:
        """Get the length of a Redis list (queue)."""
        if not self.client:
            self.connect()
        return self.client.llen(queue_name)

    def set_heartbeat(self, worker_id: str, ttl: int = 30):
        """Set worker heartbeat with TTL."""
        key = f"heartbeat:{worker_id}"
        self.set(
            key,
            {"timestamp": datetime.now().isoformat(), "status": "alive"},
            ttl=ttl,
        )

    def check_heartbeat(self, worker_id: str) -> bool:
        """Check if worker heartbeat is alive."""
        key = f"heartbeat:{worker_id}"
        return self.exists(key)

    def get_all_workers(self) -> list[str]:
        """Get all active worker IDs."""
        if not self.client:
            self.connect()

        pattern = "heartbeat:*"
        keys = self.client.keys(pattern)
        return [key.replace("heartbeat:", "") for key in keys]


class InMemoryStateStore(StateStore):
    """In-memory state storage for testing/development."""

    def __init__(self):
        self.data = {}
        self.queues = {}

    def connect(self):
        """No-op for in-memory store."""
        pass

    def disconnect(self):
        """No-op for in-memory store."""
        pass

    def set(self, key: str, value: Any, ttl: int | None = None):
        """Set a value in memory."""
        self.data[key] = value
        # TTL not implemented for simplicity

    def get(self, key: str) -> Any:
        """Get a value from memory."""
        return self.data.get(key)

    def delete(self, key: str):
        """Delete a key from memory."""
        self.data.pop(key, None)

    def exists(self, key: str) -> bool:
        """Check if a key exists in memory."""
        return key in self.data

    def push_queue(self, queue_name: str, item: Any):
        """Push an item to an in-memory queue."""
        if queue_name not in self.queues:
            self.queues[queue_name] = []
        self.queues[queue_name].append(item)

    def pop_queue(self, queue_name: str, timeout: int = 0) -> Any:
        """Pop an item from an in-memory queue."""
        if queue_name not in self.queues or not self.queues[queue_name]:
            return None
        return self.queues[queue_name].pop(0)

    def queue_length(self, queue_name: str) -> int:
        """Get the length of an in-memory queue."""
        return len(self.queues.get(queue_name, []))
