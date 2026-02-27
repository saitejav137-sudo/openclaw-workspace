"""Dependency Injection Container for OpenClaw

Provides a simple dependency injection container for managing
component dependencies and improving testability.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar, get_type_hints
from dataclasses import dataclass
from functools import wraps
import threading

T = TypeVar('T')


class DependencyError(Exception):
    """Error raised when dependency resolution fails"""
    pass


@dataclass
class Dependency:
    """Represents a registered dependency"""
    factory: Callable[..., Any]
    singleton: bool = True
    instance: Optional[Any] = None


class Container:
    """Simple dependency injection container"""

    _instance: Optional['Container'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._dependencies: Dict[str, Dependency] = {}
        self._parent: Optional['Container'] = None

    @classmethod
    def get_instance(cls) -> 'Container':
        """Get the global container instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = Container()
            return cls._instance

    @classmethod
    def set_instance(cls, container: 'Container') -> None:
        """Set the global container instance (for testing)"""
        cls._instance = container

    def register(
        self,
        interface: Type[T],
        factory: Callable[..., T],
        singleton: bool = True
    ) -> None:
        """Register a dependency

        Args:
            interface: The interface/abstract class type
            factory: Factory function to create the instance
            singleton: If True, creates one instance and reuses it
        """
        key = f"{interface.__module__}.{interface.__name__}"
        self._dependencies[key] = Dependency(
            factory=factory,
            singleton=singleton
        )

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register an existing instance

        Args:
            interface: The interface/abstract class type
            instance: The existing instance
        """
        key = f"{interface.__module__}.{interface.__name__}"
        self._dependencies[key] = Dependency(
            factory=lambda: instance,
            singleton=True,
            instance=instance
        )

    def resolve(self, interface: Type[T], **kwargs: Any) -> T:
        """Resolve a dependency

        Args:
            interface: The interface/abstract class type
            **kwargs: Additional arguments to pass to factory

        Returns:
            The resolved instance

        Raises:
            DependencyError: If dependency is not registered
        """
        key = f"{interface.__module__}.{interface.__name__}"

        # Check this container
        if key in self._dependencies:
            dep = self._dependencies[key]
            return self._create_instance(dep, **kwargs)

        # Check parent container
        if self._parent:
            return self._parent.resolve(interface, **kwargs)

        raise DependencyError(f"Dependency not registered: {interface.__name__}")

    def _create_instance(self, dep: Dependency, **kwargs: Any) -> Any:
        """Create or get an instance"""
        if dep.singleton:
            if dep.instance is None:
                dep.instance = dep.factory(**kwargs)
            return dep.instance
        return dep.factory(**kwargs)

    def clear(self) -> None:
        """Clear all registered dependencies (for testing)"""
        self._dependencies.clear()

    def clear_singletons(self) -> None:
        """Clear all singleton instances (for testing)"""
        for dep in self._dependencies.values():
            dep.instance = None


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global dependency container"""
    global _container
    if _container is None:
        _container = Container.get_instance()
    return _container


def set_container(container: Container) -> None:
    """Set the global container (for testing)"""
    global _container
    _container = container


# Decorator for dependency injection
def injectable(cls: Type[T]) -> Type[T]:
    """Decorator to mark a class as injectable

    Usage:
        @injectable
        class MyService:
            def __init__(self, db: Database):
                self.db = db
    """
    @wraps(cls)
    class Wrapper:
        def __init__(self, **kwargs: Any):
            container = get_container()
            # Get type hints for __init__
            hints = get_type_hints(cls.__init__)
            # Resolve dependencies
            init_kwargs = {}
            for param_name, param_type in hints.items():
                if param_name == 'self' or param_name == 'kwargs':
                    continue
                if param_name in kwargs:
                    init_kwargs[param_name] = kwargs[param_name]
                else:
                    try:
                        init_kwargs[param_name] = container.resolve(param_type)
                    except DependencyError:
                        # Parameter not resolved, will use default or fail
                        pass

            self._wrapped = cls(**init_kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

    return Wrapper


# Convenience functions for common registrations
def register_singleton(interface: Type[T], factory: Callable[..., T]) -> None:
    """Register a singleton dependency"""
    get_container().register(interface, factory, singleton=True)


def register_factory(interface: Type[T], factory: Callable[..., T]) -> None:
    """Register a factory (non-singleton) dependency"""
    get_container().register(interface, factory, singleton=False)


def resolve(interface: Type[T], **kwargs: Any) -> T:
    """Resolve a dependency from the container"""
    return get_container().resolve(interface, **kwargs)


__all__ = [
    "Container",
    "Dependency",
    "DependencyError",
    "get_container",
    "set_container",
    "injectable",
    "register_singleton",
    "register_factory",
    "resolve",
]
