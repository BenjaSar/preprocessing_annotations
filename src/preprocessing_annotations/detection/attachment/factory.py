"""AV-3: attachment-policy factory.

Mirrors vlm_backend.VLMFactory's shape exactly (dict of name -> class,
uniform single-config-arg constructor, register() classmethod) --
same extension pattern already used in this codebase for swapping
VLM backends, applied here to attachment policies (OCP/DIP).
"""

from typing import Dict, Type

from ...config import AttachmentConfig
from .intersection import IntersectionAttachmentPolicy
from .nearest_cutoff import NearestWithCutoffAttachmentPolicy
from .port import AttachmentPolicy


class AttachmentPolicyFactory:
    """Factory for creating AttachmentPolicy instances."""

    _policies: Dict[str, Type] = {
        "intersection": IntersectionAttachmentPolicy,
        "nearest_with_cutoff": NearestWithCutoffAttachmentPolicy,
    }

    @classmethod
    def create(cls, config: AttachmentConfig) -> AttachmentPolicy:
        """Create an AttachmentPolicy instance from config.policy.

        Raises:
            ValueError: If config.policy is not registered.
        """
        name = config.policy.lower()
        if name not in cls._policies:
            raise ValueError(
                f"Unknown attachment policy: {name!r}. "
                f"Available: {', '.join(sorted(cls._policies))}"
            )
        policy_class = cls._policies[name]
        return policy_class(config)

    @classmethod
    def register(cls, name: str, policy_class: Type) -> None:
        """Register a new attachment policy (OCP: extend here, do not
        edit the policies that already ship)."""
        cls._policies[name] = policy_class
