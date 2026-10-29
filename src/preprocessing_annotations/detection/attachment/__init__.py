"""AV-3: object-detection -> room attachment policies.

Public surface: AttachmentPolicy (port), IntersectionAttachmentPolicy
(default, byte-identical to the pre-AV-3 map_detections_to_rooms call
sites), NearestWithCutoffAttachmentPolicy (Prototype First, unvalidated),
and AttachmentPolicyFactory (config -> policy instance, mirrors
vlm_backend.VLMFactory's shape).
"""

from .factory import AttachmentPolicyFactory
from .intersection import IntersectionAttachmentPolicy
from .nearest_cutoff import NearestWithCutoffAttachmentPolicy
from .port import AttachmentPolicy

__all__ = [
    "AttachmentPolicy",
    "AttachmentPolicyFactory",
    "IntersectionAttachmentPolicy",
    "NearestWithCutoffAttachmentPolicy",
]
