"""The output contract: what one extracted change order looks like.

Every field has the same shape, {"value": ..., "confidence": 0-1}. `value` is null
when the document does not contain the field; we never fill it with a guess.
schema/change_order.schema.json is generated from these classes, so the published
schema and the code cannot drift apart.
"""
import datetime as dt
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ApprovalStatus(str, Enum):
    approved = "approved"
    pending = "pending"
    rejected = "rejected"


class Extracted(BaseModel, Generic[T]):
    """One field: the value we found and how much we trust it."""

    model_config = ConfigDict(extra="forbid")

    value: T | None = Field(
        description="The value as found in the document, normalised. Null if the document does not contain it."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "How likely the value is correct, from 0 to 1. For a null value: how likely "
            "it is that the field really is absent from the document."
        ),
    )


class ChangeOrderExtraction(BaseModel):
    """One change order, extracted and checked."""

    model_config = ConfigDict(extra="forbid")

    project_name: Extracted[str]
    contract_number: Extracted[str]
    change_order_number: Extracted[str]
    date: Extracted[dt.date] = Field(description="Date of the change order, ISO 8601 (YYYY-MM-DD).")
    requested_by: Extracted[str]
    description: Extracted[str]
    amount: Extracted[float] = Field(
        description="Change to the contract price in the document's currency: increase positive, decrease negative."
    )
    schedule_impact_days: Extracted[int] = Field(
        description="Change to the contract time in calendar days: extension positive, reduction negative, 0 for no change."
    )
    approval_status: Extracted[ApprovalStatus]
    needs_review: bool = Field(
        description="True when any check failed or any field's confidence is below the review threshold."
    )
    review_reasons: list[str] = Field(
        default_factory=list,
        description="Plain-language reasons why needs_review is true. Empty when it is false.",
    )
