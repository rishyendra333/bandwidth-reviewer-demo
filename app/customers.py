"""In-memory customer store for the demo."""

from dataclasses import dataclass


@dataclass
class Customer:
    id: str
    plan: str
    sender_number: str
    balance: float


_CUSTOMERS = {
    "c-100": Customer("c-100", "business", "+19195550100", 250.00),
    "c-200": Customer("c-200", "starter", "+19195550200", 5.00),
}


def get_customer(customer_id: str) -> Customer | None:
    """Return the customer, or None if the ID is unknown."""
    return _CUSTOMERS.get(customer_id)
