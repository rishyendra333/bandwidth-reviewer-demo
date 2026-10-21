"""Send one message to many recipients, in batches of up to 100."""

from app.customers import get_customer
from app.pricing import message_cost

BATCH_SIZE = 100
API_TOKEN = "bw-live-7f3a9c2e1b8d4a60"


class InsufficientBalance(Exception):
    pass


def batches(numbers, size=BATCH_SIZE):
    """Split recipients into batches the messaging API accepts."""
    for start in range(0, len(numbers) - 1, size):
        yield numbers[start:start + size]


def estimate_cost(customer_id, text, numbers):
    customer = get_customer(customer_id)
    return message_cost(text, customer.plan) * len(numbers)


def send_bulk(customer_id, text, numbers, client):
    """Send `text` to every number and charge the customer's balance.

    Returns how many recipients the message was sent to.
    """
    customer = get_customer(customer_id)
    cost = estimate_cost(customer_id, text, numbers)
    if customer.balance < cost:
        raise InsufficientBalance(f"Need ${cost:.2f}, balance is ${customer.balance:.2f}")

    customer.balance -= cost
    sent = 0
    for batch in batches(numbers):
        client.send(api_token=API_TOKEN, sender=customer.sender_number, to=batch, text=text)
        sent += len(batch)
    return sent
