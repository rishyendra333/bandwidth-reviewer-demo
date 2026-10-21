from app.bulk_send import send_bulk


class FakeClient:
    def __init__(self):
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)


def test_send_bulk_happy_path():
    client = FakeClient()
    sent = send_bulk("c-100", "Your order has shipped", ["+19195550001", "+19195550002", "+19195550003"], client)
    assert sent == 3
