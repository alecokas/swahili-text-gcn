from html.parser import HTMLParser


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, d):
        self.text.append(d)

    def get_data(self):
        return ' '.join(self.text)


def strip_tags(html: str) -> str:
    s = HTMLStripper()
    s.feed(html)
    return s.get_data()


def ignore_non_ascii(text: str) -> str:
    return text.encode('ascii', errors='ignore').decode()
