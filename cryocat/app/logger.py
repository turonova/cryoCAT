class StreamToList:
    def __init__(self):
        self.buffer = []  # list of (msg, source)

    def write(self, msg, source="dash"):
        if msg.strip():
            self.buffer.append((msg.strip(), source))

    def flush(self):
        pass

    def get_logs(self, last_index=0):
        """Return new logs since last_index, and if any are dash-triggered."""
        sliced = self.buffer[last_index:]
        new_messages = [msg for msg, _ in sliced]
        new_dash = any(source == "dash" for _, source in sliced)
        return new_messages, len(self.buffer), new_dash

    def get_all_logs(self):
        return "\n".join(msg for msg, _ in self.buffer)

    def clear(self):
        self.buffer.clear()


def print_dash(*args):
    dash_logger.write(" ".join(map(str, args)), source="dash")


# Global singleton instance
dash_logger = StreamToList()
