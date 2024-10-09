import sys
from io import StringIO


class FilteredStderr(StringIO):
    def __init__(self, keywords, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keywords = keywords

    def write(self, message):
        # Check if any keyword is present in the message
        if any(keyword in message for keyword in self.keywords):
            # Write to actual stderr if keyword matches
            return
        sys.__stderr__.write(message)


def suppress_plbolt_warnings():
    sys.stderr = FilteredStderr(["pl_bolt"])  # Redirect standard error (warnings)
