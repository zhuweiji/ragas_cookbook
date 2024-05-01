import textwrap


def print_long_text(text):
    wrapped_text = textwrap.wrap(text, width=120)  # Wrap text to 70 characters

    for line in wrapped_text:
        print(line)
