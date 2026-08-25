"""TUI entry point: run with ``python -m quantui`` or the ``quantui`` script.

NTH-002 STEP 4.1: ``main()`` exists so ``[project.scripts]`` can target
``quantui.__main__:main``; the ``__name__`` guard is kept for direct module
execution. The import lives inside ``main()`` so importing this module (e.g.
by packaging smoke tests) does not pull in Textual/app machinery.
"""


def main() -> None:
    """Launch the quantui TUI."""
    from .app import QuantApp

    QuantApp().run()


if __name__ == "__main__":
    main()
