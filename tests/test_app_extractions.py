"""Identity tests for the IMP-001 app.py extractions (S3B.1-S3B.3).

These pin the extraction contracts: the moved artifacts must be THE SAME
objects the app still uses (no copies), so behavior is provably unchanged.
"""

from quantui import app as appmod


def test_css_is_extracted_verbatim():
    """QuantApp.CSS must BE the MAIN_CSS object from quantui.app_css."""
    from quantui import app_css

    assert appmod.QuantApp.CSS is app_css.MAIN_CSS


def test_command_palette_provider_wired():
    """The extracted QuantCommands must be registered on QuantApp.COMMANDS."""
    from quantui import palette
    from quantui.palette import QuantCommands

    # The class moved verbatim: app.py re-exports the SAME object.
    assert palette.QuantCommands is appmod.QuantCommands
    # And the S3.1 wiring still registers it on the app.
    assert QuantCommands in set(appmod.QuantApp.COMMANDS or ())
