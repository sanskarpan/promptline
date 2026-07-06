import promptline
import promptline.cli
import promptline.core
import promptline.data
import promptline.eval
import promptline.judge
import promptline.optimizers
import promptline.registry
import promptline.server
import promptline.tui


def test_version():
    from importlib.metadata import version

    assert promptline.__version__ == version("promptline-opt")


def test_subpackages_importable():
    assert promptline.core is not None
    assert promptline.data is not None
    assert promptline.judge is not None
    assert promptline.optimizers is not None
    assert promptline.eval is not None
    assert promptline.registry is not None
    assert promptline.server is not None
    assert promptline.tui is not None
    assert promptline.cli is not None
