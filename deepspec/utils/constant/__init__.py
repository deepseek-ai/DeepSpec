try:
    import hfai
    from .internal import *
except (ImportError, ModuleNotFoundError):
    from .public import *