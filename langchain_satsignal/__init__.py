"""langchain-satsignal: anchor LangChain agent decisions on BSV via Satsignal.

See README.md for usage. The four-anchor pattern (policy snapshot at
top-level chain start, commit-reveal per decision, evidence-bundle
manifest at top-level chain end) is bound to a LangChain callback.
"""
from .callback import SatsignalCallbackHandler, SatsignalConfig
from ._anchor import APIError

__all__ = ["SatsignalCallbackHandler", "SatsignalConfig", "APIError"]
__version__ = "0.1.0"
