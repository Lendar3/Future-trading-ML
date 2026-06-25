from __future__ import annotations

from app.paper.broker_sim import PaperBrokerSim


class DemoExecutor(PaperBrokerSim):
    """Demo executor currently routes to paper simulation until real demo wiring is explicit."""
