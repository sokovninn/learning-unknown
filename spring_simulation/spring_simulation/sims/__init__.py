from habitat.core.registry import registry
from habitat.core.simulator import Simulator

# from habitat.sims.spring_simulator.actions import (
#     HabitatSimV1ActionSpaceConfiguration,
# )


def _try_register_spring_sim():

    try:
        import habitat_sim
        has_habitat_sim = True
    except ImportError as e:
        has_habitat_sim = False
        habitat_sim_import_error = e

    if has_habitat_sim:
        from spring_simulation.sims.spring_simulator import SpringSim
        from spring_simulation.sims.actions import (
            SpringSimV0ActionSpaceConfiguration,
        )
    else:

        @registry.register_simulator(name="Sim-v0")
        class HabitatSimImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise habitat_sim_import_error

_try_register_spring_sim()
