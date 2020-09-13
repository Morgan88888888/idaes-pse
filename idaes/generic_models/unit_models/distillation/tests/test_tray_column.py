##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Tests for tray column unit model (single feed tray, no side draws).

Author: Jaffer Ghouse
"""
import pytest
from pyomo.environ import (ConcreteModel, TerminationCondition,
                           SolverStatus, value)

from idaes.core import (FlowsheetBlock, MaterialBalanceType, EnergyBalanceType,
                        MomentumBalanceType)
from idaes.generic_models.unit_models.distillation import TrayColumn
from idaes.generic_models.unit_models.distillation.condenser \
    import CondenserType, TemperatureSpec
from idaes.generic_models.properties.activity_coeff_models.\
    BTX_activity_coeff_VLE import BTXParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom, \
    number_variables, number_total_constraints, number_unused_variables, \
    fixed_variables_set, activated_constraints_set
from idaes.core.util.testing import get_default_solver, \
    PhysicalParameterTestBlock, initialization_tester


# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_default_solver()


@pytest.mark.unit
def test_config():

    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.properties = BTXParameterBlock(default={"valid_phase":
                                                 ('Liq', 'Vap'),
                                                 "activity_coeff_model":
                                                 "Ideal"})

    m.fs.unit = TrayColumn(default={
                           "number_of_trays": 10,
                           "feed_tray_location": 5,
                           "condenser_type": CondenserType.totalCondenser,
                           "condenser_temperature_spec":
                               TemperatureSpec.atBubblePoint,
                           "property_package": m.fs.properties,
                           "has_heat_transfer": False,
                           "has_pressure_change": False})

    assert len(m.fs.unit.config) == 12

    assert m.fs.unit.tray[5].config.is_feed_tray

    assert hasattr(m.fs.unit, "condenser")
    assert hasattr(m.fs.unit, "reboiler")

    assert hasattr(m.fs.unit, "liq_stream")
    assert hasattr(m.fs.unit, "vap_stream")

    # Inlet feed conditions
    m.fs.unit.tray[5].feed.flow_mol.fix(100)
    m.fs.unit.tray[5].feed.temperature.fix(368)
    m.fs.unit.tray[5].feed.pressure.fix(101325)
    m.fs.unit.tray[5].feed.mole_frac_comp[0, "benzene"].fix(0.5)
    m.fs.unit.tray[5].feed.mole_frac_comp[0, "toluene"].fix(0.5)

    # unit level inputs
    m.fs.unit.condenser.reflux_ratio.fix(1.4)
    m.fs.unit.condenser.condenser_pressure.fix(101325)

    m.fs.unit.reboiler.boilup_ratio.fix(1.3)

    assert degrees_of_freedom(m) == 0

    m.fs.unit.initialize()
