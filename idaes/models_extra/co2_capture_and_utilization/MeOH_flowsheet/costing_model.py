import numpy as np
import os
import json
from pyomo.common.fileutils import this_file_dir
from pyomo.environ import (
    ConcreteModel,
    TransformationFactory,
    SolverFactory,
    Var,
    Param,
    Constraint,
    Expression,
    Objective,
    value,
    log,
    exp,
    units as pyunits,
    assert_optimal_termination
)
from pyomo.network import Arc
from idaes.core import FlowsheetBlock, UnitModelBlock, UnitModelCostingBlock
from idaes.core.solvers import get_solver
from idaes.core.util import scaling as iscale
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
from idaes.core.initialization import (
    BlockTriangularizationInitializer,
)

from power_plant_capcost_ref import (
    QGESSCosting,
    QGESSCostingData,
)
__author__ = "Maojian Wang"

pyunits.load_definitions_from_strings(
    [
        # custom units related to polymer layers plants
        # 1 lbmol = 1 g-mole * 453.59237 lb/g
        "lbmol = 453.59237 * mol",
        # https://toweringskills.com/financial-analysis/cost-indices/
        # 2018 value is 603.1, Dec 2021 value is 776.3
        # https://www.chemengonline.com/2021-cepci-updates-december-prelim-and-november-final/
        # Dec 2021 is 28.0% higher than Dec 2020
        # https://www.chemengonline.com/2020-cepci-updates-december-prelim-and-november-final/
        # Dec 2020 is 2.5% higher than Dec 2019
        # https://www.chemengonline.com/2019-cepci-updates-december-prelim-and-november-final/
        # Dec 2019 is 3.9% lower than Dec 2018
        # Dec 2021 = 776.3 = Dec 2018 * (1 - 3.9/100) * (1 + 2.5/100) * (1 + 28.0/100)
        # >>> Dec 2018 = 615.7
        "USD_2018_Dec = 615.7/500 * USD_CE500",
    ]
)

def build_costing(
        blk,
        cost_year,
        capacity_factor,
        ):

    blk.costing = QGESSCosting()
    blk.costing.capacity_factor = capacity_factor

    # check that the model solved properly and has 0 degrees of freedom
    assert degrees_of_freedom(blk) == 0

    ###########################################################################
    #  Create costing constraints                                             #
    ###########################################################################
    
    # MeOH production flowsheet accounts have tech number 10A so that no native accounts are
    # unintentionally overwritten

    directory = this_file_dir()
    with open(os.path.join(directory, "cost_info.json"), "r") as file:
            PL_costing_params = json.load(file)


    #=============================================================================
    # Add Accounts for TPC 
    #=============================================================================
    tech_id = 10
    blk.cost_VFC =  UnitModelBlock()
    VFC_accounts = ["15.6","15.7","15.9"]
    blk.Volumetric_Flow_Compressor = Var(initialize=987591.8426, 
                                       units=pyunits.cu_ft/pyunits.hr)
    blk.Volumetric_Flow_Compressor.fix()
    
    blk.cost_VFC.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": VFC_accounts,
            "scaled_param": blk.Volumetric_Flow_Compressor,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )

    blk.cost_RWW =  UnitModelBlock()
    RWW_accounts = ["3.2","9.5","14.3","14.7"]
    blk.Raw_Water_Withdrawal = Var(initialize=3719.7234, 
                                       units=pyunits.gal/pyunits.min)
    blk.Raw_Water_Withdrawal.fix()
    blk.cost_RWW.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": RWW_accounts,
            "scaled_param": blk.Raw_Water_Withdrawal,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )

    blk.cost_PWD =  UnitModelBlock()
    PWD_accounts = ["3.7"]
    blk.Process_Water_Discharge = Var(initialize=774.9992, 
                                       units=pyunits.gal/pyunits.min)
    blk.Process_Water_Discharge.fix()
    blk.cost_PWD.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": PWD_accounts,
            "scaled_param": blk.Process_Water_Discharge,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )

    blk.cost_NAL = UnitModelBlock()
    NAL_accounts = ["12.1","12.2","12.3","12.4"]
    blk.Net_Auxiliary_Load = Var(initialize= 110.180, 
                            units=pyunits.MW)
    blk.Net_Auxiliary_Load.fix()
    blk.cost_NAL.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": NAL_accounts,
            "scaled_param": blk.Net_Auxiliary_Load,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )


    blk.cost_MP = UnitModelBlock()
    MP_accounts = ['14.1','14.4','14.5','14.6','15.1','15.2','15.3','15.4','15.5']
    blk.Meoh_Production = Var(initialize= 60999867.55,
                               units = pyunits.gal/pyunits.year)
    blk.Meoh_Production.fix()
    blk.cost_MP.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": MP_accounts,
            "scaled_param": blk.Meoh_Production,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )

    blk.cost_HD = UnitModelBlock()
    HD_accounts = ['15.8']
    blk.Heat_Duty = Var(initialize= 141.0547, #MMBTU/hr
                               units = pyunits.dimensionless)
    blk.Heat_Duty.fix()
    blk.cost_HD.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": HD_accounts,
            "scaled_param": blk.Heat_Duty,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )

    blk.cost_HP = UnitModelBlock()
    HP_accounts = ['16.1','16.2','16.3']
    blk.H2_Production = Var(initialize= 56500/24, 
                               units = pyunits.kg/pyunits.hr)
    blk.H2_Production.fix()
    blk.cost_HP.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": HP_accounts,
            "scaled_param": blk.H2_Production,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )

    blk.cost_FGF = UnitModelBlock()
    FGF_accounts = ['3.8']
    blk.Feed_Gas_Flow = Var(initialize= 213694.423, 
                               units = pyunits.lb/pyunits.hr)
    blk.Feed_Gas_Flow.fix()
    blk.cost_FGF.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": FGF_accounts,
            "scaled_param": blk.Feed_Gas_Flow,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )
    blk.cost_CTD = UnitModelBlock()
    CTD_accounts = ['9.1']
    blk.Cooling_Tower_Duty = Var(initialize= 1597.2452, #MMBTU/hr
                               units = pyunits.dimensionless)
    blk.Cooling_Tower_Duty.fix()
    blk.cost_CTD.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": CTD_accounts,
            "scaled_param": blk.Cooling_Tower_Duty,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )
    blk.cost_CWF = UnitModelBlock()
    CWF_accounts = ['9.2','9.3','9.4','9.6','9.7','14.2']
    blk.Circulating_Water_Flow = Var(initialize= 159725.1579, 
                            units=pyunits.gal/pyunits.min)
    blk.Circulating_Water_Flow.fix()
    blk.cost_CWF.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": CWF_accounts,
            "scaled_param": blk.Circulating_Water_Flow,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )

    blk.cost_PBEC = UnitModelBlock()
    PBEC_accounts = ['13.1','13.2','13.3']
    blk.Partial_BEC = Var(initialize= 202242.022, #$/year maybe
                               units = pyunits.dimensionless)
    blk.Partial_BEC.fix()
    blk.cost_PBEC.costing = UnitModelCostingBlock(
        flowsheet_costing_block=blk.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": PBEC_accounts,
            "scaled_param": blk.Partial_BEC,
            "tech": tech_id,
            "ccs": "A",
            "additional_costing_params": PL_costing_params,
            "CE_index_year": cost_year,
            "multiply_project_conting": False,
        },
    )


    #=============================================================================
    # Add Varoable Cost
    #=============================================================================

    resources = ["water", 
                 "chemicals", 
                 "catalyst",
                 "electricity"]
    prices = {
          "water": 1.90/1000*pyunits.USD_2018_Dec/pyunits.gal , 
          "chemicals" :  550.00*pyunits.USD_2018_Dec/pyunits.ton, 
          "catalyst": 264.72*pyunits.USD_2018_Dec/pyunits.ft**3,
          "electricity":60.00*pyunits.USD_2018_Dec/pyunits.MWh,
    }
    blk.water = Var(blk.time,initialize=481000, units=pyunits.gal/pyunits.day)
    blk.chemicals = Var(blk.time,initialize=1.4, units=pyunits.ton/pyunits.day)
    blk.catalyst = Var(blk.time,initialize=0.5, units=pyunits.ft**3/pyunits.day)
    blk.electricity = Var(blk.time,initialize=5266, units=pyunits.MWh/pyunits.day)
    rates =[blk.water,blk.chemicals,blk.catalyst,blk.electricity]
    for var in rates:
         var.fix()

    blk.costing.land_cost_expression = Expression(expr=0.017*pyunits.MUSD_2018_Dec) 
    blk.costing.transport_cost_per_tonne_CO2 = Expression(
        expr=0*pyunits.USD_2018_Dec/pyunits.tonne
        ) 
    blk.costing.tonne_CO2_capture = Var(
        initialize=pyunits.convert(
            0 * pyunits.kg/pyunits.hr,
            to_units=pyunits.tonne/pyunits.year
            ),
        units=pyunits.tonne/pyunits.year
        )
    blk.costing.tonne_CO2_capture.fix()

    blk.costing.build_process_costs(
        capacity_factor=capacity_factor,
        fixed_OM=True,
        labor_rate=38.50,
        labor_burden=30,
        operators_per_shift=0.6,
        tech=tech_id, 
        land_cost=blk.costing.land_cost_expression,
        variable_OM=True,
        feedstock = ["chemicals","catalyst"],
        resources=resources,
        rates=rates,
        prices=prices,
        waste= resources,
        chemicals=[
            i for i in resources if i not in
            ["water","electricity"]
            ],
        transport_cost_per_tonne_CO2=blk.costing.transport_cost_per_tonne_CO2,
        tonne_CO2_capture=blk.costing.tonne_CO2_capture,  
        #annual_production_rate=blk.Meoh_Production,
        CE_index_year="2018_Dec"
        )


def report_costing_results(blk):
    #QGESSCostingData.report(blk.costing)
    QGESSCostingData.display_total_plant_costs(blk.costing)
    print()
    print("Owner's Costs Breakdown")
    print("=======================")
    print()
    
    print("6 months All Labor [$/1,000]: ", 1e3*value(blk.costing.six_month_labor))
    print("1-month Maintenance Materials [$/1,000]: ", 1e3*value(blk.costing.maintenance_material_cost/12/blk.costing.capacity_factor))
    print("1-month Non-Fuel Consumables [$/1,000]: ", 1e3*value(blk.costing.non_fuel_and_waste_OC))
    print("1-month Waste Disposal [$/1,000]: ", 1e3*value(blk.costing.waste_cost_OC))                                                                                
    print("60-Day Supply of Chemical Consumables [$/1,000]: ", 1e3*value(blk.costing.feedstock_cost_OC))
    print("0.5% TPC for Spare Parts + Other Owner's + Financing + 2% TPC [$/1,000]: ", 1e3*value(blk.costing.pct_TPC * blk.costing.total_TPC))
    print("Land [$/1,000]: ", 1e3*value(blk.costing.land_cost))
    
    print()
    print("Cost Summary")
    print("=======================")
    print("Capital LCOP [$/kg formic acid]: ", value(blk.costing.annualized_cost*1e6/(blk.Meoh_Production*blk.costing.capacity_factor)))
    print("Fixed O&M LCOP [$/kg formic acid]: ", value(blk.costing.total_fixed_OM_cost*1e6/(blk.Meoh_Production*blk.costing.capacity_factor)))
    print("Variable O&M LCOP [$/kg formic acid]: ", value(blk.costing.total_variable_OM_cost[0]*1e6/(blk.Meoh_Production*blk.costing.capacity_factor)))
    #print("Total LCOP [$/kg formic acid]: ", value(blk.costing.cost_of_production*1e6))


def model_checker(model, solver, solver_info):
    dof = degrees_of_freedom(model)
    assert dof == 0, f"Error: Degrees of freedom is {dof}, but it should be 0."
    print("DOF check passed")

    solver = get_solver(solver)
    results = solver.solve(model, tee = solver_info)    
    print(results) 
    assert_optimal_termination(results)

    return print(" model passed the check")


if __name__ == "__main__":
    m = ConcreteModel("MeOH costing")
    m.fs = FlowsheetBlock(dynamic=False, time_set=[0], time_units=pyunits.s)    
    build_costing(
    m.fs,
    cost_year="2018_Dec",
    capacity_factor=0.85,
    )
    QGESSCostingData.costing_initialization(m.fs.costing)
    model_checker(model = m, solver = "ipopt", solver_info = True)
    report_costing_results(m.fs)
