#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Wrapper for Pyomo solvers to allow us to define default solver options
"""

from copy import deepcopy

from pyomo.environ import SolverFactory

import idaes


class SolverWrapper(object):
    """
    Wrapper for Pyomo solvers to allow us to define default solver options
    """

    def __init__(self, name, register=True):
        if name is None:
            name = "default"
        self.name = name
        self.registered = register
        if name == "default":
            self.solver = None
            doc = "IDAES Configured Default Solver"
        else:
            self.solver = SolverFactory.get_class(name)
            doc = SolverFactory.doc(name)
        if register:
            SolverFactory.unregister(name)
            # Re-register the solver (register is a decorator)
            SolverFactory.register(name, doc)(self)

    def __call__(self, *args, **kwargs):
        if self.name == "default":
            name = idaes.cfg.default_solver
            solver = SolverFactory.get_class(name)
        else:
            name = self.name
            solver = self.solver
        if name in idaes.cfg and (
            idaes.cfg.use_idaes_solver_config or not self.registered
        ):
            for k, v in idaes.cfg[name].items():
                if k not in kwargs:
                    kwargs[k] = deepcopy(v)
                elif k == "options":
                    # options is in ConfigBlock and in kwargs, treat "options"
                    # special so individual options can have defaults not just
                    # the whole options block
                    for opk, opv in v.items():
                        if opk not in kwargs["options"]:
                            kwargs["options"][opk] = opv
                elif k == "writer_config":
                    # writer_config is in ConfigBlock and in kwargs, treat "writer_config"
                    # special so individual options can have defaults not just
                    # the whole options block
                    for opk, opv in v.items():
                        if opk not in kwargs["writer_config"]:
                            kwargs["writer_config"][opk] = opv

        return solver(*args, **kwargs)


def use_idaes_solver_configuration_defaults(b=True):
    """
    This function enables (or disables if given False as the argument) solvers
    getting default settings from the IDAES configuration.  When enabled this
    allows global configuration of solvers.

    Args:
        b: True to use default solver configurations from the IDAES configuration
           False to use standard Pyomo solver factories. Default is True.

    Returns:
        None
    """
    idaes.cfg.use_idaes_solver_config = b
    if b:  # This will let you explicitly state you don't want any part of this
        # so if you only do "use_idaes_solver_configuration_defaults(False)" up-
        # front you are saying I know this stuff exists and I must insist you
        # don't use it, of course you can still implicitly not use it.  You can
        # also turn it off and on, if that makes sense for you, but once you turn
        # it on, you've still registered the wrapper classes, and if you turn
        # it off they just pass-through.
        for c in list(SolverFactory):
            if isinstance(SolverFactory.get_class(c), SolverWrapper):
                continue
            SolverWrapper(c)
        if "default" not in SolverFactory:
            SolverWrapper("default")
