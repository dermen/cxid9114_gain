import libtbx.load_env
import os
Import("env_base", "env_etc")

env_etc.eigen_dist = os.path.abspath(os.path.join(libtbx.env.dist_path("boost"),"../eigen"))
if os.path.isdir(env_etc.eigen_dist):
  env_etc.eigen_include = env_etc.eigen_dist
  env_etc.cxid9114_solvers_common_includes = [
    env_etc.eigen_include,
    env_etc.libtbx_include,
    env_etc.scitbx_include,
    env_etc.boost_include,
    ]

  env = env_base.Clone(SHLINKFLAGS=env_etc.shlinkflags)
  env.Append(LIBS=["cctbx"] + env_etc.libm)
  env_etc.include_registry.append(
    env=env,
    paths=env_etc.cxid9114_solvers_common_includes)

  if (env_etc.static_libraries): builder = env.StaticLibrary
  else:                          builder = env.SharedLibrary

  if (not env_etc.no_boost_python):
    Import("env_boost_python_ext")
    env_cxid9114_solvers_boost_python_ext = env_boost_python_ext.Clone()
    env_cxid9114_solvers_boost_python_ext.SharedLibrary(
				 target="#lib/cxid9114_solvers_ext", source="solvers/solvers_ext.cpp")
    env_etc.include_registry.append(
	  env=env_cxid9114_solvers_boost_python_ext,
	  paths=env_etc.cxid9114_solvers_common_includes)
    Export("env_cxid9114_solvers_boost_python_ext")

