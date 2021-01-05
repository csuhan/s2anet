from distutils.core import setup, Extension

polyiou_module = Extension(
    '_polyiou',
    sources=['src/polyiou_wrap.cxx', 'src/polyiou.cpp'])
setup(name='polyiou',
      version='0.1',
      author="SWIG Docs",
      description="""Simple swig example from docs""",
      ext_modules=[polyiou_module],
      py_modules=["polyiou"],
      )
