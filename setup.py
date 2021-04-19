from distutils.core import setup


setup(
  name='gourdian',
  version='0.1',
  install_requires=[
    # Core deps.
    'more_itertools', 'numpy', 'pandas', 'requests',
    # Terminal-based drawing deps for gourd cli.
    'asyncio', 'colored', 'urwid',
  ],
  entry_points={
    'console_scripts': [
      'gourd = gourdian.gourd.gourd:main',
    ],
  },
)
