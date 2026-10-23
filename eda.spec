# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['statsmodels.tsa.arima.model', 'statsmodels.tsa.statespace.sarimax', 'statsmodels.tsa.seasonal', 'statsmodels.tsa.holtwinters', 'matplotlib.backends.backend_agg', 'pandas._libs.tslibs.timedeltas', 'pandas._libs.tslibs.np_datetime', 'scipy.special', 'scipy.special._ufuncs', 'scipy.linalg._fblas', 'scipy.linalg._flapack', 'scipy.sparse.csgraph._validation', 'scipy.spatial.transform._rotation']
datas += collect_data_files('sklearn')
datas += collect_data_files('openpyxl')
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('openpyxl')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('statsmodels')
hiddenimports += collect_submodules('eda_toolkit')
tmp_ret = collect_all('plotly')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('reportlab')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('python_stdnum')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\Users\\enise\\Documents\\Projelerim\\eda_analysis\\build\\launcher.py'],
    pathex=['C:\\Users\\enise\\Documents\\Projelerim\\eda_analysis\\src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='eda',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='eda',
)
