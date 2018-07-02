#!C:\Users\singh\PycharmProjects\Thesis\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'pandas-montecarlo==0.0.2','console_scripts','sample'
__requires__ = 'pandas-montecarlo==0.0.2'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('pandas-montecarlo==0.0.2', 'console_scripts', 'sample')()
    )
