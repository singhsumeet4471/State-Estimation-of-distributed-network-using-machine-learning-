#!C:\Users\singh\PycharmProjects\Thesis\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'PYPOWER==5.1.4','console_scripts','pf'
__requires__ = 'PYPOWER==5.1.4'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('PYPOWER==5.1.4', 'console_scripts', 'pf')()
    )
