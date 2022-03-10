##
import os
import sys
import tempfile
import shutil
import re
import subprocess
import shlex
import pathlib

from utils import file_path

assert len(sys.argv) == 2
f = sys.argv[1]
print(f)
import os

os.environ["SPATIALMUON_NOTEBOOK"] = f

with tempfile.TemporaryDirectory() as tempdir:
    assert os.path.isfile(f)
    dest_file = os.path.join(tempdir, os.path.basename(f))
    shutil.copyfile(f, dest_file)
    with open(dest_file, "r") as fp:
        s = fp.read()
        enclosing_folder = pathlib.Path(__file__).parent.resolve()
        prefix = f"""\
##
import os
os.chdir('{enclosing_folder}')

##
%matplotlib inline

##
"""
        s = prefix + s
        s = re.sub(r"^##", "# %%", s, flags=re.MULTILINE)
    with open(dest_file, "w") as fp:
        fp.write(s)
    ##
    activate_env = "source ~/.bashrc; conda activate ci_env"
    subprocess.check_output(
        f'bash -c "{activate_env}; jupytext --to notebook {shlex.quote(dest_file)}"',
        shell=True,
    )
    ##
    jupyter_file = dest_file.replace(".py", ".ipynb")
    DEBUG = True
    if DEBUG:
        debug = '--debug'
    else:
        debug = ''
    cmd = (
        f'bash -c "{activate_env}; jupyter nbconvert --to notebook {debug} --ExecutePreprocessor.timeout=-1 '
        f'--execute {shlex.quote(jupyter_file)}"'
    )
    print(cmd)
    subprocess.check_output(cmd, shell=True)
    ##
    jupyter_file_executed = jupyter_file.replace(".ipynb", ".nbconvert.ipynb")
    html_file = jupyter_file.replace(".ipynb", ".html")
    subprocess.check_output(
        f'bash -c "{activate_env}; jupyter nbconvert {shlex.quote(jupyter_file_executed)} --to html --output '
        f'{shlex.quote(html_file)}"',  # --HTMLExporter.theme=gruvboxd
        shell=True,
    )
    OUTPUT_FOLDER = file_path("html_exports")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    prefix_dir = os.path.dirname(f)
    # prefix_dir = '.' if prefix_dir == '' else prefix_dir
    full_dir = os.path.join(OUTPUT_FOLDER, prefix_dir)
    os.makedirs(full_dir, exist_ok=True)
    html_des = os.path.join(full_dir, os.path.basename(html_file))
    shutil.copyfile(html_file, html_des)
    print(f"created html export in {html_des}")
