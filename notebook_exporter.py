##
import os
import sys
import tempfile
import shutil
import re
import subprocess
import shlex

from utils import file_path

assert len(sys.argv) == 2
f = sys.argv[1]
print(f)
import os

os.environ["NOTEBOOK_EXPORTER"] = "aaa"

with tempfile.TemporaryDirectory() as tempdir:
    assert os.path.isfile(f)
    dest_file = os.path.join(tempdir, os.path.basename(f))
    shutil.copyfile(f, dest_file)
    with open(dest_file, "r") as fp:
        s = fp.read()
        prefix = """\
##
import os
os.chdir('/data/l989o/deployed/a')

##
%matplotlib inline

##
"""
        s = prefix + s
        s = re.sub(r"^##", "# %%", s, flags=re.MULTILINE)
    with open(dest_file, "w") as fp:
        fp.write(s)
    ##
    subprocess.check_output(
        f'bash -c "source activate ci_env; jupytext --to notebook {shlex.quote(dest_file)}"',
        shell=True,
    )
    ##
    jupyter_file = dest_file.replace(".py", ".ipynb")
    subprocess.check_output(
        f'bash -c "source activate ci_env; jupyter nbconvert --to notebook --execute {shlex.quote(jupyter_file)}"',
        shell=True,
    )
    ##
    jupyter_file_executed = jupyter_file.replace('.ipynb', '.nbconvert.ipynb')
    html_file = jupyter_file.replace(".ipynb", ".html")
    subprocess.check_output(
        f'bash -c "source activate ci_env; jupyter nbconvert {shlex.quote(jupyter_file_executed)} --to html --output '
        f'{shlex.quote(html_file)}"',  # --HTMLExporter.theme=gruvboxd
        shell=True,
    )
    OUTPUT_FOLDER = file_path('html_exports')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    prefix_dir = os.path.dirname(f)
    # prefix_dir = '.' if prefix_dir == '' else prefix_dir
    full_dir = os.path.join(OUTPUT_FOLDER, prefix_dir)
    os.makedirs(full_dir, exist_ok=True)
    html_des = os.path.join(full_dir, os.path.basename(html_file))
    shutil.copyfile(html_file, html_des)
    print(f'created html export in {html_des}')
