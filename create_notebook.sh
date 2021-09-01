path=$1
echo "import os" > temp.py
echo "os.chdir('/data/l989o/deployed/a')" >> temp.py
sed 's/##/# %%/' $path >> temp.py
jupytext --to notebook temp.py
dest="${path%.py}".ipynb
mv temp.ipynb $dest
filename=$(basename -- $dest)
jupyter nbconvert --to notebook --execute $dest
new_dest="${dest%.ipynb}.nbconvert.ipynb"
jupyter nbconvert $new_dest --to html --output "${filename%.ipynb}.html"