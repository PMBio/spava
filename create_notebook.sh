function process () {
  path=$1
  echo "import os" > temp.py
  echo "os.chdir('/data/l989o/deployed/a')" >> temp.py
  sed 's/##/# %%/' $path >> temp.py
  jupytext --to notebook temp.py
  dest="${path%.py}".ipynb
  mv temp.ipynb $dest
  #filename=$(basename -- $dest)
  #jupyter nbconvert --to notebook --execute $dest
  #new_dest="${dest%.ipynb}.nbconvert.ipynb"
  #jupyter nbconvert $new_dest --to html --output "${filename%.ipynb}.html"
}

#python_file=$1

for python_file in `find analyses/*/ -name '*.py' | grep -v ipynb_checkpoints | grep -v init`
do
  echo $python_file
  process $python_file
done
