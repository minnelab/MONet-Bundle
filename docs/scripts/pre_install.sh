mkdir -p docs/source/apidocs/tutorials
cp README.md docs/source/README.md
sed -i 's|\./|apidocs/tutorials/|g' docs/source/README.md
cp MONet_Bundle.ipynb docs/source/apidocs/tutorials/MONet_Bundle.ipynb
cp 06_monet_bundle.ipynb docs/source/apidocs/tutorials/06_monet_bundle.ipynb
cp MONet-FL.ipynb docs/source/apidocs/tutorials/MONet-FL.ipynb
cp -r images docs/source/apidocs/tutorials/images
cp -r Projects docs/source/apidocs/tutorials/Projects
python docs/scripts/generate_tutorials_rst.py
python docs/scripts/generate_scripts_rst.py