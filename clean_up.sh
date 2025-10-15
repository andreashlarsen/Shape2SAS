rm *pdb
rm *png
rm *pdf
rm *dat
rm *ses
rm *txt
rm shape2sas.log
find . -maxdepth 1 -type d ! -name "." ! -name "examples" ! -name ".git" -exec rm -rf {} +
