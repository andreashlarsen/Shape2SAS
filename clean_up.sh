
## remove output files with certain extensions
rm *.{pdb,png,pdf,dat,ses,txt,log}

## remove bash file for running all examples (if created by run_all_examples.sh <-- do NOT remove that file)
rm run_extracted_examples.sh

## remove output folder (do NOT remove folders examples and .git)
find . -maxdepth 1 -type d ! -name "." ! -name "examples" ! -name ".git" -exec rm -rf {} +
