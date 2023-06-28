#!/bin/bash
 
# make a temp file 
TMP_FILE=$(mktemp -q /home/tmp/XXXXXX)
if [ $? -ne 0 ]; then
    echo "$0: Can't create temp file, bye.."
    exit 1
fi

# write content to tmp file
echo '#!/bin/bash
'$1 > $TMP_FILE
sbatch -c 2 $TMP_FILE

rm $TMP_FILE

