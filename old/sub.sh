#!/bin/sh
#$ -q gen3.q
#$ -N methylene_dirad
#$ -S /bin/sh
#$ -cwd

. /etc/profile.d/modules.sh

# Disable production of core dump files
ulimit -c 0

echo ""
echo "***********************************************************************"
echo " Starting job:"
echo ""
echo "    Name:              "$JOB_NAME
echo "    ID:                "$JOB_ID
echo "    Hostname:          "$HOSTNAME
echo "    Working directory: "$SGE_O_WORKDIR
echo ""
echo "***********************************************************************"


conda activate helpme

# Load the requested Psi4 module file
#vulcan load psi4@master~ambit~chemps2~debug~pcmsolver~vectorization

#export PSI_SCRATCH=$TMPDIR
#export KMP_DUPLICATE_LIB_OK=TRUE
#
#psi4 -n 6 -i input.dat -o output.dat

