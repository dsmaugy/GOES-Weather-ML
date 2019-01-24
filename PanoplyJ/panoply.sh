#!/bin/sh
#

SCRIPT=`readlink -f $0`
SCRIPTDIR=`dirname $SCRIPT`

java -Xms512m -Xmx1600m -jar $SCRIPTDIR/jars/Panoply.jar "$@"

