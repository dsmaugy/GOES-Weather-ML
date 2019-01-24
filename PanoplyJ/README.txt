
PANOPLY

  panoply \PAN-uh-plee\, noun: 1. A splendid or impressive array. [...]


INTRODUCTION

  Panoply is a Java application that allows the user to make plots of data from netCDF, HDF,
  and GRIB datasets. Although its strength is in making longitude-latitude (map) plots,
  it can also make other types of georeferenced plots, including keograms and Hovmoeller
  diagrams; general 2D color plots; and line plots. Data arrays may be "sliced" from larger
  multidimensional arrays (aka, variables). A large selection of global map projections is
  available for lon-lat figures. Plot images may be saved to disk in bitmap graphic formats
  or in PDF.

  Panoply requires that a Java 8 (or later) Runtime Environment be installed on your computer.
  The package manager on some Linux computers will install Java for you if it is not already
  installed. On other computers, you will need to download and install a JRE from Java.com.


DOWNLOADING

  The current version of Panoply, along with other information about the application, may
  always be found at
  https://www.giss.nasa.gov/tools/panoply/


INSTALLING AND RUNNING THE "STANDARD" PANOPLY PACKAGE

  The complete Panoply "Generic" package should, after uncompression, includes:
  
  - Panoply launcher in a shell command file called "panoply.sh".
  - Application code files in a folder (sub-directory) called "jars".
  - This README file.
  
  On most platforms, you may be able to launch Panoply by double-clicking on the Panoply.jar
  icon on the desktop. However, doing so means that Panoply may not be allocated enough
  memory when it is started, in which case it will likely not work correctly.

  Instead, you should run Panoply from the shell command line; 'cd' into the directory where
  the above files are located and then type:
  
  ./panoply.sh

  This should execute a command in the shell file which starts Panoply and requests that it
  be allocated 1 GB of memory. You can increase the memory allocation by editing the
  panoply.sh file and increasing the value specified in the -Xmx option.


JAR FILES

  The sub-directory called jars _must_ remain in the same directory as the panoply.sh
  script, and all the "jar" files it holds must remain in the jars sub-directory. These files
  contain the Panoply application code and (re)moving any of them will break Panoply.


OTHER DOCUMENTATION

  More details about Panoply are available at:
  
    https://www.giss.nasa.gov/tools/panoply/
  
  There are a few additional installation notes at
  
    https://www.giss.nasa.gov/tools/panoply/download.html

  We note in particular that the TGZ archive of the "generic" Panoply was created using a
  BSD tar, and that uncompress the archive using a GNU tar will print several unknown-keyword
  warning messages. The uncompressed package should nevertheless be fine.


COLOR TABLES AND CONTINENTS FILES

  Beginning with version 4, Panoply's "standard" selection of color tables and map overlays
  is built into the application. Additional color tables and map overlays may be opened for
  a single session or added to your favorites library for continued use. See the Panoply
  website for for such optional support files.


CONTACT

  Panoply was developed at the

  NASA Goddard Institute for Space Studies
  2880 Broadway, New York, NY 10025 USA

  Please send bug reports, etc., to Dr. Robert Schmunk at robert.b.schmunk@nasa.gov.

ACKNOWLEDGMENTS

  Panoply uses Java classes and libraries written by several third-party organizations. A
  complete list, with links to pertinent websites containing license information and source
  code, may be accessed via Panoply's Help menu or on the Panoply website.
