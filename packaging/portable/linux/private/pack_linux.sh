#!/usr/bin/env bash

MANYLINUX_VERSION=$1      # This means that downloaded python version has been built in CentOS 7. See https://www.python.org/dev/peps/pep-0599/ for details.
PYTHON_VERSION=$2         # Three digits. Before build check the latest available versions on https://github.com/niess/python-appimage/tags
MAXATAC_VERSION=$3        # Will be always pulled from GitHub. Doesn't support build from local directory
BEDTOOLS_VERSION=$4
SAMTOOLS_VERSION=$5
PIGZ_VERSION=$6
GIT_CLIENT_VERSION=$7


WORKING_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $WORKING_DIR
echo "Current working directory ${WORKING_DIR}"


echo "Installing required building dependencies through yum"
yum install wget git gcc gcc-c++ make ncurses-devel bzip2-devel xz-devel zlib-devel gettext perl-devel file -y

echo "Downloading linuxdeploy and creating AppDir folder"
wget https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
chmod +x linuxdeploy-x86_64.AppImage
./linuxdeploy-x86_64.AppImage --appimage-extract-and-run --appdir AppDir

echo "Packaging bedtools"
BEDTOOLS_URL="https://github.com/arq5x/bedtools2/releases/download/v${BEDTOOLS_VERSION}/bedtools.static.binary"
wget $BEDTOOLS_URL && mv bedtools.static.binary bedtools && chmod +x bedtools
./linuxdeploy-x86_64.AppImage --appimage-extract-and-run --executable ./bedtools --appdir ./AppDir
rm -rf bedtools

echo "Packaging htslib and samtools"
HTSLIB_URL="https://github.com/samtools/htslib/releases/download/${SAMTOOLS_VERSION}/htslib-${SAMTOOLS_VERSION}.tar.bz2"
wget -q -O - $HTSLIB_URL | tar -jxv
cd htslib-${SAMTOOLS_VERSION} && ./configure --prefix=/usr && make -j 2 && make install DESTDIR=../AppDir && cd ..
SAMTOOLS_URL="https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2"
wget -q -O - $SAMTOOLS_URL | tar -jxv
cd samtools-${SAMTOOLS_VERSION} &&  ./configure --prefix=/usr &&  make -j 2 &&  make install DESTDIR=../AppDir && cd .. 
rm -rf htslib-${SAMTOOLS_VERSION} samtools-${SAMTOOLS_VERSION}

echo "Packaging the latest bedGraphToBigWig"
BG_TO_BW_URL="http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bedGraphToBigWig"
wget $BG_TO_BW_URL && chmod +x bedGraphToBigWig
./linuxdeploy-x86_64.AppImage --appimage-extract-and-run --executable ./bedGraphToBigWig --appdir ./AppDir
rm -rf bedGraphToBigWig

echo "Packaging pigz"
PIGZ_URL="https://github.com/madler/pigz/archive/refs/tags/v${PIGZ_VERSION}.tar.gz"
wget -q -O - $PIGZ_URL | tar -zxv
cd pigz-${PIGZ_VERSION} && make -j 2 && cd ..
./linuxdeploy-x86_64.AppImage --appimage-extract-and-run --executable ./pigz-${PIGZ_VERSION}/pigz --appdir ./AppDir
rm -rf pigz-${PIGZ_VERSION}

echo "Packaging git client"
GIT_CLIENT_URL="https://mirrors.edge.kernel.org/pub/software/scm/git/git-${GIT_CLIENT_VERSION}.tar.gz"
wget --no-check-certificate -q -O - $GIT_CLIENT_URL | tar -zxv
cd git-${GIT_CLIENT_VERSION} && ./configure --prefix=/usr && make -j 2 && make install DESTDIR=../AppDir && cd ..
rm -rf git-${GIT_CLIENT_VERSION}

echo "Packaging Python"
SHORT_PYTHON_VERSION=$(echo ${PYTHON_VERSION} | cut -d "." -f 1,2)
SHORT_PYTHON_VERSION_MONO=$(echo ${PYTHON_VERSION} | cut -d "." -f 1,2 | tr -d ".")
PYTHON_URL="https://github.com/niess/python-appimage/releases/download/python${SHORT_PYTHON_VERSION}/python${PYTHON_VERSION}-cp${SHORT_PYTHON_VERSION_MONO}-cp${SHORT_PYTHON_VERSION_MONO}-manylinux${MANYLINUX_VERSION}_x86_64.AppImage"
PYTHON_APPIMAGE="python${PYTHON_VERSION}-cp${SHORT_PYTHON_VERSION_MONO}-cp${SHORT_PYTHON_VERSION_MONO}-manylinux${MANYLINUX_VERSION}_x86_64.AppImage"
wget $PYTHON_URL
chmod +x $PYTHON_APPIMAGE
./$PYTHON_APPIMAGE --appimage-extract
mv squashfs-root python${PYTHON_VERSION}
rm $PYTHON_APPIMAGE
mv python${PYTHON_VERSION}/opt ./AppDir
mv python${PYTHON_VERSION}/usr/bin/* ./AppDir/usr/bin/
mv python${PYTHON_VERSION}/usr/lib/* ./AppDir/usr/lib
mv python${PYTHON_VERSION}/usr/share/tcltk ./AppDir/usr/share
rm -rf python${PYTHON_VERSION}

echo "Packaging maxATAC"
MAXATAC_URL="https://github.com/MiraldiLab/maxATAC"
git clone $MAXATAC_URL && cd maxATAC
git checkout $MAXATAC_VERSION
../AppDir/usr/bin/python${SHORT_PYTHON_VERSION} -m pip install . --constraint ./packaging/constraints/constraints-${SHORT_PYTHON_VERSION}.txt && cd ..
rm -rf maxATAC

echo "Simlinking maxatac"
cd ./AppDir
ln -s ./usr/bin/maxatac AppRun
cp ../maxatac.desktop .
cp ../maxatac.svg .
cd ..

echo "Converting to AppImage"
wget "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
chmod +x appimagetool-x86_64.AppImage
./appimagetool-x86_64.AppImage --appimage-extract-and-run ./AppDir

echo "Cleaning up"
rm linuxdeploy-x86_64.AppImage
rm appimagetool-x86_64.AppImage