#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y atlas-devel
ARGS=("$@")

PLAT=${ARGS[0]}
PROJECT_NAME=${ARGS[1]}
PYBINS=${ARGS[@]:2}
# Compile wheels
echo "${PYBINS[@]}"
for PYBIN in ${PYBINS[@]};do
  PYBIN="/opt/python/${PYBIN}/bin"
  echo "${PYBIN}"
  "${PYBIN}/pip" install -q -r /io/dev-requirements.txt
  "${PYBIN}/python" -m spacy download en
  "${PYBIN}/pip" wheel /io/ -w /io/wheelhouse/
done
NORM_PROJECT_NAME=${PROJECT_NAME/_/-}
# Bundle external shared libraries into the wheels
for whl in /io/wheelhouse/*.whl; do
    norm_whl=${whl/_/-}
    if [[ $norm_whl == /io/wheelhouse/${NORM_PROJECT_NAME}* ]]; then
      if [[ $whl != *none-any.whl ]]; then
#       if not an os-dependent build, the following repair will complain
        auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
      fi
    else
      rm $whl
    fi
done

#unfortunately, the project name was changed in history, which causes travis deploy rejection. Has to be fixed ugly.
for whl in in /io/wheelhouse/*.whl; do
  norm_whl=${whl/_/-}
  mv $whl $norm_whl
done

for whl in in /io/wheelhouse/*.gz; do
  norm_whl=${whl/_/-}
  mv $whl $norm_whl
done

echo `pwd`

ls / -l

ls /io/wheelhouse -l

# Install packages and test
for PYBIN in ${PYBINS[@]}; do
    PYBIN="/opt/python/${PYBIN}/bin"
    "${PYBIN}/pip" install ${PROJECT_NAME} --no-index -f /io/wheelhouse
#    (cd "$HOME";ls -l; "${PYBIN}/nosetests" ${PROJECT_NAME})
done